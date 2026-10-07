from datetime import date
import hashlib
from pathlib import Path
from uuid import UUID, uuid4
from urllib.parse import urlparse
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from ..celery_app import celery
from ..config import get_settings
from ..database import get_db
from ..models import IngestionJob, JobStatus, LegalDocument, LegalProvision, LegalVersion, VersionStatus
from ..schemas import ChatQuery, EvaluationTrace, JobSummary, ProvisionPage, ProvisionSummary, PublishRequest, ReviewSummary, TokenRequest, TokenResponse, VersionSummary
from ..security import issue_token, require_admin
from ..services.ingestion import publish_version, reindex_version
from ..services.storage import ObjectStorage
from ..services.vector_store import HybridVectorStore
from ..services.document_metadata import normalize_category, normalize_domain
from ..services.answering import GroundedAnswerService, abstain
from ..services.query_augmentation import augment_for_retrieval, detect_mixed_legal_domains
from ..services.retrieval import LegalRetriever
from ..services.prompt_security import scan_document_content
from ..rate_limit import enforce_admin_login_rate_limit
from ..versioning import windows_overlap

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


def evaluation_context(item: object) -> tuple[str, str]:
    """Format the exact provision supplied by retrieval for evaluator audit."""
    provision = item.provision
    document = item.document
    version = item.version
    locator = (
        f"{document.code} | {version.version_label} | Điều {provision.article_no}"
        f", Khoản {provision.clause_no or '-'}, Điểm {provision.point_label or '-'}"
    )
    heading = item.article_heading or provision.heading
    text = f"{locator}\n{heading + chr(10) if heading else ''}{provision.content}"
    return text, locator


def provision_summary(row: LegalProvision) -> ProvisionSummary:
    return ProvisionSummary(id=row.id, article_no=row.article_no, clause_no=row.clause_no, point_label=row.point_label, heading=row.heading, content=row.content, ordinal=row.ordinal, chunk_index=row.chunk_index)


def structure_hash(rows: list[LegalProvision]) -> str:
    """Hash every locator and excerpt that an admin is about to approve."""
    material = "\n".join(
        "\x1f".join((row.article_no, row.clause_no or "", row.point_label or "", row.heading or "", row.content))
        for row in rows
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@router.post("/token", response_model=TokenResponse)
def login(payload: TokenRequest, request: Request) -> TokenResponse:
    enforce_admin_login_rate_limit(request)
    settings = get_settings()
    if payload.email != settings.admin_email or payload.password != settings.admin_password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sai thông tin quản trị viên")
    return TokenResponse(access_token=issue_token(payload.email))


@router.post("/evaluation/query", response_model=EvaluationTrace)
async def evaluation_query(
    payload: ChatQuery,
    _: str = Depends(require_admin),
    db: Session = Depends(get_db),
) -> EvaluationTrace:
    """Run production RAG with the complete reranked context for RAGAS.

    This endpoint requires the admin JWT. Public chat exposes only citations
    included in an answer, while an evaluator needs every provision that was
    actually supplied to the answer model in order to score retrieval quality.
    """
    applied_date = payload.as_of_date or date.today()
    mixed_domains = detect_mixed_legal_domains(payload.question)
    if mixed_domains.is_mixed:
        answer = abstain(applied_date)
        answer.answer = "Câu hỏi đa lĩnh vực cần được tách trước khi đánh giá."
        return EvaluationTrace(answer=answer, retrieved_contexts=[], retrieved_locators=[])
    augmentation = await augment_for_retrieval(payload.question)
    retrieval = LegalRetriever(db).retrieve_with_metrics(
        payload.question,
        applied_date,
        inject_limit=get_settings().retrieval_inject_limit,
        standalone_query=payload.question,
        additional_queries=augmentation.query_variants,
        hyde_query=augmentation.hypothetical_document,
    )
    generation = await GroundedAnswerService().generate(
        payload.question,
        retrieval.provisions,
        applied_date,
        fallback_minimum_score=retrieval.minimum_score,
        retrieval_query=payload.question,
    )
    contexts_and_locators = [evaluation_context(item) for item in retrieval.provisions]
    return EvaluationTrace(
        answer=generation.answer,
        retrieved_contexts=[text for text, _ in contexts_and_locators],
        retrieved_locators=[locator for _, locator in contexts_and_locators],
    )


@router.post("/versions", response_model=JobSummary, status_code=status.HTTP_202_ACCEPTED)
def upload_version(
    document_code: str = Form(...), title: str = Form(...), version_label: str = Form(...), official_url: str = Form(...),
    effective_from: date = Form(...), effective_to: str | None = Form(None),
    domain: str = Form("general"), category: str = Form("general"), file: UploadFile = File(...),
    _: str = Depends(require_admin), db: Session = Depends(get_db),
) -> JobSummary:
    try:
        normalized_domain = normalize_domain(domain)
        normalized_category = normalize_category(category)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(status_code=422, detail="Chỉ chấp nhận PDF hoặc DOCX")
    parsed_effective_to: date | None = None
    if effective_to:
        try:
            parsed_effective_to = date.fromisoformat(effective_to)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Hiệu lực đến phải có định dạng YYYY-MM-DD") from exc
    if parsed_effective_to and parsed_effective_to < effective_from:
        raise HTTPException(status_code=422, detail="Ngày hết hiệu lực phải sau ngày bắt đầu hiệu lực")
    parsed_url = urlparse(official_url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise HTTPException(status_code=422, detail="URL nguồn chính thức phải dùng HTTPS hợp lệ")
    existing = db.scalar(select(LegalDocument).where(LegalDocument.code == document_code))
    document = existing or LegalDocument(
        code=document_code, title=title, domain=normalized_domain, category=normalized_category,
    )
    if not existing:
        db.add(document)
        db.flush()
    else:
        document.title = title
        document.domain = normalized_domain
        document.category = normalized_category
    existing_versions = db.scalars(select(LegalVersion).where(LegalVersion.document_id == document.id, LegalVersion.status == VersionStatus.published)).all()
    invalid_overlap = any(
        windows_overlap(effective_from, parsed_effective_to, version.effective_from, version.effective_to)
        and (version.effective_to is not None or version.effective_from >= effective_from)
        for version in existing_versions
    )
    if invalid_overlap:
        raise HTTPException(status_code=422, detail="Khoảng hiệu lực chồng lấn với phiên bản đã xuất bản")
    content = file.file.read()
    if not content or len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="Tệp rỗng hoặc vượt quá 50 MB")
    # Fast path for visible/plain document content. The worker repeats the
    # scan after Docling/OCR so compressed PDF/DOCX text cannot bypass it.
    raw_findings = scan_document_content(content.decode("utf-8", errors="ignore"))
    if raw_findings:
        rule_ids = ", ".join(finding.rule_id for finding in raw_findings)
        raise HTTPException(
            status_code=422,
            detail=f"Tệp bị từ chối do có dấu hiệu prompt injection ({rule_ids}).",
        )
    existing_version = db.scalar(select(LegalVersion).where(
        LegalVersion.document_id == document.id,
        LegalVersion.version_label == version_label,
    ))
    if existing_version and existing_version.status not in {VersionStatus.failed, VersionStatus.rejected}:
        raise HTTPException(status_code=409, detail="Phiên bản này đã tồn tại. Hãy dùng nhãn phiên bản khác.")
    version_id = existing_version.id if existing_version else uuid4()
    object_key = f"sources/{version_id}/{file.filename}"
    ObjectStorage().put(object_key, content, file.content_type or "application/octet-stream")
    if existing_version:
        version = existing_version
        version.official_url = official_url
        version.effective_from = effective_from
        version.effective_to = parsed_effective_to
        version.source_object_key = object_key
        version.parsed_object_key = None
        version.checksum = None
        version.failure_reason = None
        version.status = VersionStatus.processing
    else:
        version = LegalVersion(id=version_id, document_id=document.id, version_label=version_label, official_url=official_url, effective_from=effective_from, effective_to=parsed_effective_to, source_object_key=object_key)
        db.add(version)
    db.flush()
    job = IngestionJob(version_id=version.id, status=JobStatus.queued, progress=0, message="Đã xếp hàng xử lý")
    db.add(job)
    db.commit()
    task = celery.send_task("lawrag.process_version", args=[str(job.id), file.filename or "source.pdf"])
    job.celery_task_id = task.id
    db.commit()
    db.refresh(job)
    return JobSummary(id=job.id, version_id=version.id, status=job.status.value, progress=job.progress, message=job.message, created_at=job.created_at)


@router.get("/jobs/{job_id}", response_model=JobSummary)
def get_job(job_id: UUID, _: str = Depends(require_admin), db: Session = Depends(get_db)) -> JobSummary:
    job = db.get(IngestionJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    return JobSummary(id=job.id, version_id=job.version_id, status=job.status.value, progress=job.progress, message=job.message, created_at=job.created_at)


@router.get("/versions", response_model=list[VersionSummary])
def list_versions(_: str = Depends(require_admin), db: Session = Depends(get_db)) -> list[VersionSummary]:
    rows = db.execute(select(LegalVersion, LegalDocument).join(LegalDocument).order_by(LegalVersion.created_at.desc())).all()
    return [VersionSummary(id=v.id, document_code=d.code, document_title=d.title, version_label=v.version_label, effective_from=v.effective_from, effective_to=v.effective_to, status=v.status.value, official_url=v.official_url, domain=d.domain, category=d.category) for v, d in rows]


@router.get("/versions/{version_id}/provisions", response_model=list[ProvisionSummary])
def list_provisions(version_id: UUID, _: str = Depends(require_admin), db: Session = Depends(get_db)) -> list[ProvisionSummary]:
    version = db.get(LegalVersion, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên bản")
    rows = db.scalars(select(LegalProvision).where(LegalProvision.version_id == version_id).order_by(LegalProvision.ordinal)).all()
    return [provision_summary(row) for row in rows]


@router.get("/versions/{version_id}/chunks", response_model=ProvisionPage)
def list_published_chunks(
    version_id: UUID,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    _: str = Depends(require_admin),
    db: Session = Depends(get_db),
) -> ProvisionPage:
    """Show exactly the chunks indexed from a published version, page by page.

    Keeping this separate from the pre-publish review endpoint prevents the
    admin UI from downloading thousands of chunks at once for a long code.
    """
    version = db.get(LegalVersion, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên bản")
    if version.status != VersionStatus.published:
        raise HTTPException(status_code=422, detail="Chỉ xem chunk của phiên bản đã xuất bản")

    where = LegalProvision.version_id == version_id
    total = db.scalar(select(func.count()).select_from(LegalProvision).where(where)) or 0
    rows = db.scalars(
        select(LegalProvision)
        .where(where)
        .order_by(LegalProvision.ordinal)
        .offset(offset)
        .limit(limit)
    ).all()
    return ProvisionPage(
        version_id=version_id,
        total=total,
        offset=offset,
        limit=limit,
        provisions=[provision_summary(row) for row in rows],
    )


@router.get("/versions/{version_id}/review", response_model=ReviewSummary)
def get_review(version_id: UUID, _: str = Depends(require_admin), db: Session = Depends(get_db)) -> ReviewSummary:
    version = db.get(LegalVersion, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên bản")
    rows = db.scalars(select(LegalProvision).where(LegalProvision.version_id == version_id).order_by(LegalProvision.ordinal)).all()
    if not rows:
        raise HTTPException(status_code=422, detail="Phiên bản chưa có cấu trúc để duyệt")
    return ReviewSummary(version_id=version_id, provision_count=len(rows), structure_hash=structure_hash(rows), provisions=[provision_summary(row) for row in rows])


@router.post("/versions/{version_id}/publish")
def publish(version_id: UUID, payload: PublishRequest, _: str = Depends(require_admin), db: Session = Depends(get_db)) -> dict:
    version = db.get(LegalVersion, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên bản")
    rows = db.scalars(select(LegalProvision).where(LegalProvision.version_id == version_id).order_by(LegalProvision.ordinal)).all()
    if not rows or payload.reviewed_structure_hash != structure_hash(rows):
        raise HTTPException(status_code=422, detail="Cấu trúc đã thay đổi hoặc chưa được kiểm tra. Hãy mở lại bước kiểm tra trước khi xuất bản.")
    try:
        count = publish_version(db, version)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"version_id": str(version.id), "indexed_provisions": count, "status": version.status.value}


@router.post("/versions/{version_id}/reindex")
def reindex(version_id: UUID, _: str = Depends(require_admin), db: Session = Depends(get_db)) -> dict:
    """Re-embed và upsert lại Qdrant cho version đã publish, không cần re-upload PDF.

    Dùng khi text embedding thay đổi (ví dụ: thêm heading Điều vào khoản con)
    nhưng cấu trúc Điều/Khoản/Điểm trong PostgreSQL vẫn đúng.
    """
    version = db.get(LegalVersion, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên bản")
    if version.status != VersionStatus.published:
        raise HTTPException(status_code=422, detail="Chỉ có thể reindex phiên bản đang xuất bản")
    try:
        count = reindex_version(db, version)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reindex thất bại: {exc}") from exc
    return {"version_id": str(version.id), "reindexed_provisions": count, "status": version.status.value}


@router.post("/versions/{version_id}/withdraw")
def withdraw(version_id: UUID, _: str = Depends(require_admin), db: Session = Depends(get_db)) -> dict:
    """Remove a published version from retrieval without deleting its audit trail."""
    version = db.get(LegalVersion, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên bản")
    if version.status != VersionStatus.published:
        raise HTTPException(status_code=422, detail="Chỉ có thể thu hồi phiên bản đang xuất bản")
    HybridVectorStore().set_version_status(version.id, "REJECTED")
    version.status = VersionStatus.rejected
    version.failure_reason = "Đã thu hồi để kiểm tra lại cấu trúc Điều/Khoản/Điểm"
    db.commit()
    return {"version_id": str(version.id), "status": version.status.value}
