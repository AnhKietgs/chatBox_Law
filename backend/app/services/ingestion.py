from __future__ import annotations
import hashlib
import json
import logging
from datetime import timedelta
from uuid import UUID
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from ..config import get_settings
from ..models import IngestionJob, JobStatus, LegalProvision, LegalVersion, VersionStatus
from .legal_parser import parse_legal_text
from .storage import ObjectStorage
from .vector_store import HybridVectorStore

logger = logging.getLogger(__name__)


def extract_with_docling(source: bytes, filename: str) -> str:
    """Extract text with Docling; OCR engines are only used when Docling has no text."""
    from docling.document_converter import DocumentConverter
    import tempfile
    from pathlib import Path
    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
        temporary.write(source)
        temporary.flush()
        result = DocumentConverter().convert(temporary.name)
        markdown = result.document.export_to_markdown()
    if markdown.strip():
        return markdown
    return extract_with_ocr(source)


def extract_with_ocr(source: bytes) -> str:
    """RapidOCR first, PaddleOCR fallback. Raises to keep invalid OCR out of the index."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()
        result, _ = engine(source)
        text = "\n".join(line[1] for line in result or [])
        if text.strip():
            return text
    except Exception:
        pass
    try:
        from paddleocr import PaddleOCR
        engine = PaddleOCR(lang="vi", use_angle_cls=True)
        result = engine.ocr(source, cls=True)
        text = "\n".join(line[1][0] for page in result for line in page or [])
        if text.strip():
            return text
    except Exception as exc:
        raise ValueError("Không thể OCR văn bản bằng RapidOCR hoặc PaddleOCR") from exc
    raise ValueError("OCR không trích xuất được nội dung")


def process_version(db: Session, job_id: UUID, filename: str) -> None:
    job = db.get(IngestionJob, job_id)
    if not job:
        raise ValueError("Không tìm thấy ingestion job")
    version = db.get(LegalVersion, job.version_id)
    if not version:
        raise ValueError("Không tìm thấy phiên bản văn bản")
    job.status, job.progress, job.message = JobStatus.running, 5, "Đang đọc tệp nguồn"
    db.commit()
    try:
        storage = ObjectStorage()
        source = storage.get(version.source_object_key)
        version.checksum = hashlib.sha256(source).hexdigest()
        text = extract_with_docling(source, filename)
        provisions = parse_legal_text(text)
        job.progress, job.message = 45, f"Đã nhận diện {len(provisions)} đơn vị Điều/Khoản/Điểm"
        db.execute(delete(LegalProvision).where(LegalProvision.version_id == version.id))
        rows = [LegalProvision(version_id=version.id, article_no=p.article_no, clause_no=p.clause_no, point_label=p.point_label, heading=p.heading, content=p.content, ordinal=p.ordinal) for p in provisions]
        db.add_all(rows)
        parsed_key = f"parsed/{version.id}.json"
        storage.put(parsed_key, json.dumps([p.__dict__ for p in provisions], ensure_ascii=False).encode(), "application/json")
        version.parsed_object_key = parsed_key
        version.status = VersionStatus.pending_review
        job.status, job.progress, job.message = JobStatus.succeeded, 100, "Đã xử lý; chờ quản trị viên duyệt cấu trúc"
        db.commit()
    except Exception as exc:
        version.status = VersionStatus.failed
        version.failure_reason = str(exc)
        job.status, job.message = JobStatus.failed, str(exc)
        db.commit()
        raise


def publish_version(db: Session, version: LegalVersion) -> int:
    if version.status != VersionStatus.pending_review:
        raise ValueError("Chỉ phiên bản đã xử lý và chờ duyệt mới có thể xuất bản")
    overlapping = db.scalars(select(LegalVersion).where(
        LegalVersion.document_id == version.document_id,
        LegalVersion.status == VersionStatus.published,
        LegalVersion.id != version.id,
    )).all()
    for previous in overlapping:
        if previous.effective_from < version.effective_from and (previous.effective_to is None or previous.effective_to >= version.effective_from):
            previous.effective_to = version.effective_from - timedelta(days=1)
        elif previous.effective_from >= version.effective_from:
            raise ValueError("Phiên bản mới có khoảng hiệu lực chồng lấn với phiên bản đã xuất bản")
    vector_store = HybridVectorStore()
    records: list[tuple[UUID, str, dict]] = []
    for provision in version.provisions:
        text = f"Điều {provision.article_no}. {provision.heading or ''}\n{provision.content}"
        records.append((provision.id, text, {
            "version_id": str(version.id), "status": "PUBLISHED", "effective_from": version.effective_from.isoformat(),
            "effective_to": version.effective_to.isoformat() if version.effective_to else None,
        }))
        provision.vector_id = str(provision.id)
    batch_size = get_settings().index_batch_size
    total = len(records)
    for start in range(0, total, batch_size):
        vector_store.upsert_many(records[start:start + batch_size])
        logger.info("Publishing version %s: indexed %d/%d provisions", version.id, min(start + batch_size, total), total)
    version.status = VersionStatus.published
    db.commit()
    return total
