from __future__ import annotations
import hashlib
import json
import logging
import math
import random
from datetime import timedelta
from collections.abc import Callable
from uuid import UUID
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from ..config import get_settings
from ..models import IngestionJob, JobStatus, LegalProvision, LegalVersion, VersionStatus
from .legal_parser import ParsedProvision, estimate_tokens, parse_legal_text
from .prompt_security import scan_document_content
from .storage import ObjectStorage
from .vector_store import HybridVectorStore

logger = logging.getLogger(__name__)
PDF_PAGE_TEXT_THRESHOLD = 100


def _legal_locator(provision: ParsedProvision) -> str:
    locator = f"Điều {provision.article_no}"
    if provision.clause_no:
        locator += f", Khoản {provision.clause_no}"
    if provision.point_label:
        locator += f", Điểm {provision.point_label}"
    return locator


def log_chunk_audit(version_id: UUID, provisions: list[ParsedProvision]) -> None:
    """Emit reproducible, compact ingest diagnostics for human review."""
    if not provisions:
        logger.warning("Chunk audit: version_id=%s has no chunks", version_id)
        return
    settings = get_settings()
    estimates = sorted(estimate_tokens(item.content) for item in provisions)
    p50 = estimates[(len(estimates) - 1) // 2]
    p95 = estimates[math.ceil(len(estimates) * 0.95) - 1]
    oversized = sum(score > settings.chunk_max_tokens for score in estimates)
    split_chunks = sum(item.chunk_index > 1 for item in provisions)
    logger.info(
        "Chunk audit: version_id=%s chunks=%d token_estimate[min=%d p50=%d p95=%d max=%d] "
        "split_chunks=%d over_limit=%d policy=structure-aware; overlap=%d only-for-split-parts",
        version_id, len(provisions), estimates[0], p50, p95, estimates[-1], split_chunks, oversized,
        settings.chunk_split_overlap_tokens,
    )
    sample_size = min(settings.chunk_audit_sample_size, len(provisions))
    if not sample_size:
        return
    # A version-id seed makes an audit reproducible instead of log-noisy.
    sample_indexes = sorted(random.Random(str(version_id)).sample(range(len(provisions)), sample_size))
    for sample_number, index in enumerate(sample_indexes, start=1):
        item = provisions[index]
        excerpt = " ".join(item.content.split())[:200]
        logger.info(
            "Chunk audit sample %d/%d: ordinal=%d locator=%s part=%d tokens=%d excerpt=%r",
            sample_number, sample_size, item.ordinal, _legal_locator(item), item.chunk_index,
            estimate_tokens(item.content), excerpt,
        )


def _extract_docling_document(source: bytes, filename: str) -> str:
    """Return Docling Markdown for one document or one temporary PDF page."""
    from docling.document_converter import DocumentConverter
    import tempfile
    from pathlib import Path

    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
        temporary.write(source)
        temporary.flush()
        result = DocumentConverter().convert(temporary.name)
        return result.document.export_to_markdown()


def _read_pdf_page_texts(source: bytes) -> list[str]:
    """Inspect the embedded text for every PDF page without running OCR."""
    from io import BytesIO
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(source))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def _render_pdf_page_png(source: bytes, page_index: int) -> bytes:
    """Render one PDF page into a readable PNG for RapidOCR/PaddleOCR."""
    from io import BytesIO
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(source)
    try:
        page = document[page_index]
        try:
            bitmap = page.render(scale=2.0)
            image = bitmap.to_pil()
            try:
                output = BytesIO()
                image.save(output, format="PNG")
                return output.getvalue()
            finally:
                image.close()
        finally:
            page.close()
    finally:
        document.close()


def _extract_hybrid_pdf(source: bytes, filename: str) -> str | None:
    """Extract PDF page-by-page only when at least one page needs OCR.

    A whole-document character threshold loses scanned pages in a hybrid PDF:
    the text on ordinary pages makes the total look healthy.  Instead, inspect
    each page; pages below the threshold are rendered and OCRed while text
    pages retain their native extraction.  ``None`` means no sparse page was
    found and callers should use the higher-fidelity full Docling conversion.
    """
    try:
        page_texts = _read_pdf_page_texts(source)
    except Exception as exc:
        logger.warning("PDF page inspection failed for %s; using full Docling: %s", filename, exc)
        return None

    sparse_indexes = [index for index, text in enumerate(page_texts) if len(text) < PDF_PAGE_TEXT_THRESHOLD]
    logger.info(
        "PDF page text inspection: filename=%s pages=%d chars_per_page=%s sparse_pages=%s threshold=%d",
        filename,
        len(page_texts),
        [len(text) for text in page_texts],
        [index + 1 for index in sparse_indexes],
        PDF_PAGE_TEXT_THRESHOLD,
    )
    if not sparse_indexes:
        return None

    merged_pages = list(page_texts)
    for page_index in sparse_indexes:
        page_no = page_index + 1
        try:
            ocr_text = extract_with_ocr(_render_pdf_page_png(source, page_index)).strip()
        except Exception as exc:
            # Preserve any short embedded text instead of silently deleting a
            # page when both OCR engines fail. The reviewer can reject it.
            logger.warning("OCR failed for %s page=%d: %s", filename, page_no, exc)
            ocr_text = ""
        if ocr_text:
            merged_pages[page_index] = ocr_text
        logger.info(
            "PDF page loaded: filename=%s page=%d native_len=%d ocr_len=%d final_len=%d",
            filename,
            page_no,
            len(page_texts[page_index]),
            len(ocr_text),
            len(merged_pages[page_index]),
        )

    text = "\n\n".join(page for page in merged_pages if page.strip())
    logger.info("Hybrid PDF load complete: filename=%s len(text)=%d ocr_pages=%s", filename, len(text), [index + 1 for index in sparse_indexes])
    return text


def extract_with_docling(source: bytes, filename: str) -> str:
    """Extract text while OCRing only sparse pages of a hybrid PDF."""
    from pathlib import Path

    if Path(filename).suffix.lower() == ".pdf":
        hybrid_text = _extract_hybrid_pdf(source, filename)
        if hybrid_text is not None:
            logger.info("Document load complete: filename=%s len(text)=%d source=hybrid_pdf", filename, len(hybrid_text))
            if hybrid_text.strip():
                return hybrid_text

    markdown = _extract_docling_document(source, filename)
    logger.info("Document load complete: filename=%s len(text)=%d source=docling", filename, len(markdown))
    if markdown.strip():
        return markdown
    text = extract_with_ocr(source)
    logger.info("Document load complete: filename=%s len(text)=%d source=ocr_fallback", filename, len(text))
    return text


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
        logger.info("Ingestion text loaded: version_id=%s filename=%s len(text)=%d", version.id, filename, len(text))
        findings = scan_document_content(text)
        if findings:
            rule_ids = ", ".join(finding.rule_id for finding in findings)
            logger.warning(
                "Document security scan quarantined version_id=%s filename=%s rules=%s",
                version.id, filename, rule_ids,
            )
            # Do not persist parsed text or create Qdrant vectors. The original
            # file remains in restricted object storage for the admin audit.
            raise ValueError(
                "Tệp bị cách ly do có dấu hiệu prompt injection "
                f"({rule_ids}); không được lập chỉ mục."
            )
        settings = get_settings()
        provisions = parse_legal_text(text, settings.chunk_max_tokens, settings.chunk_split_overlap_tokens)
        log_chunk_audit(version.id, provisions)
        job.progress, job.message = 45, f"Đã nhận diện {len(provisions)} đơn vị Điều/Khoản/Điểm"
        db.execute(delete(LegalProvision).where(LegalProvision.version_id == version.id))
        rows = [LegalProvision(version_id=version.id, article_no=p.article_no, clause_no=p.clause_no, point_label=p.point_label, heading=p.heading, content=p.content, ordinal=p.ordinal, chunk_index=p.chunk_index) for p in provisions]
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


def _build_embedding_records(version: LegalVersion) -> list[tuple[UUID, str, dict]]:
    """Xây dựng danh sách (provision_id, text, payload) để upsert vào Qdrant.

    Text giữ ngắn gọn để BM25 sparse không bị dilute — hybrid recall ưu tiên
    precision của sparse term matching. Việc inject article heading để làm giàu
    ngữ cảnh được thực hiện ở tầng reranker (retrieval.py) nơi nó không ảnh
    hưởng đến recall. Dùng chung cho publish_version() và reindex_version().
    """
    findings = scan_document_content("\n".join(
        f"{provision.heading or ''}\n{provision.content}" for provision in version.provisions
    ))
    if findings:
        rule_ids = ", ".join(finding.rule_id for finding in findings)
        logger.warning("Blocked Qdrant indexing for version_id=%s rules=%s", version.id, rule_ids)
        raise ValueError(f"Không thể index: phát hiện dấu hiệu prompt injection ({rule_ids})")
    payload_base = {
        "version_id": str(version.id),
        "document_code": version.document.code,
        "domain": version.document.domain,
        "category": version.document.category,
        "status": "PUBLISHED",
        "effective_from": version.effective_from.isoformat(),
        "effective_to": version.effective_to.isoformat() if version.effective_to else None,
    }
    records: list[tuple[UUID, str, dict]] = []
    for provision in version.provisions:
        text = f"Điều {provision.article_no}. {provision.heading or ''}\n{provision.content}"
        records.append((provision.id, text, {**payload_base, "chunk_index": provision.chunk_index}))
    return records


def _upsert_records(
    records: list[tuple[UUID, str, dict]],
    label: str,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Chia batch và upsert vào Qdrant. label dùng cho log."""
    vector_store = HybridVectorStore()
    batch_size = get_settings().index_batch_size
    total = len(records)
    for start in range(0, total, batch_size):
        vector_store.upsert_many(records[start:start + batch_size])
        processed = min(start + batch_size, total)
        logger.info("%s: upserted %d/%d provisions", label, processed, total)
        if on_progress:
            on_progress(processed, total)


def reindex_version(db: Session, version: LegalVersion) -> int:
    """Re-embed và upsert lại Qdrant cho version PUBLISHED mà không đổi trạng thái.

    Dùng khi logic tạo embedding text thay đổi nhưng cấu trúc provisions
    trong PostgreSQL vẫn đúng — không cần re-upload PDF.
    """
    records = _build_embedding_records(version)
    _upsert_records(records, f"Reindex version {version.id}")
    db.commit()
    return len(records)


def reindex_version_job(db: Session, job_id: UUID) -> None:
    """Run a published-version reindex as a trackable Celery job.

    Embedding thousands of provisions can take many minutes on CPU. Keeping
    that work outside the HTTP request prevents nginx/browser timeouts while
    exposing batch progress through the existing admin job endpoint.
    """
    job = db.get(IngestionJob, job_id)
    if not job:
        raise ValueError("Không tìm thấy reindex job")
    version = db.get(LegalVersion, job.version_id)
    if not version:
        raise ValueError("Không tìm thấy phiên bản văn bản")
    if version.status != VersionStatus.published:
        raise ValueError("Chỉ có thể reindex phiên bản đang xuất bản")

    job.status = JobStatus.running
    job.progress = 2
    job.message = "Đang chuẩn bị dữ liệu để lập chỉ mục lại"
    db.commit()
    try:
        records = _build_embedding_records(version)
        total = len(records)
        if not total:
            raise ValueError("Phiên bản không có đơn vị pháp lý để lập chỉ mục")

        def update_progress(processed: int, record_count: int) -> None:
            job.progress = min(99, 5 + round(processed / record_count * 94))
            job.message = f"Đang lập chỉ mục lại: {processed}/{record_count} đơn vị"
            db.commit()

        _upsert_records(records, f"Reindex version {version.id}", update_progress)
        job.status = JobStatus.succeeded
        job.progress = 100
        job.message = f"Đã lập chỉ mục lại {total} đơn vị pháp lý"
        db.commit()
    except Exception as exc:
        db.rollback()
        failed_job = db.get(IngestionJob, job_id)
        if failed_job:
            failed_job.status = JobStatus.failed
            failed_job.message = f"Lập chỉ mục lại thất bại: {exc}"
            db.commit()
        logger.exception("Reindex job failed: job_id=%s version_id=%s", job_id, version.id)
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
    records = _build_embedding_records(version)
    for provision in version.provisions:
        provision.vector_id = str(provision.id)
    _upsert_records(records, f"Publishing version {version.id}")
    version.status = VersionStatus.published
    db.commit()
    return len(records)
