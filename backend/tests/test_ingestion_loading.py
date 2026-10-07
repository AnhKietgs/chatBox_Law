import logging

from app.services import ingestion


def test_hybrid_pdf_ocr_only_runs_on_sparse_pages(monkeypatch, caplog):
    long_page = "Điều 1. " + ("Nội dung có sẵn. " * 12)
    monkeypatch.setattr(ingestion, "_read_pdf_page_texts", lambda _: [long_page, "trang scan"])
    monkeypatch.setattr(ingestion, "_render_pdf_page_png", lambda _, page: f"png-{page}".encode())
    monkeypatch.setattr(ingestion, "extract_with_ocr", lambda image: "Điều 2. Nội dung OCR" if image == b"png-1" else "")

    with caplog.at_level(logging.INFO, logger="app.services.ingestion"):
        text = ingestion._extract_hybrid_pdf(b"pdf", "hybrid.pdf")

    assert text == f"{long_page}\n\nĐiều 2. Nội dung OCR"
    assert "page=2 native_len=10 ocr_len=20" in caplog.text
    assert "len(text)=" in caplog.text


def test_document_load_always_logs_text_length(monkeypatch, caplog):
    monkeypatch.setattr(ingestion, "_extract_hybrid_pdf", lambda *_: None)
    monkeypatch.setattr(ingestion, "_extract_docling_document", lambda *_: "Điều 1. Văn bản thử nghiệm")

    with caplog.at_level(logging.INFO, logger="app.services.ingestion"):
        text = ingestion.extract_with_docling(b"doc", "source.docx")

    assert text.startswith("Điều 1")
    assert "Document load complete: filename=source.docx len(text)=26 source=docling" in caplog.text
