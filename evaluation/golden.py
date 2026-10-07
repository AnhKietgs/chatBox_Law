"""Strict contract for the expert-maintained legal RAG evaluation set.

The file deliberately carries legal ground truth, rather than model-generated
answers.  A release evaluation is invalid until every record has a named
reviewer and a review date.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CitationLocator:
    document_code: str
    article: str
    clause: str | None
    point: str | None


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    as_of_date: str | None
    expected_status: str
    reference_answer: str
    expected_citations: tuple[CitationLocator, ...]
    reviewed_by: str
    reviewed_at: str
    notes: str


def _required(record: dict[str, Any], field: str, line_number: int) -> Any:
    value = record.get(field)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Dòng {line_number}: thiếu trường bắt buộc '{field}'")
    return value


def _parse_date(value: str, field: str, line_number: int) -> None:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Dòng {line_number}: '{field}' phải có định dạng YYYY-MM-DD") from exc


def _citation(value: Any, line_number: int) -> CitationLocator:
    if not isinstance(value, dict):
        raise ValueError(f"Dòng {line_number}: mỗi expected_citations phải là object")
    document_code = _required(value, "document_code", line_number)
    article = _required(value, "article", line_number)
    if not isinstance(document_code, str) or not isinstance(article, str):
        raise ValueError(f"Dòng {line_number}: document_code và article phải là chuỗi")
    clause = value.get("clause")
    point = value.get("point")
    if clause is not None and not isinstance(clause, str):
        raise ValueError(f"Dòng {line_number}: clause phải là chuỗi hoặc null")
    if point is not None and not isinstance(point, str):
        raise ValueError(f"Dòng {line_number}: point phải là chuỗi hoặc null")
    return CitationLocator(document_code=document_code, article=article, clause=clause, point=point)


def load_golden_cases(path: str | Path, *, required_count: int | None = None) -> list[GoldenCase]:
    """Load and validate JSONL records without making any network request."""
    source = Path(path)
    if not source.exists():
        raise ValueError(
            f"Không tìm thấy {source}. Sao chép golden_questions.example.jsonl, "
            "sau đó để chuyên gia pháp lý duyệt dữ liệu vàng."
        )
    cases: list[GoldenCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Dòng {line_number}: JSON không hợp lệ: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Dòng {line_number}: mỗi dòng phải là JSON object")
        case_id = _required(record, "id", line_number)
        question = _required(record, "question", line_number)
        expected_status = _required(record, "expected_status", line_number)
        reference_answer = record.get("reference_answer", "")
        reviewed_by = _required(record, "reviewed_by", line_number)
        reviewed_at = _required(record, "reviewed_at", line_number)
        notes = record.get("notes", "")
        if not isinstance(case_id, str) or not isinstance(question, str):
            raise ValueError(f"Dòng {line_number}: id và question phải là chuỗi")
        if case_id in seen_ids:
            raise ValueError(f"Dòng {line_number}: id '{case_id}' bị trùng")
        seen_ids.add(case_id)
        if expected_status not in {"grounded", "abstained"}:
            raise ValueError(f"Dòng {line_number}: expected_status phải là grounded hoặc abstained")
        if not isinstance(reference_answer, str):
            raise ValueError(f"Dòng {line_number}: reference_answer phải là chuỗi")
        if expected_status == "grounded" and not reference_answer.strip():
            raise ValueError(f"Dòng {line_number}: ca grounded phải có reference_answer do chuyên gia duyệt")
        as_of_date = record.get("as_of_date")
        if as_of_date is not None:
            if not isinstance(as_of_date, str):
                raise ValueError(f"Dòng {line_number}: as_of_date phải là chuỗi hoặc null")
            _parse_date(as_of_date, "as_of_date", line_number)
        _parse_date(str(reviewed_at), "reviewed_at", line_number)
        raw_citations = record.get("expected_citations")
        if not isinstance(raw_citations, list):
            raise ValueError(f"Dòng {line_number}: expected_citations phải là mảng")
        citations = tuple(_citation(value, line_number) for value in raw_citations)
        if expected_status == "grounded" and not citations:
            raise ValueError(f"Dòng {line_number}: ca grounded phải có ít nhất một căn cứ pháp lý")
        if expected_status == "abstained" and citations:
            raise ValueError(f"Dòng {line_number}: ca abstained không được có căn cứ")
        if "LEGAL_EXPERT" in reviewed_by or str(reviewed_at).startswith("YYYY"):
            raise ValueError(f"Dòng {line_number}: phải thay placeholder reviewer bằng chuyên gia pháp lý thực tế")
        cases.append(GoldenCase(
            id=case_id, question=question.strip(), as_of_date=as_of_date,
            expected_status=expected_status, reference_answer=reference_answer.strip(),
            expected_citations=citations, reviewed_by=str(reviewed_by).strip(),
            reviewed_at=str(reviewed_at), notes=str(notes),
        ))
    if not cases:
        raise ValueError("Golden set đang rỗng")
    if required_count is not None and len(cases) != required_count:
        raise ValueError(f"Release gate cần đúng {required_count} câu đã duyệt, hiện có {len(cases)}")
    return cases

