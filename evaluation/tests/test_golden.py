from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from golden import load_golden_cases


def test_loads_a_named_expert_approved_grounded_case(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"id":"tm-001","question":"Mức phạt tối đa là bao nhiêu?","as_of_date":"2026-09-21",'
        '"expected_status":"grounded","reference_answer":"Không quá 8% giá trị phần nghĩa vụ bị vi phạm.",'
        '"expected_citations":[{"document_code":"LTM-2005","article":"301","clause":null,"point":null}],'
        '"reviewed_by":"Nguyễn Văn A","reviewed_at":"2026-09-20","notes":"Đã đối chiếu văn bản."}\n',
        encoding="utf-8",
    )
    cases = load_golden_cases(path, required_count=1)
    assert cases[0].expected_citations[0].article == "301"
    assert cases[0].reference_answer.startswith("Không quá 8%")


def test_rejects_unreviewed_placeholder_and_grounded_case_without_reference(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"id":"tm-001","question":"Mức phạt tối đa là bao nhiêu?","expected_status":"grounded",'
        '"reference_answer":"","expected_citations":[{"document_code":"LTM-2005","article":"301"}],'
        '"reviewed_by":"LEGAL_EXPERT_NAME","reviewed_at":"YYYY-MM-DD"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reference_answer"):
        load_golden_cases(path)


def test_rejects_incorrect_release_set_size(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"id":"tm-001","question":"Câu hỏi kiểm thử hợp lệ","expected_status":"abstained",'
        '"reference_answer":"","expected_citations":[],"reviewed_by":"Chuyên gia",'
        '"reviewed_at":"2026-09-20"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="100"):
        load_golden_cases(path, required_count=100)
