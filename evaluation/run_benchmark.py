"""Run the deterministic citation/release gate against a running API.

Usage: python evaluation/run_benchmark.py evaluation/golden_questions.jsonl
The template intentionally contains only examples; an admin/legal expert supplies
the final 100 vetted records and their citation locators.
"""
from __future__ import annotations
import json
import os
import sys
import httpx
from golden import load_golden_cases

BASE_URL = os.getenv("EVAL_API_BASE_URL", "http://localhost:8000")


def locator(citation: dict) -> tuple:
    return (citation["document_code"], citation["article"], citation.get("clause"), citation.get("point"))


def main(path: str) -> int:
    cases = load_golden_cases(path, required_count=100)
    totals = {"grounded": 0, "citation_correct": 0, "abstention": 0, "expected_grounded": 0, "expected_abstention": 0}
    with httpx.Client(base_url=BASE_URL, timeout=120) as client:
        for case in cases:
            response = client.post("/api/v1/chat/query", json={"question": case.question, "as_of_date": case.as_of_date})
            response.raise_for_status()
            answer = response.json()
            expected_status = case.expected_status
            if expected_status == "grounded":
                totals["expected_grounded"] += 1
                if answer["status"] == "grounded":
                    totals["grounded"] += 1
                    expected = {
                        (citation.document_code, citation.article, citation.clause, citation.point)
                        for citation in case.expected_citations
                    }
                    actual = {locator(c) for c in answer["citations"]}
                    totals["citation_correct"] += expected.issubset(actual)
            else:
                totals["expected_abstention"] += 1
                totals["abstention"] += answer["status"] == "abstained"
    grounded_rate = totals["grounded"] / max(totals["expected_grounded"], 1)
    citation_rate = totals["citation_correct"] / max(totals["expected_grounded"], 1)
    abstention_rate = totals["abstention"] / max(totals["expected_abstention"], 1)
    report = {
        "grounded_rate": grounded_rate,
        "citation_rate": citation_rate,
        "safe_abstention_rate": abstention_rate,
        "passes": grounded_rate >= .90 and citation_rate == 1 and abstention_rate >= .95,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
