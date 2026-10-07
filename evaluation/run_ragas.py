"""Evaluate a running legal RAG deployment with RAGAS + deterministic legal gates.

The script sends the evaluator through the protected admin endpoint because
RAGAS must inspect all retrieved provisions, not merely the citations that are
shown publicly. It does not write to the legal corpus or alter conversations.

Example (local Docker):
  python evaluation/run_ragas.py evaluation/golden_questions.jsonl \
    --api-base-url http://localhost:8000 --admin-email "$env:ADMIN_EMAIL" \
    --admin-password "$env:ADMIN_PASSWORD" --judge-base-url http://localhost:11434/v1
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any

import httpx

from golden import GoldenCase, load_golden_cases


DEFAULT_THRESHOLDS = {
    "faithfulness": 0.90,
    "answer_relevancy": 0.80,
    "context_precision": 0.80,
    "context_recall": 0.90,
    "grounded_rate": 0.90,
    "citation_rate": 1.00,
    "safe_abstention_rate": 0.95,
}


def citation_locator(citation: dict[str, Any]) -> tuple[str, str, str | None, str | None]:
    return (citation["document_code"], citation["article"], citation.get("clause"), citation.get("point"))


def admin_token(client: httpx.Client, email: str, password: str) -> str:
    response = client.post("/api/v1/admin/token", json={"email": email, "password": password})
    response.raise_for_status()
    return response.json()["access_token"]


def create_ragas_metrics(*, model: str, base_url: str, api_key: str):
    """Build RAGAS v0.4 metrics using an OpenAI-compatible judge endpoint.

    Ollama exposes this compatibility surface at /v1, allowing a local Qwen
    judge. A stronger independently hosted model may be configured in CI.
    """
    try:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    except ImportError as exc:  # pragma: no cover - exercised in evaluator image/CI
        raise RuntimeError(
            "Thiếu evaluator dependencies. Chạy: pip install -r evaluation/requirements-ragas.txt"
        ) from exc
    client = AsyncOpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    judge_llm = llm_factory(model=model, client=client)
    return {
        "faithfulness": Faithfulness(llm=judge_llm),
        "answer_relevancy": AnswerRelevancy(llm=judge_llm),
        "context_precision": ContextPrecision(llm=judge_llm),
        "context_recall": ContextRecall(llm=judge_llm),
    }


async def score_metric(metric: Any, kwargs: dict[str, Any]) -> float:
    """Normalise RAGAS Score values and retain compatibility around reference naming."""
    try:
        result = await metric.ascore(**kwargs)
    except TypeError as exc:
        # Some early RAGAS 0.4 releases called the canonical answer `reference_answer`.
        if "reference" not in kwargs:
            raise
        compatibility_kwargs = {key: value for key, value in kwargs.items() if key != "reference"}
        compatibility_kwargs["reference_answer"] = kwargs["reference"]
        try:
            result = await metric.ascore(**compatibility_kwargs)
        except TypeError:
            raise exc
    value = getattr(result, "value", result)
    return float(value)


async def score_grounded_case(metrics: dict[str, Any], case: GoldenCase, trace: dict[str, Any]) -> dict[str, float]:
    common = {
        "user_input": case.question,
        "response": trace["answer"]["answer"],
        "retrieved_contexts": trace["retrieved_contexts"],
    }


async def score_all_grounded_cases(
    metrics: dict[str, Any],
    scoring_inputs: list[tuple[GoldenCase, dict[str, Any], dict[str, Any]]],
) -> None:
    """Score serially in one event loop; local Ollama normally serves one judge at a time."""
    for case, row, trace in scoring_inputs:
        try:
            row["metrics"] = await score_grounded_case(metrics, case, trace)
        except Exception as exc:  # Never silently mark a failed RAGAS metric as passing.
            row["error"] = f"RAGAS scoring failed: {type(exc).__name__}: {exc}"
    return {
        "faithfulness": await score_metric(metrics["faithfulness"], common),
        "answer_relevancy": await score_metric(metrics["answer_relevancy"], {
            "user_input": case.question,
            "response": trace["answer"]["answer"],
        }),
        "context_precision": await score_metric(metrics["context_precision"], {
            "user_input": case.question,
            "retrieved_contexts": trace["retrieved_contexts"],
            "reference": case.reference_answer,
        }),
        "context_recall": await score_metric(metrics["context_recall"], {
            "user_input": case.question,
            "retrieved_contexts": trace["retrieved_contexts"],
            "reference": case.reference_answer,
        }),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden_path")
    parser.add_argument("--api-base-url", default=os.getenv("EVAL_API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--admin-email", default=os.getenv("EVAL_ADMIN_EMAIL"))
    parser.add_argument("--admin-password", default=os.getenv("EVAL_ADMIN_PASSWORD"))
    parser.add_argument("--judge-base-url", default=os.getenv("RAGAS_JUDGE_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--judge-api-key", default=os.getenv("RAGAS_JUDGE_API_KEY", "ollama"))
    parser.add_argument("--judge-model", default=os.getenv("RAGAS_JUDGE_MODEL", "Qwen3.5:4b"))
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.admin_email or not args.admin_password:
        raise SystemExit("Cần EVAL_ADMIN_EMAIL và EVAL_ADMIN_PASSWORD của môi trường đánh giá.")
    cases = load_golden_cases(args.golden_path, required_count=100)
    metrics = create_ragas_metrics(
        model=args.judge_model, base_url=args.judge_base_url, api_key=args.judge_api_key,
    )
    case_reports: list[dict[str, Any]] = []
    scoring_inputs: list[tuple[GoldenCase, dict[str, Any], dict[str, Any]]] = []
    with httpx.Client(base_url=args.api_base_url.rstrip("/"), timeout=180) as client:
        token = admin_token(client, args.admin_email, args.admin_password)
        headers = {"Authorization": f"Bearer {token}"}
        for case in cases:
            response = client.post(
                "/api/v1/admin/evaluation/query",
                headers=headers,
                json={"question": case.question, "as_of_date": case.as_of_date},
            )
            response.raise_for_status()
            trace = response.json()
            answer = trace["answer"]
            actual_citations = {citation_locator(item) for item in answer["citations"]}
            expected_citations = {
                (item.document_code, item.article, item.clause, item.point)
                for item in case.expected_citations
            }
            row: dict[str, Any] = {
                "id": case.id,
                "expected_status": case.expected_status,
                "actual_status": answer["status"],
                "citation_correct": expected_citations.issubset(actual_citations),
                "retrieved_context_count": len(trace["retrieved_contexts"]),
                "metrics": {},
            }
            if case.expected_status == "grounded" and answer["status"] == "grounded":
                scoring_inputs.append((case, row, trace))
            case_reports.append(row)

    asyncio.run(score_all_grounded_cases(metrics, scoring_inputs))

    expected_grounded = [row for row in case_reports if row["expected_status"] == "grounded"]
    expected_abstained = [row for row in case_reports if row["expected_status"] == "abstained"]
    summary: dict[str, float] = {
        "grounded_rate": sum(row["actual_status"] == "grounded" for row in expected_grounded) / max(len(expected_grounded), 1),
        "citation_rate": sum(row["citation_correct"] for row in expected_grounded) / max(len(expected_grounded), 1),
        "safe_abstention_rate": sum(row["actual_status"] == "abstained" for row in expected_abstained) / max(len(expected_abstained), 1),
    }
    for metric_name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        values = [row["metrics"][metric_name] for row in expected_grounded if metric_name in row["metrics"]]
        summary[metric_name] = fmean(values) if values else 0.0
    scoring_errors = [row["id"] for row in case_reports if "error" in row]
    passes = not scoring_errors and all(summary[name] >= threshold for name, threshold in DEFAULT_THRESHOLDS.items())
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "api_base_url": args.api_base_url,
        "judge_model": args.judge_model,
        "case_count": len(cases),
        "thresholds": DEFAULT_THRESHOLDS,
        "summary": summary,
        "scoring_errors": scoring_errors,
        "passes": passes,
        "cases": case_reports,
    }
    target = Path(args.report) if args.report else Path("evaluation/reports") / f"ragas-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("case_count", "thresholds", "summary", "scoring_errors", "passes")}, ensure_ascii=False, indent=2))
    print(f"Report: {target}")
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
