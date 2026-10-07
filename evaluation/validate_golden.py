"""Validate golden-set structure before invoking expensive RAGAS judge models."""
from __future__ import annotations

import argparse
import json

from golden import load_golden_cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--required-count", type=int, default=None)
    args = parser.parse_args()
    cases = load_golden_cases(args.path, required_count=args.required_count)
    summary = {
        "cases": len(cases),
        "grounded": sum(case.expected_status == "grounded" for case in cases),
        "abstained": sum(case.expected_status == "abstained" for case in cases),
        "reviewers": sorted({case.reviewed_by for case in cases}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
