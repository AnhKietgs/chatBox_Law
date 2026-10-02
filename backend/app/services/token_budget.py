"""Token budget: estimate token usage and fit sources into a prompt budget.

Estimation uses ceil(len(text) / 3.5) — a conservative approximation that
works for Vietnamese UTF-8 text (each accented character is 1 Unicode code
point, but the BPE tokeniser used by most LLMs maps Vietnamese syllables to
1-2 tokens, so dividing character count by 3.5 stays safely below the real
token count and never under-estimates).
"""
from __future__ import annotations

import math
import logging

logger = logging.getLogger(__name__)


class TokenBudget:
    """Measure token usage and decide which sources fit inside a prompt budget.

    Args:
        model_context_limit: Maximum tokens the model can accept in total.
        reserved_output:     Tokens set aside for the model's generated answer.
        overhead:            Fixed overhead for JSON wrappers, system markers, etc.
    """

    def __init__(self, model_context_limit: int, reserved_output: int, overhead: int) -> None:
        self._limit = model_context_limit
        self._reserved = reserved_output
        self._overhead = overhead

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, text: str) -> int:
        """Return a conservative token estimate for *text*.

        Uses ceil(len / 3.5); always returns ≥ 1 for non-empty input.
        """
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 3.5))

    def available(self, fixed_prompt_text: str) -> int:
        """Return tokens remaining after accounting for the fixed prompt parts.

        ``fixed_prompt_text`` is everything that is not the source list:
        system prompt + conversation context + user question.
        May return a negative number when the fixed prompt alone exceeds budget.
        """
        used = self.estimate(fixed_prompt_text) + self._reserved + self._overhead
        return self._limit - used

    def fit_sources_to_budget(
        self,
        sources: list[dict],
        fixed_prompt_text: str,
    ) -> tuple[list[dict], bool]:
        """Return the longest prefix of *sources* that fits in the remaining budget.

        Sources must already be ordered by descending relevance score (the
        reranker does this); the method greedily takes sources from the front
        until the budget is exhausted.

        Returns:
            (fitted_sources, was_truncated) — fitted_sources is always a list
            (possibly empty); was_truncated is True when at least one source
            was dropped.
        """
        remaining = self.available(fixed_prompt_text)
        if remaining <= 0:
            logger.warning(
                "TokenBudget: fixed prompt alone exceeds budget "
                "(available=%d); injecting 0 sources",
                remaining,
            )
            return [], True

        fitted: list[dict] = []
        for source in sources:
            # Serialise the same way answering.py will — as a JSON-like object.
            text = source.get("text", "")
            citation = source.get("citation", "")
            source_id = source.get("source_id", "")
            token_cost = self.estimate(f"{source_id} {citation} {text}")
            if token_cost > remaining:
                break
            fitted.append(source)
            remaining -= token_cost

        was_truncated = len(fitted) < len(sources)
        if was_truncated:
            logger.warning(
                "TokenBudget: truncated source list from %d to %d",
                len(sources),
                len(fitted),
            )
        return fitted, was_truncated
