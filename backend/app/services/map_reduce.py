"""Map-Reduce summarizer for very long provision lists.

When the total token budget is exhausted after fitting sources, this module
splits the full source list into manageable batches, asks the LLM to
summarise each batch relative to the user's question (Map), then merges all
batch summaries into a single condensed text (Reduce).  A second LLM call
for Reduce is only made when the merged summaries still exceed the target
budget.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter

import httpx

from ..config import get_settings
from .token_budget import TokenBudget

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class MapReduceResult:
    merged_text: str
    batch_count: int
    total_input_tokens: int


class MapReduceSummarizer:
    """Summarise a list of legal provisions via Map-Reduce over Ollama."""

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def summarize(
        self,
        question: str,
        sources: list[dict],
        target_token_budget: int,
    ) -> MapReduceResult:
        """Return a single condensed text that fits within *target_token_budget* tokens.

        Args:
            question:             The user's original question (guides the Map prompt).
            sources:              List of source dicts (source_id, citation, text).
            target_token_budget:  Maximum token count for the final merged output.

        On any LLM failure the method returns a MapReduceResult with empty
        ``merged_text`` so callers can fall back gracefully.
        """
        settings = get_settings()
        tb = TokenBudget(settings.context_model_limit, 0, 0)  # raw estimate only
        total_input_tokens = sum(tb.estimate(s.get("text", "")) for s in sources)

        batches = self._split_into_batches(sources, settings.map_batch_token_limit, tb)
        batch_count = len(batches)
        logger.info("MapReduce: %d sources → %d batches", len(sources), batch_count)

        # ---- MAP phase ----
        summaries: list[str] = []
        for i, batch in enumerate(batches):
            try:
                batch_text = "\n\n".join(
                    f"[{s['source_id']}] {s.get('citation', '')}\n{s.get('text', '')}"
                    for s in batch
                )
                prompt = (
                    f"Câu hỏi: {question}\n\n"
                    f"Dưới đây là các điều khoản pháp lý. "
                    f"Hãy tóm tắt CHỈ những nội dung trực tiếp liên quan đến câu hỏi trên, "
                    f"bằng tiếng Việt, ngắn gọn, không suy diễn thêm:\n\n{batch_text}"
                )
                summary = await self._call_llm(prompt, settings.map_reduce_timeout_seconds)
                summaries.append(summary)
                logger.info("MapReduce: batch %d/%d summarized (%d chars)", i + 1, batch_count, len(summary))
            except Exception as exc:
                logger.warning("MapReduce: batch %d/%d failed: %s", i + 1, batch_count, exc)
                # On partial failure, include whatever text we have so the answer
                # is not completely empty — use raw provision text as fallback.
                summaries.append(
                    " ".join(s.get("text", "")[:200] for s in batch)
                )

        # ---- REDUCE phase (only if needed) ----
        if not summaries:
            return MapReduceResult("", batch_count, total_input_tokens)

        if batch_count == 1:
            # Single batch: no reduce call needed.
            merged = summaries[0]
        else:
            joined = "\n\n".join(f"Tóm tắt {i+1}:\n{s}" for i, s in enumerate(summaries))
            if tb.estimate(joined) > target_token_budget:
                logger.info("MapReduce: reduce phase triggered (joined %d tokens > budget %d)", tb.estimate(joined), target_token_budget)
                try:
                    reduce_prompt = (
                        f"Câu hỏi: {question}\n\n"
                        f"Dưới đây là các tóm tắt từ nhiều phần. "
                        f"Hãy tổng hợp thành một đoạn văn ngắn gọn duy nhất bằng tiếng Việt, "
                        f"chỉ giữ lại thông tin trực tiếp trả lời câu hỏi:\n\n{joined}"
                    )
                    merged = await self._call_llm(reduce_prompt, settings.map_reduce_timeout_seconds)
                except Exception as exc:
                    logger.warning("MapReduce: reduce phase failed: %s", exc)
                    merged = joined  # fallback: use concatenated summaries
            else:
                merged = joined

        return MapReduceResult(merged, batch_count, total_input_tokens)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_into_batches(
        self,
        sources: list[dict],
        batch_token_limit: int,
        tb: TokenBudget,
    ) -> list[list[dict]]:
        """Greedily pack sources into batches, each ≤ *batch_token_limit* tokens."""
        batches: list[list[dict]] = []
        current_batch: list[dict] = []
        current_tokens = 0

        for source in sources:
            cost = tb.estimate(
                f"{source.get('source_id', '')} {source.get('citation', '')} {source.get('text', '')}"
            )
            if current_batch and current_tokens + cost > batch_token_limit:
                batches.append(current_batch)
                current_batch = [source]
                current_tokens = cost
            else:
                current_batch.append(source)
                current_tokens += cost

        if current_batch:
            batches.append(current_batch)

        return batches if batches else [[]]

    async def _call_llm(self, prompt: str, timeout: float) -> str:
        """Call Ollama chat API and return the response text."""
        settings = get_settings()
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "keep_alive": "30m",
                    "options": {"temperature": 0},
                },
            )
            response.raise_for_status()
            return str(response.json().get("message", {}).get("content", "")).strip()
