"""Conversation memory: summarise old turns into a short paragraph.

Turns outside the sliding window are condensed by the LLM into a single
paragraph (≤150 tokens by default).  This summary is stored in
``Conversation.memory_summary`` and prepended as context for future turns.

Failure is non-fatal: if the LLM call fails the function returns an empty
summary so the caller can proceed with raw history.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class MemoryUpdate:
    summary: str
    turns_summarized: int
    llm_ms: float


async def summarize_history(
    messages: list,
    existing_summary: str | None,
) -> MemoryUpdate:
    """Summarise *messages* (the old turns outside the sliding window).

    Args:
        messages:         List of ChatMessage-like objects with ``.role`` and
                          ``.content`` attributes.  These are the OLDER turns
                          that fall outside the recent sliding window.
        existing_summary: A previous summary to be incorporated, so context
                          is not lost across multiple summarisation passes.

    Returns:
        MemoryUpdate with the new summary text.  On LLM failure the summary
        is an empty string — the caller must handle this gracefully.
    """
    settings = get_settings()
    started = perf_counter()

    if not messages:
        return MemoryUpdate(summary=existing_summary or "", turns_summarized=0, llm_ms=0.0)

    # Build the conversation text to summarise.
    lines: list[str] = []
    if existing_summary:
        lines.append(f"Tóm tắt trước: {existing_summary}")
    for msg in messages:
        role_label = "Người dùng" if getattr(msg, "role", "user") == "user" else "Trợ lý"
        content = " ".join(str(getattr(msg, "content", "")).split())
        if content:
            lines.append(f"{role_label}: {content}")

    conversation_text = "\n".join(lines)

    system_prompt = (
        "Bạn là công cụ tóm tắt hội thoại. "
        "Hãy tóm tắt nội dung cuộc trò chuyện dưới đây thành MỘT đoạn văn ngắn (tối đa 150 token) bằng tiếng Việt. "
        "Chỉ giữ lại chủ đề, câu hỏi chính và kết quả quan trọng. "
        "Không bịa thêm thông tin pháp lý. Không dùng Markdown."
    )
    user_prompt = f"Cuộc trò chuyện cần tóm tắt:\n\n{conversation_text}"

    try:
        async with httpx.AsyncClient(timeout=settings.contextualize_timeout_seconds) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ollama_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "think": False,
                    "keep_alive": "30m",
                    "options": {"temperature": 0, "num_predict": 200},
                },
            )
            response.raise_for_status()
            summary = str(response.json().get("message", {}).get("content", "")).strip()
        llm_ms = (perf_counter() - started) * 1000
        logger.info(
            "Memory summary created: %d turns → %d chars in %.0fms",
            len(messages),
            len(summary),
            llm_ms,
        )
        return MemoryUpdate(summary=summary, turns_summarized=len(messages), llm_ms=llm_ms)
    except Exception as exc:
        llm_ms = (perf_counter() - started) * 1000
        logger.warning("Memory summarization failed (non-fatal): %s", exc)
        return MemoryUpdate(summary="", turns_summarized=len(messages), llm_ms=llm_ms)
