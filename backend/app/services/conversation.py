"""Conversation context helpers. History resolves references but is never evidence."""
from collections.abc import Iterable


def build_conversation_context(
    messages: Iterable[object],
    max_chars: int,
    memory_summary: str | None = None,
    window_turns: int | None = None,  # reserved for future per-call override; unused in logic
) -> str:
    """Keep the most recent complete messages within a bounded prompt budget.

    When *memory_summary* is provided (and non-empty), it is prepended as a
    context prefix labelled ``[Ngữ cảnh trước]``.  This lets older turns that
    have been condensed by the memory module stay visible to the LLM without
    consuming the full sliding-window budget.

    Backward-compatible: callers that do not pass *memory_summary* get the
    same output as before.
    """
    selected: list[str] = []
    used = 0
    for message in reversed(list(messages)):
        role = getattr(message, "role", "user")
        content = " ".join(str(getattr(message, "content", "")).split())
        if not content:
            continue
        line = f"{'Người dùng' if role == 'user' else 'Trợ lý'}: {content}"
        if selected and used + len(line) > max_chars:
            break
        selected.append(line[:max_chars - used])
        used += len(selected[-1])
        if used >= max_chars:
            break
    recent_turns_text = "\n".join(reversed(selected))

    if memory_summary:
        return f"[Ngữ cảnh trước]: {memory_summary}\n\n{recent_turns_text}"
    return recent_turns_text


def contextual_retrieval_query(question: str, context: str) -> str:
    if not context:
        return question
    return f"Câu hỏi hiện tại: {question}\nNgữ cảnh hội thoại (không phải căn cứ pháp lý):\n{context}"
