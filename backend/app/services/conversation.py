"""Conversation context helpers. History resolves references but is never evidence."""
from collections.abc import Iterable


def build_conversation_context(messages: Iterable[object], max_chars: int) -> str:
    """Keep the most recent complete messages within a bounded prompt budget."""
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
    return "\n".join(reversed(selected))


def contextual_retrieval_query(question: str, context: str) -> str:
    if not context:
        return question
    return f"Câu hỏi hiện tại: {question}\nNgữ cảnh hội thoại (không phải căn cứ pháp lý):\n{context}"
