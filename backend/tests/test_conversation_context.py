"""Tests for build_conversation_context — sliding window + memory prefix."""
from __future__ import annotations

from unittest.mock import MagicMock

from app.services.conversation import build_conversation_context


def make_msg(role: str, content: str) -> MagicMock:
    m = MagicMock()
    m.role = role
    m.content = content
    return m


# ---------------------------------------------------------------------------
# Backward compatibility — no memory_summary
# ---------------------------------------------------------------------------

def test_backward_compat_no_summary():
    """Không truyền memory_summary → output giống hệt hành vi cũ."""
    msgs = [make_msg("user", "hỏi về luật"), make_msg("assistant", "trả lời")]
    ctx = build_conversation_context(msgs, max_chars=1000)
    assert "Người dùng: hỏi về luật" in ctx
    assert "Trợ lý: trả lời" in ctx
    assert "[Ngữ cảnh trước]" not in ctx


def test_none_summary_not_prepended():
    msgs = [make_msg("user", "hỏi")]
    ctx = build_conversation_context(msgs, max_chars=1000, memory_summary=None)
    assert "[Ngữ cảnh trước]" not in ctx


def test_empty_summary_not_prepended():
    msgs = [make_msg("user", "hỏi")]
    ctx = build_conversation_context(msgs, max_chars=1000, memory_summary="")
    assert "[Ngữ cảnh trước]" not in ctx


# ---------------------------------------------------------------------------
# With memory_summary prefix
# ---------------------------------------------------------------------------

def test_memory_summary_as_prefix():
    msgs = [make_msg("user", "câu hỏi mới")]
    ctx = build_conversation_context(
        msgs, max_chars=1000, memory_summary="Tóm tắt cũ: hỏi về hợp đồng."
    )
    assert ctx.startswith("[Ngữ cảnh trước]: Tóm tắt cũ:")
    assert "câu hỏi mới" in ctx


def test_memory_summary_separated_from_recent():
    """Summary và recent turns phải cách nhau bằng dòng trống."""
    msgs = [make_msg("user", "recent")]
    ctx = build_conversation_context(msgs, max_chars=1000, memory_summary="old summary")
    assert "[Ngữ cảnh trước]: old summary\n\n" in ctx


def test_memory_summary_before_recent_turns():
    msgs = [make_msg("user", "câu hỏi gần nhất")]
    ctx = build_conversation_context(msgs, max_chars=1000, memory_summary="tóm lịch sử")
    summary_pos = ctx.index("[Ngữ cảnh trước]")
    recent_pos = ctx.index("câu hỏi gần nhất")
    assert summary_pos < recent_pos


# ---------------------------------------------------------------------------
# max_chars budget respected
# ---------------------------------------------------------------------------

def test_max_chars_truncation():
    msgs = [make_msg("user", "x" * 5000)]
    ctx = build_conversation_context(msgs, max_chars=100)
    assert len(ctx) <= 100


def test_empty_messages_returns_empty_string():
    ctx = build_conversation_context([], max_chars=1000)
    assert ctx == ""


def test_empty_messages_with_summary_returns_only_summary():
    ctx = build_conversation_context([], max_chars=1000, memory_summary="old context")
    assert "[Ngữ cảnh trước]: old context" in ctx
