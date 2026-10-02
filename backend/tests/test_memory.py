"""Tests for memory.summarize_history."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.memory import summarize_history, MemoryUpdate


def make_message(role: str, content: str) -> MagicMock:
    m = MagicMock()
    m.role = role
    m.content = content
    return m


def _mock_response(content: str) -> MagicMock:
    m = MagicMock()
    m.json.return_value = {"message": {"content": content}}
    m.raise_for_status = lambda: None
    return m


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarize_produces_output():
    messages = [
        make_message("user", "hỏi " * 30),
        make_message("assistant", "trả lời " * 30),
    ]
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _mock_response("Tóm tắt: người dùng hỏi về hợp đồng.")
        result = await summarize_history(messages, existing_summary=None)

    assert isinstance(result, MemoryUpdate)
    assert result.turns_summarized == 2
    assert len(result.summary) > 0
    assert result.llm_ms >= 0


@pytest.mark.asyncio
async def test_summarize_shorter_than_input():
    long_content = "hỏi về luật lao động " * 50
    messages = [make_message("user", long_content)]
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _mock_response("Tóm ngắn.")
        result = await summarize_history(messages, existing_summary=None)

    assert len(result.summary) < len(long_content)


# ---------------------------------------------------------------------------
# existing_summary được include trong prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_existing_summary_included_in_prompt():
    messages = [make_message("user", "câu hỏi mới")]
    captured: dict = {}

    async def fake_post(url, json=None, **kwargs):
        captured["body"] = json
        m = MagicMock()
        m.json.return_value = {"message": {"content": "summary mới"}}
        m.raise_for_status = lambda: None
        return m

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        await summarize_history(messages, existing_summary="Tóm tắt cũ: người dùng hỏi về hợp đồng.")

    user_msg_content = next(
        msg["content"]
        for msg in captured["body"]["messages"]
        if msg["role"] == "user"
    )
    assert "Tóm tắt cũ" in user_msg_content


# ---------------------------------------------------------------------------
# Fail silently
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_failure_returns_empty_summary():
    """Khi LLM timeout/fail, trả về empty summary không raise."""
    messages = [make_message("user", "câu hỏi")]
    with patch("httpx.AsyncClient.post", side_effect=Exception("connection refused")):
        result = await summarize_history(messages, existing_summary=None)

    assert result.summary == ""
    assert result.turns_summarized == 1
    assert result.llm_ms >= 0


@pytest.mark.asyncio
async def test_llm_failure_non_fatal_does_not_raise():
    """summarize_history không được raise exception dù LLM fail."""
    messages = [make_message("user", "test")]
    try:
        with patch("httpx.AsyncClient.post", side_effect=RuntimeError("server down")):
            result = await summarize_history(messages, existing_summary=None)
        assert result.summary == ""
    except Exception:
        pytest.fail("summarize_history raised an exception on LLM failure")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_messages_returns_existing_summary():
    result = await summarize_history([], existing_summary="Tóm tắt cũ")
    assert result.summary == "Tóm tắt cũ"
    assert result.turns_summarized == 0


@pytest.mark.asyncio
async def test_empty_messages_no_existing_returns_empty():
    result = await summarize_history([], existing_summary=None)
    assert result.summary == ""
    assert result.turns_summarized == 0
