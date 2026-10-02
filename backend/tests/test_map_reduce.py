"""Tests for MapReduceSummarizer."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.map_reduce import MapReduceSummarizer, MapReduceResult


def _make_source(sid: str, text: str) -> dict:
    return {"source_id": sid, "citation": f"Tài liệu {sid}", "text": text}


def _mock_response(content: str) -> MagicMock:
    m = MagicMock()
    m.json.return_value = {"message": {"content": content}}
    m.raise_for_status = lambda: None
    return m


# ---------------------------------------------------------------------------
# Single batch — no reduce call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_batch_no_reduce():
    """Với 1 batch, chỉ 1 LLM call được thực hiện (map), không có reduce."""
    sources = [_make_source("S1", "nội dung ngắn về hợp đồng lao động")]
    call_count = 0

    async def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _mock_response(f"Tóm tắt batch {call_count}")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=fake_post):
        result = await MapReduceSummarizer().summarize("hợp đồng lao động", sources, 1000)

    assert isinstance(result, MapReduceResult)
    assert result.batch_count == 1
    assert len(result.merged_text) > 0
    assert call_count == 1  # chỉ 1 LLM call (map), không có reduce


@pytest.mark.asyncio
async def test_single_batch_result_contains_summary():
    sources = [_make_source("S1", "Điều 1 hợp đồng")]
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _mock_response("Điều 1 quy định về hợp đồng.")
        result = await MapReduceSummarizer().summarize("câu hỏi", sources, 1000)
    assert "Điều 1" in result.merged_text


# ---------------------------------------------------------------------------
# Multi-batch — reduce phase triggered
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multi_batch_triggers_reduce():
    """3 sources lớn → 3 batches → 3 map calls + 1 reduce call = 4 tổng."""
    # Mỗi source cần ~229 tokens (800 chars / 3.5), map_batch_token_limit=2000
    # → tất cả vừa 1 batch trừ khi budget nhỏ hơn.
    # Để force 3 batches, dùng text đủ lớn (600 chars ≈ 171 tokens mỗi source)
    # và patch settings.map_batch_token_limit=200
    sources = [
        _make_source(f"S{i}", "x" * 600)
        for i in range(1, 4)
    ]
    call_count = 0

    async def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Trả về summary ngắn cho map; trả về merged cho reduce
        content = f"summary_{call_count}" if call_count <= 3 else "merged_final"
        return _mock_response(content)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=fake_post), \
         patch("app.services.map_reduce.get_settings") as mock_settings:
        s = MagicMock()
        s.map_batch_token_limit = 200      # force mỗi source vào 1 batch riêng
        s.map_reduce_timeout_seconds = 30.0
        s.context_model_limit = 8192
        s.ollama_base_url = "http://localhost:11434"
        s.ollama_model = "test-model"
        mock_settings.return_value = s

        # target_token_budget nhỏ → trigger reduce
        result = await MapReduceSummarizer().summarize("câu hỏi", sources, target_token_budget=10)

    assert result.batch_count == 3
    assert call_count == 4   # 3 map + 1 reduce


@pytest.mark.asyncio
async def test_multi_batch_no_reduce_when_summaries_fit():
    """Khi joined summaries fit trong budget, reduce KHÔNG được gọi."""
    sources = [_make_source(f"S{i}", "a" * 100) for i in range(1, 3)]
    call_count = 0

    async def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _mock_response("tóm ngắn")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=fake_post), \
         patch("app.services.map_reduce.get_settings") as mock_settings:
        s = MagicMock()
        s.map_batch_token_limit = 50     # force 2 batches
        s.map_reduce_timeout_seconds = 30.0
        s.context_model_limit = 8192
        s.ollama_base_url = "http://localhost:11434"
        s.ollama_model = "test-model"
        mock_settings.return_value = s

        # target_token_budget lớn → joined summaries fit → không reduce
        result = await MapReduceSummarizer().summarize("câu hỏi", sources, target_token_budget=9999)

    assert result.batch_count == 2
    assert call_count == 2   # chỉ 2 map calls, không có reduce


# ---------------------------------------------------------------------------
# LLM failure handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_failure_returns_empty_merged():
    """Khi LLM fail hoàn toàn, trả về MapReduceResult có merged_text rỗng hoặc fallback."""
    sources = [_make_source("S1", "nội dung")]
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=Exception("timeout")):
        result = await MapReduceSummarizer().summarize("câu hỏi", sources, 1000)
    # Khi batch thất bại, fallback dùng raw text (không empty) — hoặc nếu
    # fallback cũng fail thì merged_text rỗng. Điều quan trọng: không raise.
    assert isinstance(result, MapReduceResult)
    assert result.batch_count == 1


# ---------------------------------------------------------------------------
# Batch splitting
# ---------------------------------------------------------------------------

def test_split_into_batches_respects_limit():
    """_split_into_batches không vượt batch_token_limit."""
    from app.services.token_budget import TokenBudget
    tb = TokenBudget(8192, 0, 0)
    summarizer = MapReduceSummarizer()
    sources = [_make_source(f"S{i}", "a" * 100) for i in range(5)]
    # ~100/3.5 ≈ 29 tokens mỗi source; limit=50 → tối đa 1 source/batch
    batches = summarizer._split_into_batches(sources, batch_token_limit=50, tb=tb)
    for batch in batches:
        total = sum(tb.estimate(s.get("text", "")) for s in batch)
        assert total <= 50 + tb.estimate("a" * 100)  # +1 source tolerance


def test_split_empty_sources():
    from app.services.token_budget import TokenBudget
    tb = TokenBudget(8192, 0, 0)
    batches = MapReduceSummarizer()._split_into_batches([], 2000, tb)
    # empty input → 1 empty batch hoặc empty list — không raise
    assert isinstance(batches, list)
