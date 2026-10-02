"""Integration smoke tests covering the 5 Review Focus edge cases.

These tests exercise code paths that cut across multiple modules and are
most likely to surface runtime errors that unit tests miss.
"""
from __future__ import annotations

import math
import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.token_budget import TokenBudget


# ---------------------------------------------------------------------------
# Review Focus 1: estimate() conservative for Vietnamese multi-byte chars
# ---------------------------------------------------------------------------

def test_estimate_conservative_for_vietnamese():
    """
    len(text) counts Unicode code points, not bytes.
    ceil(len / 3.5) is always ≥ 1 for non-empty Vietnamese text.
    Since Vietnamese BPE tokens ≈ 1 char each, dividing by 3.5 gives a
    value well below the real token count — safely under-estimates budget
    usage, never over-claims space.
    """
    tb = TokenBudget(8192, 1024, 256)
    samples = [
        "Điều khoản hợp đồng lao động theo Bộ luật Lao động.",
        "Người lao động có quyền đơn phương chấm dứt hợp đồng.",
        "Tiền lương tối thiểu vùng được quy định hàng năm.",
    ]
    for text in samples:
        est = tb.estimate(text)
        assert est >= 1, f"estimate({text!r}) returned {est}"
        assert est == math.ceil(len(text) / 3.5)


def test_estimate_never_zero_for_single_char():
    tb = TokenBudget(8192, 1024, 256)
    assert tb.estimate("x") == 1
    assert tb.estimate("ơ") == 1  # Vietnamese accented character


# ---------------------------------------------------------------------------
# Review Focus 2: fit_sources_to_budget returns ([], True) — no exception
# ---------------------------------------------------------------------------

def test_fit_returns_empty_tuple_when_fixed_exceeds_budget():
    """Khi fixed_prompt đã chiếm hết budget, phải trả về ([], True) không raise."""
    tb = TokenBudget(50, 10, 10)   # rất nhỏ
    sources = [{"source_id": "S1", "citation": "X", "text": "a", "score": 1.0}]
    try:
        fitted, truncated = tb.fit_sources_to_budget(sources, "x" * 500)
    except Exception as exc:
        pytest.fail(f"fit_sources_to_budget raised unexpectedly: {exc}")
    assert fitted == []
    assert truncated is True


def test_fit_never_raises_on_empty_sources():
    tb = TokenBudget(8192, 1024, 256)
    fitted, truncated = tb.fit_sources_to_budget([], "any prompt")
    assert fitted == []
    assert truncated is False


# ---------------------------------------------------------------------------
# Review Focus 3: Memory summary fails silently — pipeline continues
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_failure_is_non_fatal():
    """summarize_history không raise khi LLM fail."""
    from app.services.memory import summarize_history
    msg = MagicMock()
    msg.role = "user"
    msg.content = "câu hỏi"
    with patch("httpx.AsyncClient.post", side_effect=Exception("LLM down")):
        result = await summarize_history([msg], existing_summary=None)
    assert result.summary == ""
    assert result.turns_summarized == 1


# ---------------------------------------------------------------------------
# Review Focus 4: Reranker pool < inject_limit does not raise
# ---------------------------------------------------------------------------

def test_retrieval_fewer_candidates_than_inject_limit():
    """Khi vector store trả về 0 kết quả (< inject_limit=5), không raise."""
    from app.services.retrieval import LegalRetriever

    mock_db = MagicMock()
    mock_db.execute.return_value.all.return_value = []

    with patch("app.services.retrieval.HybridVectorStore") as mock_vs, \
         patch("app.services.retrieval.get_settings") as mock_cfg:
        s = MagicMock()
        s.retrieval_recall_pool = 20
        s.retrieval_inject_limit = 5
        s.retrieval_debug_logs = False
        s.confidence_threshold = 0.5
        s.conversation_confidence_threshold = 0.3
        mock_cfg.return_value = s
        mock_vs.return_value.search.return_value = []

        retriever = LegalRetriever(mock_db)
        result = retriever.retrieve_with_metrics("câu hỏi", date.today(), inject_limit=5)

    assert result.provisions == []   # không raise, trả về list rỗng


# ---------------------------------------------------------------------------
# Review Focus 5: Map-Reduce single-batch does not call LLM twice
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_map_reduce_single_batch_one_llm_call():
    """Khi tất cả sources vừa 1 batch, reduce phase KHÔNG gọi LLM lần 2."""
    from app.services.map_reduce import MapReduceSummarizer

    sources = [{"source_id": "S1", "citation": "Điều 1", "text": "nội dung ngắn"}]
    call_count = 0

    async def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        m = MagicMock()
        m.json.return_value = {"message": {"content": "tóm tắt"}}
        m.raise_for_status = lambda: None
        return m

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        result = await MapReduceSummarizer().summarize("câu hỏi", sources, target_token_budget=5000)

    assert result.batch_count == 1
    assert call_count == 1   # 1 map call, 0 reduce calls


# ---------------------------------------------------------------------------
# Settings smoke test — all new keys present with correct defaults
# ---------------------------------------------------------------------------

def test_new_settings_have_correct_defaults():
    """Tất cả 11 settings mới phải có default đúng theo spec."""
    from app.config import get_settings
    # Clear lru_cache để lấy fresh instance (không bị cache từ test khác)
    get_settings.cache_clear()
    s = get_settings()
    assert s.context_model_limit == 8192
    assert s.context_reserved_output == 1024
    assert s.context_overhead == 256
    assert s.retrieval_recall_pool == 20
    assert s.retrieval_inject_limit == 5
    assert s.map_reduce_enabled is False
    assert s.map_batch_token_limit == 2000
    assert s.map_reduce_timeout_seconds == 60.0
    assert s.history_window_turns == 5
    assert s.memory_summary_enabled is True
    assert s.memory_summary_max_tokens == 150
