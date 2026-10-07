"""Tests for TokenBudget — token estimation and source fitting."""
from __future__ import annotations

import pytest
from app.services.token_budget import TokenBudget
from app.services.token_estimation import estimate_vietnamese_tokens


# ---------------------------------------------------------------------------
# estimate()
# ---------------------------------------------------------------------------

def test_estimate_ascii():
    tb = TokenBudget(8192, 1024, 256)
    text = "a" * 350
    # One long identifier is covered by the character bound: ceil(350 / 2.5).
    assert tb.estimate(text) == 140


def test_estimate_vietnamese():
    tb = TokenBudget(8192, 1024, 256)
    # One Vietnamese lexical item reserves two BPE tokens.
    assert tb.estimate("điều") == 2


def test_estimate_never_zero_for_nonempty():
    tb = TokenBudget(8192, 1024, 256)
    assert tb.estimate("x") >= 1


def test_estimate_empty_returns_zero():
    tb = TokenBudget(8192, 1024, 256)
    assert tb.estimate("") == 0


def test_estimate_reserves_two_tokens_per_vietnamese_syllable():
    tb = TokenBudget(8192, 1024, 256)
    text = "Điều khoản hợp đồng lao động theo Bộ luật Lao động."
    # 10 lexical items x 2 tokens. This is materially more conservative than
    # the former ceil(len/3.5) estimate for Vietnamese.
    assert tb.estimate(text) >= 20
    assert tb.estimate(text) == estimate_vietnamese_tokens(text)


# ---------------------------------------------------------------------------
# available()
# ---------------------------------------------------------------------------

def test_available_decreases_with_longer_fixed_prompt():
    tb = TokenBudget(8192, 1024, 256)
    short = tb.available("ngắn")
    long_ = tb.available("x" * 5000)
    assert short > long_


def test_available_negative_when_fixed_exceeds_limit():
    """Khi fixed prompt vượt budget, available() trả về số âm."""
    tb = TokenBudget(100, 50, 10)  # limit=100, reserved=50, overhead=10 → max fixed ~ 40
    result = tb.available("x" * 1000)
    assert result < 0


# ---------------------------------------------------------------------------
# fit_sources_to_budget()
# ---------------------------------------------------------------------------

def _make_source(sid: str, text: str, score: float = 0.9) -> dict:
    return {"source_id": sid, "citation": "Tài liệu", "text": text, "score": score}


def test_fit_all_sources_when_under_budget():
    tb = TokenBudget(8192, 1024, 256)
    sources = [_make_source("S1", "nội dung ngắn")]
    fitted, truncated = tb.fit_sources_to_budget(sources, "prompt ngắn")
    assert fitted == sources
    assert truncated is False


def test_fit_truncates_when_over_budget():
    """Khi budget nhỏ, source thứ 2 bị loại."""
    tb = TokenBudget(120, 30, 10)  # rất nhỏ
    sources = [
        _make_source("S1", "a" * 50),
        _make_source("S2", "b" * 400),
    ]
    fitted, truncated = tb.fit_sources_to_budget(sources, "")
    assert len(fitted) < 2
    assert truncated is True


def test_fit_returns_empty_when_fixed_prompt_already_exceeds_budget():
    """Khi fixed_prompt đã vượt budget, trả về ([], True) không raise."""
    tb = TokenBudget(50, 10, 10)  # available < 0 vì fixed_prompt lớn
    sources = [_make_source("S1", "a")]
    fitted, truncated = tb.fit_sources_to_budget(sources, "x" * 500)
    assert fitted == []
    assert truncated is True


def test_fit_prefers_front_of_list():
    """Các source ở đầu list (score cao hơn) được ưu tiên giữ lại."""
    tb = TokenBudget(200, 30, 10)
    # available("") = 200 - (0 + 30 + 10) = 160 tokens
    # S1 "ngắn" ≈ 5 tokens → remaining ≈ 155
    # S2 cần > 155 tokens: 155 × 3.5 ≈ 542 chars → dùng 600 để chắc chắn
    # ceil((len("S2 Tài liệu ") + 600) / 3.5) = ceil(613/3.5) = 176 > 155 → không fit
    long_text = "z" * 600
    sources = [
        _make_source("S1", "ngắn"),    # sẽ fit
        _make_source("S2", long_text), # sẽ bị cắt
        _make_source("S3", "ngắn2"),   # sẽ không được thử vì S2 đã cắt
    ]
    fitted, truncated = tb.fit_sources_to_budget(sources, "")
    assert fitted[0]["source_id"] == "S1"
    assert truncated is True


def test_fit_empty_sources_list():
    tb = TokenBudget(8192, 1024, 256)
    fitted, truncated = tb.fit_sources_to_budget([], "prompt")
    assert fitted == []
    assert truncated is False
