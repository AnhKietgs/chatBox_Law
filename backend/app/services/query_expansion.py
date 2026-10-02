"""Conservative retrieval-only expansion for commercial-law queries.

Expansion is deterministic and never adds legal conclusions.  The original user
question remains the reranker and LLM input, so generated citations can only
refer to retrieved legal provisions, never to these helper keywords.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ExpandedQuery:
    original: str
    retrieval_query: str
    applied: bool


def _normalized(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


def expand_commercial_query(question: str) -> ExpandedQuery:
    """Add neutral retrieval anchors without replacing a user's original question.

    Detailed questions also benefit from an explicit legal anchor. For example,
    a question mentioning a sale of goods can otherwise dilute the key phrase
    "mức phạt vi phạm" that identifies the governing provision.
    """
    original = " ".join(question.split())
    normalized = _normalized(original)
    additions: list[str] = []
    if "phạt" in normalized or "vi phạm" in normalized:
        additions.append("mức phạt và chế tài vi phạm nghĩa vụ hợp đồng")
        if "tối đa" in normalized or "bao nhiêu" in normalized:
            additions.append("giới hạn mức phạt theo tỷ lệ phần trăm")
    # Chỉ expand sang thương mại khi có dấu hiệu rõ ràng — tránh drift
    # sang LTM-2005 khi câu hỏi là về hợp đồng lao động, dân sự, v.v.
    _labor_signals = {"lao động", "thử việc", "nhân viên", "người lao động",
                      "sa thải", "thôi việc", "lương", "tiền lương", "bảo hiểm xã hội"}
    _civil_signals  = {"dân sự", "thừa kế", "ly hôn", "hôn nhân", "tài sản",
                       "bồi thường thiệt hại ngoài hợp đồng"}
    _is_labor = any(s in normalized for s in _labor_signals)
    _is_civil = any(s in normalized for s in _civil_signals)
    _has_commercial = any(s in normalized for s in
                          {"mua bán", "hàng hóa", "thương mại", "dịch vụ thương mại",
                           "đại lý", "nhượng quyền", "logistics"})
    if "hợp đồng" in normalized and _has_commercial and not _is_labor and not _is_civil:
        additions.append("hợp đồng thương mại mua bán hàng hóa")
    if "mua bán" in normalized or "hàng hóa" in normalized:
        additions.append("mua bán hàng hóa trong hoạt động thương mại")
    if "hàng cấm" in normalized or "cấm kinh doanh" in normalized:
        additions.append("hàng hóa cấm kinh doanh")
    if "bồi thường" in normalized:
        additions.append("bồi thường thiệt hại do vi phạm hợp đồng")

    if not additions:
        return ExpandedQuery(original, original, False)
    # This is deliberately a separate focused query. VectorStore searches it
    # alongside the original wording, instead of diluting the original dense
    # representation by appending a long list of helper terms to it.
    helper_terms = "; ".join(dict.fromkeys(additions))
    return ExpandedQuery(original, helper_terms, True)


def expand_short_commercial_query(question: str) -> ExpandedQuery:
    """Compatibility alias for callers from the initial short-query MVP."""
    return expand_commercial_query(question)
