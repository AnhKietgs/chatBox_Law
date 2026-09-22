"""Conservative retrieval-only expansion for short commercial-law queries.

Expansion is deterministic and never adds legal conclusions.  The original user
question remains the reranker and LLM input, so generated citations can only
refer to retrieved legal provisions, never to these helper keywords.
"""
from dataclasses import dataclass
import re


SHORT_QUERY_WORD_LIMIT = 6


@dataclass(frozen=True)
class ExpandedQuery:
    original: str
    retrieval_query: str
    applied: bool


def _normalized(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


def expand_short_commercial_query(question: str) -> ExpandedQuery:
    """Add neutral search terms only when the short query is in MVP scope."""
    original = " ".join(question.split())
    normalized = _normalized(original)
    if len(normalized.split()) > SHORT_QUERY_WORD_LIMIT:
        return ExpandedQuery(original, original, False)

    additions: list[str] = []
    if "phạt" in normalized or "vi phạm" in normalized:
        additions.append("mức phạt và chế tài vi phạm nghĩa vụ hợp đồng")
    if "hợp đồng" in normalized:
        additions.append("hợp đồng thương mại mua bán hàng hóa")
    if "mua bán" in normalized or "hàng hóa" in normalized:
        additions.append("mua bán hàng hóa trong hoạt động thương mại")
    if "hàng cấm" in normalized or "cấm kinh doanh" in normalized:
        additions.append("hàng hóa cấm kinh doanh")
    if "bồi thường" in normalized:
        additions.append("bồi thường thiệt hại do vi phạm hợp đồng")

    if not additions:
        return ExpandedQuery(original, original, False)
    helper_terms = "; ".join(dict.fromkeys(additions))
    return ExpandedQuery(original, f"{original}\nTừ khóa hỗ trợ truy xuất: {helper_terms}", True)
