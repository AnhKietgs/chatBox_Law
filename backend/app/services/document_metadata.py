"""Controlled legal-document metadata used by PostgreSQL and Qdrant filters."""
from __future__ import annotations

import re

DOCUMENT_DOMAINS = (
    "general",
    "commercial",
    "civil",
    "labor",
    "enterprise",
    "tax",
    "family",
    "criminal",
)

_DOMAIN_ALIASES = {
    "thuong_mai": "commercial", "thương_mại": "commercial", "commercial": "commercial",
    "dan_su": "civil", "dân_sự": "civil", "civil": "civil",
    "lao_dong": "labor", "lao_động": "labor", "labor": "labor",
    "doanh_nghiep": "enterprise", "doanh_nghiệp": "enterprise", "enterprise": "enterprise",
    "thue": "tax", "thuế": "tax", "tax": "tax",
    "gia_dinh": "family", "gia_đình": "family", "family": "family",
    "hinh_su": "criminal", "hình_sự": "criminal", "criminal": "criminal",
    "general": "general", "khac": "general", "khác": "general",
}


def normalize_domain(value: str) -> str:
    key = re.sub(r"\s+", "_", value.strip().casefold())
    domain = _DOMAIN_ALIASES.get(key)
    if domain is None:
        allowed = ", ".join(DOCUMENT_DOMAINS)
        raise ValueError(f"Lĩnh vực không hợp lệ. Chọn một trong: {allowed}")
    return domain


def normalize_category(value: str | None) -> str:
    """Keep category as controlled, filter-safe metadata without guessing law."""
    normalized = re.sub(r"\s+", "_", (value or "general").strip().casefold())
    normalized = re.sub(r"[^\wđ]", "", normalized, flags=re.UNICODE)
    if not normalized or len(normalized) > 100:
        raise ValueError("Nhóm nghiệp vụ phải dài từ 1 đến 100 ký tự")
    return normalized


def infer_legacy_domain(document_code: str, title: str) -> str:
    """Safe migration defaults for the two initial corpus documents only."""
    text = f"{document_code} {title}".casefold()
    if "ltm" in text or "thương mại" in text:
        return "commercial"
    if "blds" in text or "dân sự" in text:
        return "civil"
    return "general"


def explicit_article_numbers(question: str) -> tuple[str, ...]:
    """Return distinct article numbers explicitly written by the user."""
    return tuple(dict.fromkeys(re.findall(r"\bđiều\s+(\d+[a-zđ]?)\b", question.casefold())))


def question_mentions_document(question: str, document_code: str, title: str) -> bool:
    """Match an explicit document name/code without guessing legal scope."""
    normalized = " ".join(question.casefold().split())
    code = document_code.strip().casefold()
    title_normalized = " ".join(title.casefold().split())
    title_without_year = re.sub(r"\s+\d{4}\s*$", "", title_normalized).strip()
    if code and code in normalized:
        return True
    if title_normalized and title_normalized in normalized:
        return True
    if title_without_year and len(title_without_year) >= 5 and title_without_year in normalized:
        return True
    prefix = code.split("-", 1)[0]
    return len(prefix) >= 3 and re.search(rf"(?<!\w){re.escape(prefix)}(?!\w)", normalized) is not None
