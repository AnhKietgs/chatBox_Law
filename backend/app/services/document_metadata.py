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
