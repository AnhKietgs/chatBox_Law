"""Defence-in-depth helpers for untrusted document and prompt content.

Legal text is evidence, never executable instructions.  The scanner is a
conservative high-signal gate: it quarantines a version before it becomes a
retrieval chunk and the prompt formatter makes every source explicit data.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
import re


@dataclass(frozen=True)
class InjectionFinding:
    rule_id: str
    excerpt: str


# These phrases intentionally require an instruction-like combination. A
# statute saying "bỏ qua quy định" alone does not match; it needs a target such
# as instructions/system/prompt or a model-identity/secret extraction attempt.
_HIGH_SIGNAL_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore-instructions", re.compile(
        r"(?:bỏ\s*qua|phớt\s*lờ|ignore|disregard)\s+(?:mọi\s+|all\s+)?(?:chỉ\s*dẫn|hướng\s*dẫn|instructions?|previous\s+instructions?|quy\s*tắc|rules?)",
        re.IGNORECASE,
    )),
    ("system-prompt", re.compile(
        r"(?:system\s*prompt|developer\s*message|prompt\s*hệ\s*thống|lời\s*nhắc\s*hệ\s*thống)", re.IGNORECASE,
    )),
    ("role-override", re.compile(
        r"(?:you\s+are\s+(?:now\s+)?(?:chatgpt|an?\s+admin)|bạn\s+(?:bây\s+giờ\s+)?là\s+(?:admin|quản\s*trị\s*viên))",
        re.IGNORECASE,
    )),
    ("secret-exfiltration", re.compile(
        r"(?:tiết\s*lộ|in\s*ra|show|reveal|print|export)\s+(?:.*?\s+)?(?:mật\s*khẩu|api\s*key|token|secret|thông\s*tin\s*đăng\s*nhập)",
        re.IGNORECASE,
    )),
    ("instruction-delimiter", re.compile(
        r"(?:<\s*/?\s*(?:system|instruction|assistant)\s*>|\[\s*(?:system|instruction|assistant)\s*\])",
        re.IGNORECASE,
    )),
)


def scan_document_content(text: str, *, max_findings: int = 3) -> list[InjectionFinding]:
    """Return high-signal prompt-injection indicators in extracted document text."""
    normalized = " ".join(text.split())
    findings: list[InjectionFinding] = []
    for rule_id, pattern in _HIGH_SIGNAL_RULES:
        match = pattern.search(normalized)
        if not match:
            continue
        start = max(0, match.start() - 80)
        end = min(len(normalized), match.end() + 120)
        findings.append(InjectionFinding(rule_id=rule_id, excerpt=normalized[start:end]))
        if len(findings) >= max_findings:
            break
    return findings


def format_untrusted_documents(sources: list[dict[str, str]]) -> str:
    """Serialise retrieved evidence as XML data boundaries, never instructions.

    Escaping content prevents a source from closing a boundary and injecting a
    fake ``<instruction>`` section. Source IDs/citations originate from the
    backend but are escaped as well for a single safe rendering path.
    """
    documents = []
    for source in sources:
        documents.append(
            "<document source_id=\"{source_id}\">\n"
            "  <citation>{citation}</citation>\n"
            "  <content>{content}</content>\n"
            "</document>".format(
                source_id=escape(str(source.get("source_id", "")), quote=True),
                citation=escape(str(source.get("citation", ""))),
                content=escape(str(source.get("text", ""))),
            )
        )
    return "<documents>\n" + "\n".join(documents) + "\n</documents>"
