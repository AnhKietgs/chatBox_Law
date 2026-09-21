"""Conservative Vietnamese legal-structure parser.

The parser only emits provision-level units when an Article can be identified. A
human reviewer can therefore reject malformed source extraction before indexing.
"""
from dataclasses import dataclass
import re

# Docling's Markdown export uses headings (for example ``## Điều 301``) and
# lists (for example ``- 1.`` or ``- a)``).  Both forms need to preserve the
# legal locator; otherwise a later article is silently appended to the prior
# article's text and its citation becomes false.
ARTICLE_RE = re.compile(r"(?im)^\s*(?:#{1,6}\s*)?Điều\s+(\d+[a-zA-Z]?)\s*[\.\-:]?\s*(.*)$")
CLAUSE_RE = re.compile(r"(?im)^\s*(?:[-*+]\s+)?(\d+)\s*[\.\)]\s*(.+)$")
POINT_RE = re.compile(r"(?im)^\s*(?:[-*+]\s+)?([a-zđ])\s*[\)\.]\s*(.+)$")


@dataclass(frozen=True)
class ParsedProvision:
    article_no: str
    clause_no: str | None
    point_label: str | None
    heading: str | None
    content: str
    ordinal: int


def parse_legal_text(text: str) -> list[ParsedProvision]:
    articles = list(ARTICLE_RE.finditer(text))
    if not articles:
        raise ValueError("Không nhận diện được cấu trúc Điều trong văn bản")
    results: list[ParsedProvision] = []
    ordinal = 0
    for index, article_match in enumerate(articles):
        article_no, heading = article_match.group(1), article_match.group(2).strip() or None
        block_end = articles[index + 1].start() if index + 1 < len(articles) else len(text)
        block = text[article_match.end():block_end].strip()
        clauses = list(CLAUSE_RE.finditer(block))
        if not clauses:
            ordinal += 1
            results.append(ParsedProvision(article_no, None, None, heading, block, ordinal))
            continue
        preamble = block[:clauses[0].start()].strip()
        if preamble:
            ordinal += 1
            results.append(ParsedProvision(article_no, None, None, heading, preamble, ordinal))
        for clause_index, clause_match in enumerate(clauses):
            clause_no = clause_match.group(1)
            clause_end = clauses[clause_index + 1].start() if clause_index + 1 < len(clauses) else len(block)
            clause_text = clause_match.group(2) + block[clause_match.end():clause_end]
            points = list(POINT_RE.finditer(clause_text))
            if not points:
                ordinal += 1
                results.append(ParsedProvision(article_no, clause_no, None, heading, clause_text.strip(), ordinal))
                continue
            preamble = clause_text[:points[0].start()].strip()
            if preamble:
                ordinal += 1
                results.append(ParsedProvision(article_no, clause_no, None, heading, preamble, ordinal))
            for point_index, point_match in enumerate(points):
                point_end = points[point_index + 1].start() if point_index + 1 < len(points) else len(clause_text)
                point_text = point_match.group(2) + clause_text[point_match.end():point_end]
                ordinal += 1
                results.append(ParsedProvision(article_no, clause_no, point_match.group(1), heading, point_text.strip(), ordinal))
    parsed = [item for item in results if item.content]
    # A nested article heading means boundaries were not preserved.  Refuse to
    # index it rather than attach text from a later article to a false locator.
    if any(ARTICLE_RE.search(item.content) for item in parsed):
        raise ValueError("Cấu trúc Điều bị lồng/chồng; không an toàn để xuất bản")
    return parsed
