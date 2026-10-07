"""Conservative Vietnamese legal-structure parser.

The parser only emits provision-level units when an Article can be identified. A
human reviewer can therefore reject malformed source extraction before indexing.
"""
from dataclasses import dataclass, replace
import re

from .token_estimation import CHARACTERS_PER_TOKEN, estimate_vietnamese_tokens

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
    chunk_index: int = 1


def estimate_tokens(text: str) -> int:
    """Conservative, dependency-free estimate shared with the chunk splitter."""
    return estimate_vietnamese_tokens(text)


def _tail_overlap(text: str, token_limit: int) -> str:
    """Take a word-boundary tail used only between split parts of one unit."""
    if token_limit <= 0:
        return ""
    # Start from a bounded tail, then trim words until it is actually within
    # the shared estimate. Character slicing alone used to undercount Vietnamese.
    max_chars = int(token_limit * CHARACTERS_PER_TOKEN)
    tail = text[-int(max_chars):].strip()
    if " " in tail:
        tail = tail.split(" ", 1)[1]
    words = tail.split()
    while words and estimate_tokens(" ".join(words)) > token_limit:
        words.pop(0)
    tail = " ".join(words)
    return tail


def _hard_split(text: str, token_limit: int) -> list[str]:
    """Last-resort split for one sentence longer than the configured budget."""
    if estimate_tokens(text) <= token_limit:
        return [text.strip()]
    pieces: list[str] = []
    words = text.strip().split()
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and estimate_tokens(candidate) > token_limit:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    # A single unbroken token (for example malformed OCR/URL) needs a final
    # character split. Use a stricter width than the 2.5 chars/token bound.
    safe_chars = max(1, int(token_limit * 2))
    bounded: list[str] = []
    for piece in pieces:
        if estimate_tokens(piece) <= token_limit:
            bounded.append(piece)
            continue
        bounded.extend(piece[offset:offset + safe_chars] for offset in range(0, len(piece), safe_chars))
    return bounded


def split_legal_chunk(content: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split only an oversized legal unit while preserving its legal locator.

    Structure is the primary boundary.  Sentence/paragraph boundaries are used
    only after one Điều/Khoản/Điểm exceeds ``max_tokens``.  Limited overlap is
    applied solely between these sibling parts, never across legal provisions.
    """
    content = content.strip()
    if not content or estimate_tokens(content) <= max_tokens:
        return [content] if content else []

    # Keep paragraphs first; then split each paragraph at sentence boundaries.
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n+", content):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = [piece.strip() for piece in re.split(r"(?<=[.!?;:])\s+", paragraph) if piece.strip()]
        units.extend(sentences or [paragraph])
    flattened = [piece for unit in units for piece in _hard_split(unit, max_tokens)]

    chunks: list[str] = []
    current = ""
    for unit in flattened:
        separator = "\n\n" if current else ""
        if current and estimate_tokens(f"{current}{separator}{unit}") > max_tokens:
            chunks.append(current.strip())
            overlap = _tail_overlap(current, overlap_tokens)
            current = overlap
        separator = "\n\n" if current else ""
        # A hard-split unit can still be too large after prepending overlap.
        if current and estimate_tokens(f"{current}{separator}{unit}") > max_tokens:
            chunks.append(current.strip())
            current = ""
        current = f"{current}{'\n\n' if current else ''}{unit}".strip()
    if current:
        chunks.append(current)
    return chunks


def parse_legal_text(
    text: str,
    chunk_max_tokens: int = 450,
    chunk_split_overlap_tokens: int = 40,
) -> list[ParsedProvision]:
    if chunk_max_tokens <= 0:
        raise ValueError("chunk_max_tokens phải lớn hơn 0")
    if chunk_split_overlap_tokens < 0:
        raise ValueError("chunk_split_overlap_tokens không được âm")
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
    split: list[ParsedProvision] = []
    ordinal = 0
    for item in parsed:
        parts = split_legal_chunk(item.content, chunk_max_tokens, chunk_split_overlap_tokens)
        for chunk_index, part in enumerate(parts, start=1):
            ordinal += 1
            split.append(replace(item, content=part, ordinal=ordinal, chunk_index=chunk_index))
    return split
