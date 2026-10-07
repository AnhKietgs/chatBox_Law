"""Conservative, dependency-free token estimates for Vietnamese legal text.

The actual tokenizer lives inside Ollama and may change with ``OLLAMA_MODEL``.
Loading a HuggingFace tokenizer on every API/worker process would add network
and memory overhead, so budgeting uses a deliberately pessimistic bound:

* Vietnamese is largely monosyllabic; one written syllable commonly becomes
  1.5--2 BPE tokens.  We reserve 2 tokens per lexical item.
* Long identifiers, URLs, numbers and text without spaces are covered by the
  character estimate (2.5 characters per token).

Taking the larger estimate makes prompt construction fail closed: it may invoke
MapReduce a little earlier, but it does not risk exceeding the model context.
"""
from __future__ import annotations

import math
import re

VIETNAMESE_TOKENS_PER_LEXICAL_ITEM = 2.0
CHARACTERS_PER_TOKEN = 2.5
_LEXICAL_ITEM_RE = re.compile(r"[^\W_]+", re.UNICODE)


def estimate_vietnamese_tokens(text: str) -> int:
    """Return a safety-first estimate for arbitrary Vietnamese/legal text."""
    if not text:
        return 0
    lexical_items = len(_LEXICAL_ITEM_RE.findall(text))
    by_lexical_items = math.ceil(lexical_items * VIETNAMESE_TOKENS_PER_LEXICAL_ITEM)
    by_characters = math.ceil(len(text) / CHARACTERS_PER_TOKEN)
    return max(1, by_lexical_items, by_characters)

