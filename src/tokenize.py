"""Tokenizer shared by BM25 and the extractive answer fallback."""

from __future__ import annotations

import re

# Keep hyphenated identifiers such as ERR-VEC-404 and POL-RET-90 as one token.
# A whitespace split would do that too, but it also glues trailing punctuation
# onto the token, so "ERR-VEC-404," would miss the query "ERR-VEC-404".
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())
