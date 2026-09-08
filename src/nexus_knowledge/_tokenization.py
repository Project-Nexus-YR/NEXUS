"""Dependency-neutral tokenization shared by lexical and embedding code."""

from __future__ import annotations

import re

__all__ = ["tokenize"]

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-cased alphanumeric token sequence."""
    return [token.lower() for token in _TOKEN_RE.findall(text)]
