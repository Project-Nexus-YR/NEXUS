"""Deterministic hashing embedder.

Produces fixed, reproducible dense vectors from text using feature
hashing with signed weights and L2 normalization. Deterministic across
processes and machines (no random state, no downloads).

Used as the default local provider and in tests; a real ML embedding
model can be plugged in later through the same :class:`EmbeddingProvider`
port.
"""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

__all__ = ["tokenize", "FeatureHashEmbedder"]

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-cased alphanumeric token sequence."""
    return [token.lower() for token in _TOKEN_RE.findall(text)]


class FeatureHashEmbedder:
    """Dense deterministic embedding via signed feature hashing."""

    def __init__(self, dimensionality: int = 256) -> None:
        if dimensionality <= 0:
            raise ValueError("dimensionality must be positive")
        self.dimensionality = dimensionality

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensionality, dtype=np.float64)
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensionality
            sign = 1.0 if (int.from_bytes(digest[4:8], "little") & 1) == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(float(np.dot(vector, vector)))
        if norm > 0.0:
            vector /= norm
        return vector
