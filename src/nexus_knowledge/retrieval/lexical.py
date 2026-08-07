"""Retrieval hit and lexical (BM25) retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..domain.document import Chunk
from ..embedding.hashing import tokenize

__all__ = ["RetrievalHit", "LexicalRetriever"]


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    object_id: str
    score: float
    method: str
    features: dict[str, float] = field(default_factory=dict)


class LexicalRetriever:
    """BM25 lexical retrieval over ingested chunks.

    Indexes are built lazily from the repository on first search and
    invalidated when new chunks are ingested, keeping the pipeline
    deterministic.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: dict[str, str] = {}
        self._term_tf: dict[str, dict[str, int]] = {}
        self._doc_len: dict[str, int] = {}
        self._doc_freq: dict[str, int] = {}
        self._avgdl = 0.0
        self._doc_count = 0
        self._dirty = True

    def invalidate(self) -> None:
        self._dirty = True

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk.text
        self._dirty = True

    def _rebuild(self) -> None:
        if not self._dirty:
            return
        self._term_tf = {}
        self._doc_len = {}
        self._doc_freq = {}
        total_length = 0
        for chunk_id, text in self._chunks.items():
            tokens = tokenize(text)
            self._doc_len[chunk_id] = len(tokens)
            total_length += len(tokens)
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            for token in tf:
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
            self._term_tf[chunk_id] = tf
        self._doc_count = len(self._chunks)
        self._avgdl = total_length / self._doc_count if self._doc_count else 0.0
        self._dirty = False

    def search(
        self,
        tokens: list[str],
        top_k: int = 10,
        metadata_filter: dict[str, object] | None = None,
    ) -> list[RetrievalHit]:
        self._rebuild()
        if not tokens or not self._chunks:
            return []
        scores: dict[str, float] = {}
        for token in tokens:
            df = self._doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (self._doc_count - df + 0.5) / (df + 0.5))
            for chunk_id, length in self._doc_len.items():
                tf = self._term_tf[chunk_id].get(token, 0)
                if tf == 0:
                    continue
                denominator = tf + self.k1 * (1.0 - self.b + self.b * length / self._avgdl)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (tf * (self.k1 + 1.0)) / denominator
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            RetrievalHit(object_id=chunk_id, score=score, method="lexical")
            for chunk_id, score in ranked[:top_k]
        ]
