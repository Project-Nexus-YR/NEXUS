"""Local in-memory vector store.

Deterministic cosine-similarity search over :class:`Embedding` records.
The interface (see :class:`VectorStore`) is suitable for a distributed
vector backend later; this implementation is the reference backend.
"""

from __future__ import annotations

import numpy as np

from ..port.embeddings import Embedding
from ..port.vector_store import VectorHit, VectorStore

__all__ = ["LocalVectorStore"]


class LocalVectorStore:
    """Implements :class:`VectorStore` with exact cosine similarity."""

    def __init__(self) -> None:
        self._entries: dict[str, Embedding] = {}

    def upsert(self, embedding: Embedding) -> None:
        self._entries[embedding.object_id] = embedding

    def upsert_many(self, embeddings: list[Embedding]) -> None:
        for embedding in embeddings:
            self.upsert(embedding)

    def _matches(self, metadata: dict[str, object], filt: dict[str, object]) -> bool:
        return all(metadata.get(key) == value for key, value in filt.items())

    def query(
        self,
        vector: tuple[float, ...],
        top_k: int = 10,
        metadata_filter: dict[str, object] | None = None,
    ) -> list[VectorHit]:
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        query = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(query))
        if norm > 0.0:
            query = query / norm
        scored: list[tuple[float, str, dict[str, object]]] = []
        for object_id, embedding in self._entries.items():
            if metadata_filter and not self._matches(embedding.metadata, metadata_filter):
                continue
            candidate = np.asarray(embedding.vector, dtype=np.float64)
            cnorm = float(np.linalg.norm(candidate))
            if cnorm > 0.0:
                candidate = candidate / cnorm
            score = float(np.dot(query, candidate))
            scored.append((score, object_id, dict(embedding.metadata)))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [VectorHit(object_id=oid, score=score, metadata=meta) for score, oid, meta in scored[:top_k]]

    def get(self, object_id: str) -> Embedding | None:
        return self._entries.get(object_id)

    def delete(self, object_id: str) -> bool:
        return self._entries.pop(object_id, None) is not None

    def size(self) -> int:
        return len(self._entries)
