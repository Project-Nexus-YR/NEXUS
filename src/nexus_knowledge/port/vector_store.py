"""Vector store port.

Suitable for a local in-memory backend today and a distributed vector
database later. The interface exposes only upsert/query/delete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .embeddings import Embedding

__all__ = ["VectorHit", "VectorStore"]


@dataclass(frozen=True, slots=True)
class VectorHit:
    object_id: str
    score: float  # higher is better
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    """Stores and queries embeddings by similarity."""

    def upsert(self, embedding: Embedding) -> None: ...

    def upsert_many(self, embeddings: list[Embedding]) -> None: ...

    def query(
        self,
        vector: tuple[float, ...],
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]: ...

    def get(self, object_id: str) -> Embedding | None: ...

    def delete(self, object_id: str) -> bool: ...

    def size(self) -> int: ...
