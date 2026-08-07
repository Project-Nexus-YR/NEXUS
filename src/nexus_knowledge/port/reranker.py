"""Reranker port and retrieval candidate record."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

__all__ = ["RerankCandidate", "Reranker"]


@dataclass(slots=True)
class RerankCandidate:
    object_id: str
    text: str
    score: float  # pre-rerank score (higher is better)
    features: dict[str, float] = field(default_factory=dict)


class Reranker(Protocol):
    """Re-orders retrieval candidates for a query."""

    def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[RerankCandidate]:
        """Return the candidates in reranked order (best first)."""
        ...
