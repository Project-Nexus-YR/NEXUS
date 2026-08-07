"""Candidate fusion for hybrid retrieval.

Candidates from independent retrieval methods are fused with weighted
reciprocal-rank fusion (RRF), not simple concatenation. This gives a
deterministic, per-method-contributing score.
"""

from __future__ import annotations

from .lexical import RetrievalHit

__all__ = ["ReciprocalRankFusion"]


class ReciprocalRankFusion:
    """Weighted reciprocal-rank fusion of retrieval results."""

    def __init__(self, k: int = 60, method_weights: dict[str, float] | None = None) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self._weights = dict(method_weights or {})

    def fuse(self, results: list[list[RetrievalHit]]) -> list[tuple[str, float, dict[str, float]]]:
        """Merge per-method candidate lists into ``(object_id, score, contributions)``."""
        contributions: dict[str, dict[str, float]] = {}
        for method_results in results:
            for rank, hit in enumerate(method_results):
                contributions.setdefault(hit.object_id, {})[hit.method] = (
                    self._weights.get(hit.method, 1.0) / (self.k + rank)
                )
        fused = []
        for object_id, by_method in contributions.items():
            score = sum(by_method.values())
            fused.append((object_id, score, by_method))
        fused.sort(key=lambda item: item[1], reverse=True)
        return fused
