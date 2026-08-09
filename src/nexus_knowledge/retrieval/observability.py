"""Observability records for retrieval.

Every retrieval operation emits a structured :class:`RetrievalTrace`
containing the request id, query, contributing methods, candidate
counts, ranking features and per-method latencies. These feed the
system's visualization and evaluation layers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..domain.ids import new_id

__all__ = ["RetrievalTrace", "MethodLatency"]


@dataclass(frozen=True, slots=True)
class MethodLatency:
    method: str
    candidates: int
    latency_ms: float


@dataclass(slots=True)
class RetrievalTrace:
    query: str
    request_id: str = field(default_factory=lambda: new_id("req"))
    method_results: list[MethodLatency] = field(default_factory=list)
    candidate_count: int = 0
    final_count: int = 0
    reranker: str = ""
    ranking_features: dict[str, float] = field(default_factory=dict)
    graph_paths: list[list[str]] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    total_ms: float = 0.0

    def stop(self) -> None:
        self.total_ms = (time.perf_counter() - self.started_at) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "method_results": [
                {
                    "method": m.method,
                    "candidates": m.candidates,
                    "latency_ms": round(m.latency_ms, 3),
                }
                for m in self.method_results
            ],
            "candidate_count": self.candidate_count,
            "final_count": self.final_count,
            "reranker": self.reranker,
            "ranking_features": self.ranking_features,
            "graph_paths": self.graph_paths,
            "total_ms": round(self.total_ms, 3),
        }
