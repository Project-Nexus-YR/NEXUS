"""Hybrid retrieval engine.

Implements the retrieval pipeline:

    query -> analysis -> parallel retrieval -> fusion -> features
    -> reranking -> candidates

Four retrieval methods are supported: lexical (BM25), vector (dense),
entity (direct relation grounding) and graph (PPR traversal). Methods
are selectable so experiments can compare lexical/vector/graph-only
against hybrid and hybrid+reranker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..domain.document import Chunk
from ..graph.graph import KnowledgeGraph
from ..port.embeddings import EmbeddingProvider
from ..port.reranker import RerankCandidate
from ..port.repository import KnowledgeRepository
from ..port.vector_store import VectorStore
from .entity import EntityIndex
from .entity_retrieval import EntityRetriever
from .features import FeatureExtractor
from .fusion import ReciprocalRankFusion
from .lexical import LexicalRetriever
from .observability import MethodLatency, RetrievalTrace
from .query import QueryAnalysis, analyze_query
from .rerank import DeterministicReranker
from .vector_graph import GraphRetriever, VectorRetriever

__all__ = ["HybridRetriever", "RankedCandidate", "RetrievalResult"]

DEFAULT_METHODS = ("lexical", "vector", "entity", "graph")


@dataclass(slots=True)
class RankedCandidate:
    chunk_id: str
    chunk: Chunk
    score: float
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 5),
            "text": self.chunk.text,
            "features": {k: round(v, 5) for k, v in self.features.items()},
        }


@dataclass(slots=True)
class RetrievalResult:
    query: str
    request_id: str
    analysis: QueryAnalysis
    candidates: list[RankedCandidate]
    trace: RetrievalTrace

    def top(self, k: int = 5) -> list[RankedCandidate]:
        return self.candidates[:k]

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "request_id": self.request_id,
            "entities": list(self.analysis.entity_names),
            "candidates": [c.to_dict() for c in self.candidates],
            "trace": self.trace.to_dict(),
        }


class HybridRetriever:
    """Deterministic hybrid retrieval orchestrator."""

    def __init__(
        self,
        repository: KnowledgeRepository,
        graph: KnowledgeGraph,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
        entity_index: EntityIndex | None = None,
        active_methods: tuple[str, ...] = DEFAULT_METHODS,
        method_weights: dict[str, float] | None = None,
        reranker: object | None = None,
        lexical_pool_factor: int = 4,
    ) -> None:
        unknown = set(active_methods) - set(DEFAULT_METHODS)
        if unknown:
            raise ValueError(f"unknown retrieval methods: {sorted(unknown)}")
        self._repository = repository
        self._graph = graph
        self._vector_store = vector_store
        self._embedder = embedder
        self._entity_index = entity_index or EntityIndex(repository.entities)
        self._active_methods = active_methods
        self._lexical = LexicalRetriever()
        self._vector = VectorRetriever(embedder, vector_store)
        self._entity = EntityRetriever(graph)
        self._graph_retriever = GraphRetriever(graph, repository.relations)
        self._fusion = ReciprocalRankFusion(method_weights=method_weights)
        self._features = FeatureExtractor(repository, graph)
        self._reranker = reranker if reranker is not None else DeterministicReranker()
        self._lexical_pool_factor = lexical_pool_factor
        self._indexed_chunk_count = -1

    # -- public API ---------------------------------------------------
    def retrieve(
        self,
        text: str,
        top_k: int = 10,
        metadata_filter: dict[str, object] | None = None,
    ) -> RetrievalResult:
        trace = RetrievalTrace(query=text)
        self._sync_lexical_index()

        matched = self._entity_index.match(text)
        analysis = analyze_query(
            text,
            entity_ids=[e.id for e in matched],
            entity_names=[e.canonical for e in matched],
            filters=metadata_filter,
        )

        pool = max(top_k * self._lexical_pool_factor, 20)
        method_results: list[list] = []
        if "lexical" in self._active_methods:
            method_results.append(
                self._timed("lexical", lambda: self._lexical.search(analysis.tokens, top_k=pool), trace)
            )
        if "vector" in self._active_methods:
            method_results.append(
                self._timed(
                    "vector",
                    lambda: self._vector.search(text, top_k=pool, metadata_filter=metadata_filter),
                    trace,
                )
            )
        if "entity" in self._active_methods and analysis.has_entities():
            method_results.append(
                self._timed(
                    "entity",
                    lambda: self._entity.search(list(analysis.entity_ids), top_k=pool),
                    trace,
                )
            )
        if "graph" in self._active_methods and analysis.has_entities():
            method_results.append(
                self._timed(
                    "graph",
                    lambda: self._graph_retriever.search(list(analysis.entity_ids), top_k=pool),
                    trace,
                )
            )

        fused = self._fusion.fuse(method_results)
        contributions = {object_id: contrib for object_id, _, contrib in fused}
        chunk_ids = [object_id for object_id, _, _ in fused]

        pagerank = self._graph.pagerank()
        feature_map = self._features.features_for(chunk_ids, contributions, pagerank)

        candidates: list[RankedCandidate] = []
        chunks = {c.id: c for c in self._repository.chunks.all()}
        for object_id, score, _ in fused:
            chunk = chunks.get(object_id)
            if chunk is None:
                continue
            candidates.append(
                RankedCandidate(
                    chunk_id=object_id,
                    chunk=chunk,
                    score=score,
                    features=feature_map.get(object_id, {}),
                )
            )

        reranker_name = type(self._reranker).__name__
        if candidates:
            rerank_input = [
                RerankCandidate(
                    object_id=c.chunk_id,
                    text=c.chunk.text,
                    score=c.score,
                    features=dict(c.features),
                )
                for c in candidates
            ]
            reranked = self._reranker.rerank(text, rerank_input)
            by_id = {c.chunk_id: c for c in candidates}
            candidates = [by_id[c.object_id] for c in reranked]
            if reranker_name != "Dummy":
                trace.reranker = reranker_name
        else:
            trace.reranker = "none"

        trace.candidate_count = len(fused)
        trace.final_count = len(candidates)
        trace.stop()
        return RetrievalResult(
            query=text,
            request_id=trace.request_id,
            analysis=analysis,
            candidates=candidates[:top_k],
            trace=trace,
        )

    def _timed(self, method: str, fn, trace: RetrievalTrace) -> list:
        started = time.perf_counter()
        results = fn()
        latency = (time.perf_counter() - started) * 1000.0
        trace.method_results.append(MethodLatency(method, len(results), latency))
        return results

    def _sync_lexical_index(self) -> None:
        count = self._repository.chunks.count()
        if count != self._indexed_chunk_count:
            self._lexical.add_chunks(self._repository.chunks.all())
            self._indexed_chunk_count = count
