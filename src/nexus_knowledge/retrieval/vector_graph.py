"""Vector and graph-backed retrieval."""

from __future__ import annotations

from ..embedding.hashing import tokenize
from ..graph.graph import KnowledgeGraph
from ..port.embeddings import EmbeddingProvider
from ..port.repository import RelationRepository
from ..port.vector_store import VectorStore
from .lexical import RetrievalHit

__all__ = ["VectorRetriever", "GraphRetriever"]


class VectorRetriever:
    """Dense retrieval over the vector store."""

    def __init__(self, embedder: EmbeddingProvider, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def search(
        self,
        text: str,
        top_k: int = 10,
        metadata_filter: dict[str, object] | None = None,
    ) -> list[RetrievalHit]:
        embedding = self._embedder.embed(text)
        hits = self._vector_store.query(
            embedding.vector, top_k=top_k, metadata_filter=metadata_filter
        )
        return [
            RetrievalHit(object_id=hit.object_id, score=hit.score, method="vector")
            for hit in hits
        ]


class GraphRetriever:
    """Graph-aware retrieval.

    Seeded on query entities, it runs personalized PageRank and a bounded
    traversal, then maps traversed relations back to the chunks that
    provided their provenance.
    """

    def __init__(self, graph: KnowledgeGraph, relations: RelationRepository) -> None:
        self._graph = graph
        self._relations = relations

    def search(
        self,
        entity_ids: list[str],
        top_k: int = 10,
        depth: int = 2,
    ) -> list[RetrievalHit]:
        if not entity_ids:
            return []
        ppr = self._graph.personalized_pagerank(entity_ids)
        _, edges = self._graph.traversal(entity_ids, depth=depth, direction="both")
        chunk_scores: dict[str, float] = {}
        for edge in edges:
            relation = self._graph.get_relation(edge.relation_id)
            if relation is None:
                continue
            entity_score = ppr.get(edge.subject_id, 0.0) + ppr.get(edge.object_id, 0.0)
            contribution = entity_score * edge.weight
            for chunk_id in relation.provenance:
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + contribution
        ranked = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)
        return [
            RetrievalHit(object_id=chunk_id, score=score, method="graph")
            for chunk_id, score in ranked[:top_k]
        ]
