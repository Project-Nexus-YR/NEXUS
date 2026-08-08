"""Entity-grounded chunk retrieval."""

from __future__ import annotations

from ..graph.graph import KnowledgeGraph
from .lexical import RetrievalHit

__all__ = ["EntityRetriever"]


class EntityRetriever:
    """Retrieves chunks that ground query entities via direct relations.

    For every query entity, the direct relations incident to it point
    back to the chunks that produced them (provenance), which are scored
    by entity confidence times relation weight.
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

    def search(
        self,
        entity_ids: list[str],
        top_k: int = 10,
    ) -> list[RetrievalHit]:
        chunk_scores: dict[str, float] = {}
        for entity_id in entity_ids:
            for edge in self._graph.neighbors(entity_id, "both"):
                relation = self._graph.get_relation(edge.relation_id)
                if relation is None:
                    continue
                for chunk_id in relation.provenance:
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + edge.weight
        ranked = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)
        return [
            RetrievalHit(object_id=chunk_id, score=score, method="entity")
            for chunk_id, score in ranked[:top_k]
        ]
