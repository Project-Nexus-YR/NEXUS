"""Ranking features for retrieval candidates.

Features are computed deterministically from measurable properties of
the knowledge base (evidence counts, entity centrality, source
diversity) and are exposed on every candidate for the visualization
layer and for reranking.
"""

from __future__ import annotations

from ..graph.graph import KnowledgeGraph
from ..port.repository import KnowledgeRepository

__all__ = ["FeatureExtractor"]


class FeatureExtractor:
    """Computes per-chunk ranking features."""

    def __init__(self, repository: KnowledgeRepository, graph: KnowledgeGraph) -> None:
        self._repository = repository
        self._graph = graph
        self._chunk_document: dict[str, str] = {}
        for chunk in repository.chunks.all():
            self._chunk_document[chunk.id] = chunk.document_id
        # map chunk -> relations that cite it as provenance
        self._chunk_relations: dict[str, list[object]] = {}
        for relation in repository.relations.all():
            for chunk_id in relation.provenance:
                self._chunk_relations.setdefault(chunk_id, []).append(relation)
        self._chunk_claims: dict[str, int] = {}
        for claim in repository.claims.all():
            for chunk_id in claim.provenance:
                self._chunk_claims[chunk_id] = self._chunk_claims.get(chunk_id, 0) + 1
        self._evidence_by_chunk: dict[str, int] = {}
        for evidence in repository.evidence.all():
            self._evidence_by_chunk[evidence.chunk_id] = (
                self._evidence_by_chunk.get(evidence.chunk_id, 0) + 1
            )

    def features_for(
        self,
        chunk_ids: list[str],
        fused: dict[str, dict[str, float]],
        pagerank: dict[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        pagerank = pagerank if pagerank is not None else self._graph.pagerank()
        result: dict[str, dict[str, float]] = {}
        for chunk_id in chunk_ids:
            contributions = fused.get(chunk_id, {})
            features = {
                "fused_score": sum(contributions.values()),
                "evidence_count": float(self._evidence_by_chunk.get(chunk_id, 0)),
                "claim_count": float(self._chunk_claims.get(chunk_id, 0)),
                "source_diversity": float(len(self._sources_for(chunk_id))),
            }
            for method, value in contributions.items():
                features[f"{method}_rrf"] = value
            centrality = self._max_centrality(chunk_id, pagerank)
            features["entity_centrality"] = centrality
            result[chunk_id] = features
        return result

    def _sources_for(self, chunk_id: str) -> set[str]:
        sources: set[str] = set()
        for relation in self._chunk_relations.get(chunk_id, []):
            sources.update(relation.source_ids)
        return sources

    def _max_centrality(self, chunk_id: str, pagerank: dict[str, float]) -> float:
        best = 0.0
        for relation in self._chunk_relations.get(chunk_id, []):
            for entity_id in (relation.subject_id, relation.object_id):
                best = max(best, pagerank.get(entity_id, 0.0))
        return best
