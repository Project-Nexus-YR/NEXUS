"""GraphRAG engine.

Given a query:

1. identify relevant entities
2. retrieve candidate chunks (hybrid)
3. build a bounded evidence subgraph around the query entities
4. rank paths within the subgraph
5. collect claims, evidence and sources for the subgraph
6. aggregate confidence

The output is a self-contained :class:`EvidenceGraph` that explicitly
carries query, entities, relations, claims, evidence, sources and
confidence — never a black-box text answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.claim import Claim, Evidence
from ..domain.entity import Entity, Relation
from ..domain.source import Source
from ..graph.graph import KnowledgeGraph, Path
from ..port.repository import KnowledgeRepository
from .hybrid import HybridRetriever, RetrievalResult

__all__ = ["GraphRAGEngine", "EvidenceGraph"]


@dataclass(slots=True)
class EvidenceGraph:
    query: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)
    confidence: float = 0.0
    retrieval: RetrievalResult | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "entities": [e.canonical for e in self.entities],
            "relations": [
                {"subject": r.subject_id, "predicate": r.predicate, "object": r.object_id}
                for r in self.relations
            ],
            "claims": [c.text for c in self.claims],
            "evidence_count": len(self.evidence),
            "sources": [s.reference for s in self.sources],
            "paths": [list(p.nodes) for p in self.paths],
            "confidence": round(self.confidence, 4),
        }


class GraphRAGEngine:
    """Builds evidence graphs grounded in the knowledge base."""

    def __init__(
        self,
        repository: KnowledgeRepository,
        graph: KnowledgeGraph,
        retriever: HybridRetriever,
    ) -> None:
        self._repository = repository
        self._graph = graph
        self._retriever = retriever

    def query(
        self,
        text: str,
        top_k: int = 8,
        depth: int = 2,
    ) -> EvidenceGraph:
        retrieval = self._retriever.retrieve(text, top_k=top_k)
        entity_ids = list(retrieval.analysis.entity_ids)
        graph = EvidenceGraph(query=text, retrieval=retrieval)

        subgraph = self._graph.subgraph(entity_ids, depth=depth, max_nodes=200, max_edges=800)
        graph.entities = [subgraph.nodes[eid] for eid in sorted(subgraph.nodes)]
        relation_ids = {edge.relation_id for edge in subgraph.edges}
        relations = {
            r.id: r
            for r in self._repository.relations.all()
            if r.id in relation_ids
        }
        graph.relations = [relations[rid] for rid in sorted(relations)]

        chunk_ids = set()
        for edge in subgraph.edges:
            relation = relations.get(edge.relation_id)
            if relation is not None:
                chunk_ids.update(relation.provenance)

        claims_by_id = {c.id: c for c in self._repository.claims.all()}
        evidence_by_id = {e.id: e for e in self._repository.evidence.all()}
        selected_claims: dict[str, Claim] = {}
        for claim in claims_by_id.values():
            if set(claim.provenance) & chunk_ids:
                selected_claims[claim.id] = claim
            elif claim.subject and any(
                e.name.lower() == claim.subject.lower() for e in graph.entities
            ):
                selected_claims[claim.id] = claim
        graph.claims = [selected_claims[cid] for cid in sorted(selected_claims)]

        evidence_ids = set()
        for claim in graph.claims:
            evidence_ids.update(claim.supporting_evidence)
            evidence_ids.update(claim.contradicting_evidence)
        graph.evidence = [
            evidence_by_id[eid] for eid in sorted(evidence_ids) if eid in evidence_by_id
        ]

        source_ids = set()
        for claim in graph.claims:
            source_ids.update(claim.source_ids)
        for relation in graph.relations:
            source_ids.update(relation.source_ids)
        sources = {s.id: s for s in self._repository.sources.all()}
        graph.sources = [sources[sid] for sid in sorted(source_ids) if sid in sources]

        if len(entity_ids) >= 2:
            graph.paths = self._ranked_paths(entity_ids)

        graph.confidence = self._aggregate_confidence(graph.claims, entity_ids)
        return graph

    def _ranked_paths(self, entity_ids: list[str]) -> list[Path]:
        paths: list[Path] = []
        seen: set[tuple[str, ...]] = set()
        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                for path in self._graph.paths(entity_ids[i], entity_ids[j], max_length=4, max_paths=10):
                    if path.nodes in seen:
                        continue
                    seen.add(path.nodes)
                    paths.append(path)
        paths.sort(key=lambda p: p.weight, reverse=True)
        return paths[:10]

    def _aggregate_confidence(self, claims: list[Claim], entity_ids: list[str]) -> float:
        if not claims:
            if not entity_ids:
                return 0.0
            ppr = self._graph.personalized_pagerank(entity_ids)
            return min(1.0, sum(ppr.values()) * 2.0)
        return sum(float(c.confidence) for c in claims) / len(claims)
