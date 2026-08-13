"""Read-only, bounded knowledge exploration for visual clients.

The explorer is deliberately transport neutral.  It turns the existing knowledge
domain objects into a heterogeneous graph projection without exposing repositories or
graph-storage adapters to the HTTP layer.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from difflib import SequenceMatcher
from enum import Enum
from threading import RLock
from typing import Any

from nexus_knowledge.domain.claim import Claim, Evidence
from nexus_knowledge.domain.contradiction import Contradiction
from nexus_knowledge.domain.entity import Entity, Relation
from nexus_knowledge.domain.knowledge_gap import KnowledgeGap

from .engine import KnowledgeEngine

NODE_KINDS = frozenset(
    {"entity", "claim", "evidence", "document", "source", "gap", "investigation"}
)


@dataclass(frozen=True, slots=True)
class ExplorerFilters:
    """Server-side graph slicing controls with conservative hard bounds."""

    node_kinds: frozenset[str] = NODE_KINDS
    relation_types: frozenset[str] = frozenset()
    verification_states: frozenset[str] = frozenset()
    min_confidence: float = 0.0
    max_nodes: int = 750
    max_edges: int = 2_500

    def __post_init__(self) -> None:
        unknown = self.node_kinds - NODE_KINDS
        if unknown:
            raise ValueError(f"unknown node kinds: {', '.join(sorted(unknown))}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between zero and one")
        if not 1 <= self.max_nodes <= 2_000:
            raise ValueError("max_nodes must be between 1 and 2000")
        if not 1 <= self.max_edges <= 10_000:
            raise ValueError("max_edges must be between 1 and 10000")


@dataclass(frozen=True, slots=True)
class ExplorerNode:
    id: str
    kind: str
    label: str
    subtitle: str = ""
    confidence: float | None = None
    uncertainty: float | None = None
    importance: float | None = None
    verification_state: str | None = None
    status: str | None = None
    degree: int = 0
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExplorerEdge:
    id: str
    source: str
    target: str
    kind: str
    label: str
    confidence: float | None = None
    verification_state: str | None = None
    directed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExplorerGraph:
    nodes: tuple[ExplorerNode, ...]
    edges: tuple[ExplorerEdge, ...]
    total_nodes: int
    total_edges: int
    truncated: bool
    snapshot_id: str
    statistics: dict[str, int | float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "truncated": self.truncated,
            "snapshot_id": self.snapshot_id,
            "statistics": self.statistics,
        }


@dataclass(frozen=True, slots=True)
class ExplorerNodeDetail:
    node: ExplorerNode
    attributes: dict[str, Any]
    connections: tuple[dict[str, Any], ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    contradictions: tuple[dict[str, Any], ...] = ()
    gaps: tuple[dict[str, Any], ...] = ()
    history: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node.to_dict(),
            "attributes": self.attributes,
            "connections": list(self.connections),
            "evidence": list(self.evidence),
            "provenance": self.provenance,
            "contradictions": list(self.contradictions),
            "gaps": list(self.gaps),
            "history": list(self.history),
        }


class KnowledgeExplorer:
    """Build bounded projections through the existing ``KnowledgeEngine`` boundary."""

    def __init__(self, knowledge: KnowledgeEngine) -> None:
        self._knowledge = knowledge
        self._cache_lock = RLock()
        self._analysis_signature = ""
        self._cached_gaps: tuple[KnowledgeGap, ...] = ()
        self._cached_contradictions: tuple[Contradiction, ...] = ()

    def graph(self, filters: ExplorerFilters | None = None) -> ExplorerGraph:
        filters = filters or ExplorerFilters()
        nodes, edges = self._project()
        return self._slice(nodes, edges, filters)

    def neighborhood(
        self,
        node_id: str,
        *,
        depth: int = 1,
        filters: ExplorerFilters | None = None,
    ) -> ExplorerGraph:
        if depth not in {1, 2, 3}:
            raise ValueError("neighborhood depth must be 1, 2, or 3")
        filters = filters or ExplorerFilters(max_nodes=500, max_edges=2_000)
        nodes, edges = self._project()
        if node_id not in nodes:
            raise KeyError(node_id)
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in edges.values():
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
        visited = {node_id}
        frontier = {node_id}
        for _ in range(depth):
            frontier = {
                neighbor
                for current in frontier
                for neighbor in adjacency.get(current, ())
                if neighbor not in visited
            }
            remaining = max(0, filters.max_nodes - len(visited))
            if len(frontier) > remaining:
                frontier = set(
                    sorted(frontier, key=lambda item: self._node_rank(nodes[item]), reverse=True)[
                        :remaining
                    ]
                )
            visited.update(frontier)
            if not frontier or len(visited) >= filters.max_nodes:
                break
        local_nodes = {key: value for key, value in nodes.items() if key in visited}
        local_edges = {
            key: value
            for key, value in edges.items()
            if value.source in visited and value.target in visited
        }
        return self._slice(local_nodes, local_edges, filters, force_ids={node_id})

    def search(self, query: str, *, limit: int = 25) -> list[dict[str, Any]]:
        text = " ".join(query.casefold().split())
        if not text:
            return []
        if not 1 <= limit <= 100:
            raise ValueError("search limit must be between 1 and 100")
        nodes, _ = self._project()
        ranked: list[tuple[float, ExplorerNode]] = []
        for node in nodes.values():
            haystack = " ".join((node.label, node.subtitle, node.kind)).casefold()
            if text in haystack:
                score = 2.0 + len(text) / max(1, len(haystack))
            else:
                score = SequenceMatcher(None, text, haystack).ratio()
            if score >= 0.32:
                ranked.append((score + min(node.degree, 20) / 100.0, node))
        ranked.sort(key=lambda item: (-item[0], item[1].label.casefold(), item[1].id))
        return [{"score": round(score, 4), **node.to_dict()} for score, node in ranked[:limit]]

    def node_detail(self, node_id: str) -> ExplorerNodeDetail:
        nodes, edges = self._project()
        node = nodes.get(node_id)
        if node is None:
            raise KeyError(node_id)
        connections = self._connection_refs(node_id, nodes, edges.values())
        if node.kind == "entity":
            return self._entity_detail(node, connections)
        if node.kind == "claim":
            return self._claim_detail(node, connections)
        if node.kind == "evidence":
            return self._evidence_detail(node, connections)
        if node.kind == "document":
            return self._document_detail(node, connections)
        if node.kind == "source":
            return self._source_detail(node, connections)
        return self._gap_detail(node, connections)

    def gaps(self) -> list[dict[str, Any]]:
        gaps = sorted(
            self._current_gaps(),
            key=lambda item: (-item.priority, -item.importance, item.id),
        )
        investigations = self._knowledge.repository.investigations.all()
        by_gap: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in investigations:
            by_gap[item.gap_id].append(self._plain(item))
        return [
            {
                **self._plain(gap),
                "priority": gap.priority,
                "investigations": by_gap.get(gap.id, []),
                "resolved": bool(gap.metadata.get("resolved", False)),
            }
            for gap in gaps
        ]

    def contradictions(self) -> list[dict[str, Any]]:
        return [self._contradiction_dict(item) for item in self._current_contradictions()]

    def sources(self) -> list[dict[str, Any]]:
        """Return source navigation records independently of the active graph slice."""
        repository = self._knowledge.repository
        return [
            ExplorerNode(
                id=item.id,
                kind="source",
                label=item.title,
                subtitle=item.kind,
                degree=len(repository.documents.by_source(item.id)),
                created_at=item.ingested_at,
                metadata={"reference": item.reference, **dict(item.metadata)},
            ).to_dict()
            for item in repository.sources.all()
        ]

    def _project(self) -> tuple[dict[str, ExplorerNode], dict[str, ExplorerEdge]]:
        repository = self._knowledge.repository
        entities = repository.entities.all()
        relations = repository.relations.all()
        claims = repository.claims.all()
        evidence = repository.evidence.all()
        documents = repository.documents.all()
        sources = repository.sources.all()
        contradictions = self._current_contradictions()
        gaps = self._current_gaps()

        degree: dict[str, int] = defaultdict(int)
        for relation in relations:
            degree[relation.subject_id] += 1
            degree[relation.object_id] += 1
        evidence_count: dict[str, int] = defaultdict(int)
        for item in evidence:
            evidence_count[item.claim_id] += 1

        nodes: dict[str, ExplorerNode] = {}
        for entity in entities:
            nodes[entity.id] = ExplorerNode(
                id=entity.id,
                kind="entity",
                label=entity.name,
                subtitle=entity.entity_type,
                degree=degree[entity.id],
                metadata={
                    "entity_type": entity.entity_type,
                    "canonical_name": entity.canonical,
                    "aliases": list(entity.aliases),
                    **dict(entity.metadata),
                },
            )
        for claim in claims:
            nodes[claim.id] = ExplorerNode(
                id=claim.id,
                kind="claim",
                label=claim.text,
                subtitle=claim.predicate or "claim",
                confidence=float(claim.confidence),
                verification_state=claim.verification_state.value,
                status=claim.verification_state.value,
                degree=evidence_count[claim.id],
                created_at=claim.created_at,
                updated_at=claim.updated_at,
                metadata=dict(claim.metadata),
            )
        for item in evidence:
            nodes[item.id] = ExplorerNode(
                id=item.id,
                kind="evidence",
                label=item.text,
                subtitle=item.role,
                confidence=item.quality,
                status=item.role,
                degree=2,
                created_at=item.extracted_at,
                metadata={"claim_id": item.claim_id, "document_id": item.document_id},
            )
        for document in documents:
            nodes[document.id] = ExplorerNode(
                id=document.id,
                kind="document",
                label=document.title,
                subtitle=document.content_type,
                degree=len(repository.chunks.by_document(document.id)),
                created_at=document.ingested_at,
                metadata=dict(document.metadata),
            )
        for source in sources:
            nodes[source.id] = ExplorerNode(
                id=source.id,
                kind="source",
                label=source.title,
                subtitle=source.kind,
                degree=len(repository.documents.by_source(source.id)),
                created_at=source.ingested_at,
                metadata={"reference": source.reference, **dict(source.metadata)},
            )
        for gap in gaps:
            nodes[gap.id] = ExplorerNode(
                id=gap.id,
                kind="gap",
                label=gap.description,
                subtitle=gap.kind,
                uncertainty=gap.uncertainty,
                importance=gap.importance,
                status="resolved" if gap.metadata.get("resolved") else "unresolved",
                degree=len(gap.affected_entities)
                + len(gap.affected_relations)
                + len(gap.affected_claims),
                created_at=gap.created_at,
                metadata={"reason": gap.reason, **dict(gap.metadata)},
            )

        edges: dict[str, ExplorerEdge] = {}
        for relation in relations:
            if relation.subject_id in nodes and relation.object_id in nodes:
                edges[relation.id] = self._relation_edge(relation)
        entity_names = self._entity_name_index(entities)
        for claim in claims:
            subject_id = entity_names.get(self._normalize(claim.subject))
            object_id = entity_names.get(self._normalize(claim.object))
            if subject_id is not None:
                edge_id = f"claim-subject:{claim.id}:{subject_id}"
                edges[edge_id] = ExplorerEdge(
                    edge_id,
                    subject_id,
                    claim.id,
                    "claim_subject",
                    claim.predicate or "asserts",
                    float(claim.confidence),
                    claim.verification_state.value,
                )
            if object_id is not None:
                edge_id = f"claim-object:{claim.id}:{object_id}"
                edges[edge_id] = ExplorerEdge(
                    edge_id,
                    claim.id,
                    object_id,
                    "claim_object",
                    claim.predicate or "references",
                    float(claim.confidence),
                    claim.verification_state.value,
                )
        for item in evidence:
            if item.claim_id in nodes:
                edge_id = f"evidence-claim:{item.id}:{item.claim_id}"
                edges[edge_id] = ExplorerEdge(
                    edge_id,
                    item.id,
                    item.claim_id,
                    item.role,
                    item.role,
                    item.quality,
                )
            if item.document_id in nodes:
                edge_id = f"evidence-document:{item.id}:{item.document_id}"
                edges[edge_id] = ExplorerEdge(
                    edge_id,
                    item.id,
                    item.document_id,
                    "provenance",
                    "from document",
                )
        for document in documents:
            if document.source_id in nodes:
                edge_id = f"document-source:{document.id}:{document.source_id}"
                edges[edge_id] = ExplorerEdge(
                    edge_id,
                    document.id,
                    document.source_id,
                    "provenance",
                    "from source",
                )
        for gap in gaps:
            affected = (*gap.affected_entities, *gap.affected_relations, *gap.affected_claims)
            for target in affected:
                if target not in nodes:
                    relation = repository.relations.get(target)
                    if relation is not None:
                        for entity_id in (relation.subject_id, relation.object_id):
                            self._add_gap_edge(edges, gap.id, entity_id)
                    continue
                self._add_gap_edge(edges, gap.id, target)
        for contradiction in contradictions:
            if contradiction.claim_a_id in nodes and contradiction.claim_b_id in nodes:
                edges[contradiction.id] = ExplorerEdge(
                    contradiction.id,
                    contradiction.claim_a_id,
                    contradiction.claim_b_id,
                    "contradiction",
                    contradiction.kind,
                    contradiction.strength,
                    "contradicted",
                    directed=False,
                    metadata={
                        "description": contradiction.description,
                        "evidence_a": list(contradiction.evidence_a),
                        "evidence_b": list(contradiction.evidence_b),
                        **dict(contradiction.metadata),
                    },
                )
        return nodes, edges

    def _slice(
        self,
        nodes: Mapping[str, ExplorerNode],
        edges: Mapping[str, ExplorerEdge],
        filters: ExplorerFilters,
        *,
        force_ids: set[str] | None = None,
    ) -> ExplorerGraph:
        force_ids = force_ids or set()
        eligible = {
            key: node
            for key, node in nodes.items()
            if node.kind in filters.node_kinds
            and (
                not filters.verification_states
                or node.verification_state is None
                or node.verification_state in filters.verification_states
            )
            and (node.confidence is None or node.confidence >= filters.min_confidence)
        }
        ordered = sorted(
            eligible.values(),
            key=lambda item: (item.id not in force_ids, -self._node_rank(item), item.id),
        )
        selected = {node.id: node for node in ordered[: filters.max_nodes]}
        eligible_edges = [
            edge
            for edge in edges.values()
            if edge.source in eligible
            and edge.target in eligible
            and (not filters.relation_types or edge.kind in filters.relation_types)
            and (edge.confidence is None or edge.confidence >= filters.min_confidence)
            and (
                not filters.verification_states
                or edge.verification_state is None
                or edge.verification_state in filters.verification_states
            )
        ]
        selected_edges = [
            edge
            for edge in eligible_edges
            if edge.source in selected and edge.target in selected
        ]
        selected_edges.sort(
            key=lambda item: (-(item.confidence if item.confidence is not None else 0.5), item.id)
        )
        selected_edges = selected_edges[: filters.max_edges]
        selected_nodes = tuple(selected[key] for key in sorted(selected))
        edge_tuple = tuple(selected_edges)
        snapshot = self._snapshot_id(selected_nodes, edge_tuple)
        return ExplorerGraph(
            selected_nodes,
            edge_tuple,
            total_nodes=len(eligible),
            total_edges=len(eligible_edges),
            truncated=(
                len(selected) < len(eligible) or len(selected_edges) < len(eligible_edges)
            ),
            snapshot_id=snapshot,
            statistics=self._knowledge.graph_statistics(),
        )

    def _entity_detail(
        self, node: ExplorerNode, connections: tuple[dict[str, Any], ...]
    ) -> ExplorerNodeDetail:
        entity = self._require(self._knowledge.repository.entities.get(node.id), node.id)
        names = {
            self._normalize(entity.name),
            self._normalize(entity.canonical),
            *map(self._normalize, entity.aliases),
        }
        claims = [
            item
            for item in self._knowledge.repository.claims.all()
            if self._normalize(item.subject) in names or self._normalize(item.object) in names
        ]
        gaps = [
            self._gap_summary(item)
            for item in self._current_gaps()
            if node.id in item.affected_entities
        ]
        return ExplorerNodeDetail(
            node,
            attributes=self._plain(entity),
            connections=connections,
            evidence=tuple(self._claim_summary(item) for item in claims),
            gaps=tuple(gaps),
        )

    def _claim_detail(
        self, node: ExplorerNode, connections: tuple[dict[str, Any], ...]
    ) -> ExplorerNodeDetail:
        claim = self._require(self._knowledge.repository.claims.get(node.id), node.id)
        evidence = tuple(
            self._evidence_dict(item)
            for item in self._knowledge.repository.evidence.by_claim(claim.id)
        )
        contradictions = tuple(
            self._contradiction_dict(item)
            for item in self._current_contradictions()
            if claim.id in {item.claim_a_id, item.claim_b_id}
        )
        gaps = tuple(
            self._gap_summary(item)
            for item in self._current_gaps()
            if claim.id in item.affected_claims
        )
        provenance = self._knowledge.provenance(claim.id).to_dict()
        history = tuple(
            item
            for item in (
                {"event": "created", "at": claim.created_at},
                {"event": "updated", "at": claim.updated_at},
                {"event": "observed", "at": claim.observed_at},
            )
            if item["at"]
        )
        return ExplorerNodeDetail(
            node,
            self._plain(claim),
            connections,
            evidence,
            provenance,
            contradictions,
            gaps,
            history,
        )

    def _evidence_detail(
        self, node: ExplorerNode, connections: tuple[dict[str, Any], ...]
    ) -> ExplorerNodeDetail:
        item = self._require(self._knowledge.repository.evidence.get(node.id), node.id)
        document = self._knowledge.repository.documents.get(item.document_id)
        source = (
            None if document is None else self._knowledge.repository.sources.get(document.source_id)
        )
        provenance = {
            "claim_id": item.claim_id,
            "chunk_id": item.chunk_id,
            "document": None if document is None else self._plain(document),
            "source": None if source is None else self._plain(source),
        }
        return ExplorerNodeDetail(
            node,
            self._plain(item),
            connections,
            provenance=provenance,
            history=({"event": "extracted", "at": item.extracted_at},),
        )

    def _document_detail(
        self, node: ExplorerNode, connections: tuple[dict[str, Any], ...]
    ) -> ExplorerNodeDetail:
        document = self._require(self._knowledge.repository.documents.get(node.id), node.id)
        chunks = self._knowledge.repository.chunks.by_document(document.id)
        evidence = [
            self._evidence_dict(item)
            for item in self._knowledge.repository.evidence.all()
            if item.document_id == document.id
        ]
        return ExplorerNodeDetail(
            node,
            {
                **self._plain(document),
                "text_preview": document.text[:2_000],
                "chunk_count": len(chunks),
            },
            connections,
            tuple(evidence),
            {"source_id": document.source_id, "chunk_ids": [item.id for item in chunks]},
            history=({"event": "ingested", "at": document.ingested_at},),
        )

    def _source_detail(
        self, node: ExplorerNode, connections: tuple[dict[str, Any], ...]
    ) -> ExplorerNodeDetail:
        source = self._require(self._knowledge.repository.sources.get(node.id), node.id)
        documents = self._knowledge.repository.documents.by_source(source.id)
        return ExplorerNodeDetail(
            node,
            self._plain(source),
            connections,
            provenance={"documents": [self._plain(item) for item in documents]},
            history=({"event": "ingested", "at": source.ingested_at},),
        )

    def _gap_detail(
        self, node: ExplorerNode, connections: tuple[dict[str, Any], ...]
    ) -> ExplorerNodeDetail:
        gap = self._require(self._knowledge.repository.gaps.get(node.id), node.id)
        investigations = [
            self._plain(item)
            for item in self._knowledge.repository.investigations.all()
            if item.gap_id == gap.id
        ]
        return ExplorerNodeDetail(
            node,
            {**self._gap_dict(gap), "investigations": investigations},
            connections,
            gaps=(self._gap_dict(gap),),
            history=({"event": "identified", "at": gap.created_at},),
        )

    def _connection_refs(
        self,
        node_id: str,
        nodes: Mapping[str, ExplorerNode],
        edges: Iterable[ExplorerEdge],
    ) -> tuple[dict[str, Any], ...]:
        result: list[dict[str, Any]] = []
        for edge in edges:
            if node_id not in {edge.source, edge.target}:
                continue
            other_id = edge.target if edge.source == node_id else edge.source
            other = nodes.get(other_id)
            if other is None:
                continue
            result.append(
                {
                    "edge": edge.to_dict(),
                    "node": {
                        "id": other.id,
                        "kind": other.kind,
                        "label": other.label,
                        "subtitle": other.subtitle,
                    },
                }
            )
        result.sort(key=lambda item: (item["node"]["kind"], item["node"]["label"]))
        return tuple(result)

    def _current_gaps(self) -> list[KnowledgeGap]:
        self._refresh_analysis()
        return list(self._cached_gaps)

    def _current_contradictions(self) -> list[Contradiction]:
        self._refresh_analysis()
        return list(self._cached_contradictions)

    def _refresh_analysis(self) -> None:
        signature = self._knowledge_signature()
        with self._cache_lock:
            if signature == self._analysis_signature:
                return
            contradictions = tuple(self._knowledge.detect_contradictions())
            gaps = tuple(self._knowledge.find_knowledge_gaps())
            self._cached_contradictions = contradictions
            self._cached_gaps = gaps
            self._analysis_signature = signature

    def _knowledge_signature(self) -> str:
        repository = self._knowledge.repository
        state = {
            "entities": [
                (item.id, item.name, item.entity_type) for item in repository.entities.all()
            ],
            "relations": [
                (item.id, item.updated_at, float(item.confidence), item.verification_state.value)
                for item in repository.relations.all()
            ],
            "claims": [
                (
                    item.id,
                    item.updated_at,
                    float(item.confidence),
                    item.verification_state.value,
                    tuple(item.supporting_evidence),
                    tuple(item.contradicting_evidence),
                )
                for item in repository.claims.all()
            ],
            "evidence": [(item.id, item.quality, item.role) for item in repository.evidence.all()],
        }
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _relation_edge(relation: Relation) -> ExplorerEdge:
        return ExplorerEdge(
            relation.id,
            relation.subject_id,
            relation.object_id,
            relation.predicate,
            relation.predicate,
            float(relation.confidence),
            relation.verification_state.value,
            metadata={
                "semantic_kind": "relation",
                "predicate": relation.predicate,
                "source_ids": list(relation.source_ids),
                "provenance": list(relation.provenance),
                **dict(relation.metadata),
            },
        )

    @staticmethod
    def _add_gap_edge(edges: dict[str, ExplorerEdge], gap_id: str, target_id: str) -> None:
        edge_id = f"gap-target:{gap_id}:{target_id}"
        edges[edge_id] = ExplorerEdge(
            edge_id,
            gap_id,
            target_id,
            "knowledge_gap",
            "affects",
            directed=False,
        )

    @staticmethod
    def _entity_name_index(entities: Sequence[Entity]) -> dict[str, str]:
        result: dict[str, str] = {}
        for entity in entities:
            for name in (entity.name, entity.canonical, *entity.aliases):
                if name.strip():
                    result.setdefault(KnowledgeExplorer._normalize(name), entity.id)
        return result

    @staticmethod
    def _node_rank(node: ExplorerNode) -> float:
        kind_weight = {"gap": 5.0, "entity": 4.0, "claim": 3.0, "source": 2.0}.get(node.kind, 1.0)
        return (
            kind_weight
            + min(node.degree, 100) / 10.0
            + (node.importance or 0.0) * 3.0
            + (node.confidence or 0.0)
        )

    @staticmethod
    def _snapshot_id(nodes: Sequence[ExplorerNode], edges: Sequence[ExplorerEdge]) -> str:
        state = {
            "nodes": [
                (item.id, item.updated_at, item.status, item.confidence, item.uncertainty)
                for item in nodes
            ],
            "edges": [(item.id, item.confidence, item.verification_state) for item in edges],
        }
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split())

    @staticmethod
    def _require(value: Any | None, node_id: str) -> Any:
        if value is None:
            raise KeyError(node_id)
        return value

    @staticmethod
    def _plain(value: Any) -> dict[str, Any]:
        if not is_dataclass(value):
            raise TypeError(f"cannot serialize {type(value).__name__}")

        def convert(item: Any) -> Any:
            if isinstance(item, Enum):
                return item.value
            if is_dataclass(item):
                return {key: convert(field_value) for key, field_value in asdict(item).items()}
            if isinstance(item, dict):
                return {str(key): convert(field_value) for key, field_value in item.items()}
            if isinstance(item, list | tuple | set | frozenset):
                return [convert(field_value) for field_value in item]
            return item

        result = convert(value)
        if not isinstance(result, dict):
            raise TypeError("domain object did not serialize to an object")
        return result

    @classmethod
    def _gap_dict(cls, gap: KnowledgeGap) -> dict[str, Any]:
        return {**cls._plain(gap), "priority": gap.priority}

    @staticmethod
    def _gap_summary(gap: KnowledgeGap) -> dict[str, Any]:
        return {
            "id": gap.id,
            "kind": gap.kind,
            "description": gap.description,
            "reason": gap.reason,
            "uncertainty": gap.uncertainty,
            "importance": gap.importance,
            "priority": gap.priority,
        }

    @classmethod
    def _evidence_dict(cls, item: Evidence) -> dict[str, Any]:
        return cls._plain(item)

    @classmethod
    def _claim_summary(cls, item: Claim) -> dict[str, Any]:
        return {
            "id": item.id,
            "text": item.text,
            "confidence": float(item.confidence),
            "verification_state": item.verification_state.value,
            "supporting_evidence": list(item.supporting_evidence),
            "contradicting_evidence": list(item.contradicting_evidence),
        }

    @classmethod
    def _contradiction_dict(cls, item: Contradiction) -> dict[str, Any]:
        return cls._plain(item)
