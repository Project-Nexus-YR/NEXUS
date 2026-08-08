"""In-memory knowledge-graph backend.

Deterministic, dependency-free implementation of :class:`KnowledgeGraph`.
Serves as the local backend and as the reference behaviour for future
distributed backends.
"""

from __future__ import annotations

from dataclasses import replace

from ..domain.common import VerificationState
from ..domain.entity import Entity, Relation
from . import algorithms
from .graph import Edge, GraphStats, KnowledgeGraph, KnowledgeSubgraph, Path

__all__ = ["InMemoryGraph"]


class InMemoryGraph(KnowledgeGraph):
    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relations: dict[str, Relation] = {}
        self._tuple_key: dict[str, str] = {}  # (subject, predicate, object) -> relation id

    # -- entities -----------------------------------------------------
    def add_entity(self, entity: Entity) -> Entity:
        existing = self._entities.get(entity.id)
        if existing:
            return existing
        self._entities[entity.id] = entity
        return entity

    def update_entity(self, entity: Entity) -> Entity:
        self._entities[entity.id] = entity
        return entity

    def remove_entity(self, entity_id: str) -> bool:
        if entity_id not in self._entities:
            return False
        for relation in list(self._relations.values()):
            if relation.subject_id == entity_id or relation.object_id == entity_id:
                self.remove_relation(relation.id)
        del self._entities[entity_id]
        return True

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def all_entities(self) -> list[Entity]:
        return list(self._entities.values())

    # -- relations ----------------------------------------------------
    def add_relation(self, relation: Relation) -> Relation:
        key = relation.tuple
        existing_id = self._tuple_key.get(key)
        if existing_id is not None:
            existing = self._relations[existing_id]
            merged = replace(
                existing,
                confidence=relation.confidence,
                provenance=sorted(set(existing.provenance) | set(relation.provenance)),
                source_ids=sorted(set(existing.source_ids) | set(relation.source_ids)),
                supporting_evidence=sorted(
                    set(existing.supporting_evidence) | set(relation.supporting_evidence)
                ),
                contradicting_evidence=sorted(
                    set(existing.contradicting_evidence) | set(relation.contradicting_evidence)
                ),
                verification_state=VerificationState.UNVERIFIED,
            )
            self._relations[existing.id] = merged
            return merged
        self._relations[relation.id] = relation
        self._tuple_key[key] = relation.id
        return relation

    def update_relation(self, relation: Relation) -> Relation:
        self._relations[relation.id] = relation
        self._tuple_key[relation.tuple] = relation.id
        return relation

    def remove_relation(self, relation_id: str) -> bool:
        relation = self._relations.pop(relation_id, None)
        if relation is None:
            return False
        self._tuple_key.pop(relation.tuple, None)
        return True

    def get_relation(self, relation_id: str) -> Relation | None:
        return self._relations.get(relation_id)

    def all_relations(self) -> list[Relation]:
        return list(self._relations.values())

    def _out_edges(self, entity_id: str) -> list[Relation]:
        return [r for r in self._relations.values() if r.subject_id == entity_id]

    def _in_edges(self, entity_id: str) -> list[Relation]:
        return [r for r in self._relations.values() if r.object_id == entity_id]

    # -- traversal ----------------------------------------------------
    def neighbors(self, entity_id: str, direction: str = "out") -> list[Edge]:
        if direction == "out":
            relations = self._out_edges(entity_id)
        elif direction == "in":
            relations = self._in_edges(entity_id)
        elif direction == "both":
            relations = self._out_edges(entity_id) + self._in_edges(entity_id)
        else:
            raise ValueError(f"direction must be 'out'|'in'|'both', got {direction!r}")
        return [Edge.from_relation(r) for r in relations]

    def traversal(
        self,
        seed_ids: list[str],
        depth: int = 2,
        max_nodes: int = 100,
        max_edges: int = 500,
        direction: str = "both",
    ) -> tuple[set[str], list[Edge]]:
        visited: set[str] = set(seed_ids)
        edges: list[Edge] = []
        frontier = list(seed_ids)
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                for edge in self.neighbors(node, direction):
                    if len(edges) >= max_edges:
                        return visited, edges
                    edges.append(edge)
                    other = edge.object_id if edge.subject_id == node else edge.subject_id
                    if other not in visited and len(visited) < max_nodes:
                        visited.add(other)
                        next_frontier.add(other)
            frontier = list(next_frontier)
            if not frontier:
                break
        return visited, edges

    def paths(
        self,
        subject_id: str,
        object_id: str,
        max_length: int = 4,
        max_paths: int = 20,
    ) -> list[Path]:
        adjacency = self._adjacency()
        node_lists = algorithms.enumerate_paths(
            adjacency, subject_id, object_id, max_length=max_length, max_paths=max_paths
        )
        results: list[Path] = []
        for nodes in node_lists:
            edges: list[Edge] = []
            weight = 1.0
            for i in range(len(nodes) - 1):
                edges_on_hop = [e for e in self._out_edges(nodes[i]) if e.object_id == nodes[i + 1]]
                if not edges_on_hop:
                    break
                edge = Edge.from_relation(edges_on_hop[0])
                edges.append(edge)
                weight *= edge.weight
            else:
                results.append(Path(nodes=tuple(nodes), edges=tuple(edges), weight=weight))
        results.sort(key=lambda p: p.weight, reverse=True)
        return results

    def subgraph(
        self,
        entity_ids: list[str],
        depth: int = 1,
        max_nodes: int = 100,
        max_edges: int = 500,
        direction: str = "both",
    ) -> KnowledgeSubgraph:
        visited, edges = self.traversal(
            entity_ids, depth=depth, max_nodes=max_nodes, max_edges=max_edges, direction=direction
        )
        nodes = {eid: self._entities[eid] for eid in visited if eid in self._entities}
        return KnowledgeSubgraph(nodes=nodes, edges=tuple(edges))

    # -- analytics ----------------------------------------------------
    def _adjacency(self) -> algorithms.Adjacency:
        adj: algorithms.Adjacency = {}
        for entity_id in self._entities:
            adj.setdefault(entity_id, [])
        for relation in self._relations.values():
            adj.setdefault(relation.subject_id, [])
            adj[relation.subject_id].append((relation.object_id, float(relation.confidence)))
        return adj

    def pagerank(
        self, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6
    ) -> dict[str, float]:
        return algorithms.pagerank(self._adjacency(), damping=damping, max_iter=max_iter, tol=tol)

    def personalized_pagerank(
        self,
        seed_ids: list[str],
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict[str, float]:
        return algorithms.personalized_pagerank(
            self._adjacency(), seed_ids, damping=damping, max_iter=max_iter, tol=tol
        )

    def degree_centrality(self) -> dict[str, float]:
        return algorithms.degree_centrality(self._adjacency())

    def betweenness_centrality(self) -> dict[str, float]:
        return algorithms.betweenness_centrality(self._adjacency())

    def communities(self) -> dict[str, int]:
        return algorithms.label_propagation_communities(self._adjacency())

    def statistics(self) -> GraphStats:
        adjacency = self._adjacency()
        components = algorithms.connected_components(adjacency)
        return GraphStats(
            num_entities=len(self._entities),
            num_relations=len(self._relations),
            num_edges=len(self._tuple_key),
            density=algorithms.density(adjacency),
            num_components=len(components),
            average_degree=algorithms.average_degree(adjacency),
            num_isolated=sum(1 for c in components if len(c) == 1),
        )

    def __len__(self) -> int:
        return len(self._entities)
