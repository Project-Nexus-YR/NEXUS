"""Knowledge-graph abstraction.

The :class:`KnowledgeGraph` interface is the single contract the
application layer depends on for graph storage. It does **not** assume
a single-node deployment: operations are expressed over entity and
relation IDs so that a later partitioned/distributed backend can
implement the same interface.

The local backend lives in :mod:`nexus_knowledge.graph.memory`.
Algorithm implementations (PageRank, PPR, centrality, communities,
paths) live in :mod:`nexus_knowledge.graph.algorithms` and operate on a
plain adjacency structure, so they can be reused by any backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..domain.entity import Entity, Relation

__all__ = [
    "Edge",
    "GraphStats",
    "KnowledgeGraph",
    "KnowledgeSubgraph",
    "Path",
]


@dataclass(frozen=True, slots=True)
class Edge:
    """A directed graph edge backed by a Relation."""

    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    weight: float = 1.0

    @classmethod
    def from_relation(cls, relation: Relation) -> Edge:
        return cls(
            relation_id=relation.id,
            subject_id=relation.subject_id,
            predicate=relation.predicate,
            object_id=relation.object_id,
            weight=float(relation.confidence),
        )


@dataclass(frozen=True, slots=True)
class Path:
    """An entity-id path: ``[subject, ..., object]``."""

    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]
    weight: float

    @property
    def score(self) -> float:
        return self.weight


@dataclass(frozen=True, slots=True)
class KnowledgeSubgraph:
    """A bounded extract of the graph around a set of nodes."""

    nodes: dict[str, Entity]
    edges: tuple[Edge, ...]

    def entity_ids(self) -> list[str]:
        return list(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)


@dataclass(slots=True)
class GraphStats:
    num_entities: int = 0
    num_relations: int = 0
    num_edges: int = 0
    density: float = 0.0
    num_components: int = 0
    average_degree: float = 0.0
    num_isolated: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "num_edges": self.num_edges,
            "density": self.density,
            "num_components": self.num_components,
            "average_degree": self.average_degree,
            "num_isolated": self.num_isolated,
        }


class KnowledgeGraph(ABC):
    """Operations required of any knowledge-graph storage backend."""

    # -- entities -----------------------------------------------------
    @abstractmethod
    def add_entity(self, entity: Entity) -> Entity: ...

    @abstractmethod
    def update_entity(self, entity: Entity) -> Entity: ...

    @abstractmethod
    def remove_entity(self, entity_id: str) -> bool: ...

    @abstractmethod
    def get_entity(self, entity_id: str) -> Entity | None: ...

    @abstractmethod
    def all_entities(self) -> list[Entity]: ...

    # -- relations ----------------------------------------------------
    @abstractmethod
    def add_relation(self, relation: Relation) -> Relation: ...

    @abstractmethod
    def update_relation(self, relation: Relation) -> Relation: ...

    @abstractmethod
    def remove_relation(self, relation_id: str) -> bool: ...

    @abstractmethod
    def get_relation(self, relation_id: str) -> Relation | None: ...

    @abstractmethod
    def all_relations(self) -> list[Relation]: ...

    # -- traversal ----------------------------------------------------
    @abstractmethod
    def neighbors(self, entity_id: str, direction: str = "out") -> list[Edge]:
        """Outgoing (default), incoming or bidirectional edges."""

    @abstractmethod
    def traversal(
        self,
        seed_ids: list[str],
        depth: int = 2,
        max_nodes: int = 100,
        max_edges: int = 500,
        direction: str = "both",
    ) -> tuple[set[str], list[Edge]]:
        """Bounded expansion around seed entities.

        Returns ``(visited_node_ids, edges_touched)``. ``direction`` is
        ``"out"``, ``"in"`` or ``"both"``.
        """

    @abstractmethod
    def paths(
        self,
        subject_id: str,
        object_id: str,
        max_length: int = 4,
        max_paths: int = 20,
    ) -> list[Path]:
        """Enumerate simple paths between two entities, best-weighted first."""

    @abstractmethod
    def subgraph(
        self,
        entity_ids: list[str],
        depth: int = 1,
        max_nodes: int = 100,
        max_edges: int = 500,
        direction: str = "both",
    ) -> KnowledgeSubgraph:
        """Extract a bounded subgraph around the given entities."""

    # -- analytics ----------------------------------------------------
    @abstractmethod
    def pagerank(
        self, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6
    ) -> dict[str, float]: ...

    @abstractmethod
    def personalized_pagerank(
        self,
        seed_ids: list[str],
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict[str, float]: ...

    @abstractmethod
    def degree_centrality(self) -> dict[str, float]: ...

    @abstractmethod
    def betweenness_centrality(self) -> dict[str, float]: ...

    @abstractmethod
    def communities(self) -> dict[str, int]:
        """Map each entity id to a community id."""

    @abstractmethod
    def statistics(self) -> GraphStats: ...

    # -- iteration ----------------------------------------------------
    @abstractmethod
    def __len__(self) -> int: ...
