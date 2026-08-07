"""In-memory persistence adapters.

Implement the repository ports without any external dependency.
Deterministic ordering (by id) is used for ``all()``.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from ..domain.claim import Claim, Evidence
from ..domain.document import Chunk, Document
from ..domain.entity import Entity, Relation
from ..domain.hypothesis import Experiment, Hypothesis, Observation, Result
from ..domain.knowledge_gap import Investigation, KnowledgeGap
from ..domain.source import Source
from ..port.repository import (
    ClaimRepository,
    ChunkRepository,
    DocumentRepository,
    EntityRepository,
    EvidenceRepository,
    KnowledgeRepository,
    RelationRepository,
    Repository,
    SourceRepository,
)

T = TypeVar("T")

__all__ = [
    "InMemoryRepository",
    "InMemoryKnowledgeRepository",
    "InMemoryClaimRepository",
    "InMemoryDocumentRepository",
    "InMemoryEntityRepository",
]


class InMemoryRepository(Generic[T]):
    """A simple deterministic in-memory repository."""

    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def save(self, item: T) -> T:
        self._items[getattr(item, "id")] = item
        return item

    def get(self, item_id: str) -> T | None:
        return self._items.get(item_id)

    def delete(self, item_id: str) -> bool:
        return self._items.pop(item_id, None) is not None

    def all(self) -> list[T]:
        return [self._items[k] for k in sorted(self._items)]

    def count(self) -> int:
        return len(self._items)


class InMemorySourceRepository(InMemoryRepository[Source], SourceRepository): ...


class InMemoryDocumentRepository(InMemoryRepository[Document], DocumentRepository):
    def by_source(self, source_id: str) -> list[Document]:
        return [d for d in self.all() if d.source_id == source_id]


class InMemoryChunkRepository(InMemoryRepository[Chunk], ChunkRepository):
    def by_document(self, document_id: str) -> list[Chunk]:
        return sorted(
            (c for c in self.all() if c.document_id == document_id),
            key=lambda c: c.index,
        )


class InMemoryEntityRepository(InMemoryRepository[Entity], EntityRepository):
    def by_name(self, name: str) -> Entity | None:
        lowered = name.strip().lower()
        for entity in self.all():
            candidates = [entity.name, entity.canonical_name] + entity.aliases
            if any(c and c.strip().lower() == lowered for c in candidates):
                return entity
        return None


class InMemoryRelationRepository(InMemoryRepository[Relation], RelationRepository): ...


class InMemoryClaimRepository(InMemoryRepository[Claim], ClaimRepository):
    def by_subject(self, subject: str) -> list[Claim]:
        return [c for c in self.all() if c.subject == subject]


class InMemoryEvidenceRepository(InMemoryRepository[Evidence], EvidenceRepository):
    def by_claim(self, claim_id: str) -> list[Evidence]:
        return [e for e in self.all() if e.claim_id == claim_id]


class InMemoryKnowledgeRepository(KnowledgeRepository):
    """Aggregate in-memory persistence for the whole knowledge engine."""

    def __init__(self) -> None:
        self.sources = InMemorySourceRepository()
        self.documents = InMemoryDocumentRepository()
        self.chunks = InMemoryChunkRepository()
        self.entities = InMemoryEntityRepository()
        self.relations = InMemoryRelationRepository()
        self.claims = InMemoryClaimRepository()
        self.evidence = InMemoryEvidenceRepository()
        self.hypotheses: Repository[Hypothesis] = InMemoryRepository[Hypothesis]()
        self.experiments: Repository[Experiment] = InMemoryRepository[Experiment]()
        self.results: Repository[Result] = InMemoryRepository[Result]()
        self.observations: Repository[Observation] = InMemoryRepository[Observation]()
        self.gaps: Repository[KnowledgeGap] = InMemoryRepository[KnowledgeGap]()
        self.investigations: Repository[Investigation] = InMemoryRepository[Investigation]()

    def healthcheck(self) -> dict[str, int]:
        return {
            "sources": self.sources.count(),
            "documents": self.documents.count(),
            "chunks": self.chunks.count(),
            "entities": self.entities.count(),
            "relations": self.relations.count(),
            "claims": self.claims.count(),
            "evidence": self.evidence.count(),
        }
