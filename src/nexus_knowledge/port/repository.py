"""Repository ports for domain aggregates.

Persistence adapters (in-memory, JSON, later a real DB) implement these.
The application/service layer never touches storage directly; it only
sees these interfaces.
"""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar

from ..domain.claim import Claim, Evidence
from ..domain.document import Chunk, Document
from ..domain.entity import Entity, Relation
from ..domain.hypothesis import Experiment, Hypothesis, Observation, Result
from ..domain.knowledge_gap import Investigation, KnowledgeGap
from ..domain.source import Source

T = TypeVar("T")

__all__ = [
    "Repository",
    "SourceRepository",
    "DocumentRepository",
    "ChunkRepository",
    "EntityRepository",
    "RelationRepository",
    "ClaimRepository",
    "EvidenceRepository",
    "KnowledgeRepository",
]


class Repository(Protocol, Generic[T]):
    def save(self, item: T) -> T: ...

    def get(self, item_id: str) -> T | None: ...

    def delete(self, item_id: str) -> bool: ...

    def all(self) -> list[T]: ...

    def count(self) -> int: ...


class SourceRepository(Repository[Source], Protocol): ...


class DocumentRepository(Repository[Document], Protocol):
    def by_source(self, source_id: str) -> list[Document]: ...


class ChunkRepository(Repository[Chunk], Protocol):
    def by_document(self, document_id: str) -> list[Chunk]: ...


class EntityRepository(Repository[Entity], Protocol):
    def by_name(self, name: str) -> Entity | None: ...


class RelationRepository(Repository[Relation], Protocol): ...


class ClaimRepository(Repository[Claim], Protocol):
    def by_subject(self, subject: str) -> list[Claim]: ...


class EvidenceRepository(Repository[Evidence], Protocol):
    def by_claim(self, claim_id: str) -> list[Evidence]: ...


class KnowledgeRepository(Protocol):
    """Aggregate view over all domain repositories.

    Exposes the full persistence surface needed by the knowledge engine
    without leaking the concrete storage implementation.
    """

    sources: SourceRepository
    documents: DocumentRepository
    chunks: ChunkRepository
    entities: EntityRepository
    relations: RelationRepository
    claims: ClaimRepository
    evidence: EvidenceRepository
    hypotheses: Repository[Hypothesis]
    experiments: Repository[Experiment]
    results: Repository[Result]
    observations: Repository[Observation]
    gaps: Repository[KnowledgeGap]
    investigations: Repository[Investigation]

    def healthcheck(self) -> dict[str, int]: ...
