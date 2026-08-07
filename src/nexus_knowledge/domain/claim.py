"""Claims, evidence and provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import Confidence, VerificationState, now_iso
from .document import Span
from .ids import new_id

__all__ = ["Claim", "Evidence", "EvidenceRole", "Provenance"]


class EvidenceRole:
    """Role of an evidence item relative to a claim."""

    SUPPORT = "support"
    CONTRADICT = "contradict"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class Evidence:
    """A verifiable unit of evidence for a claim.

    ``span`` points back into the source document text so the evidence
    can be re-verified programmatically, not only via an LLM-generated
    explanation.
    """

    claim_id: str
    chunk_id: str
    document_id: str
    text: str
    role: str = EvidenceRole.SUPPORT
    span: Span | None = None
    quality: float = 0.5
    id: str = field(default_factory=lambda: new_id("ev"))
    extracted_at: str = field(default_factory=now_iso)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Evidence) and other.id == self.id


@dataclass(slots=True)
class Claim:
    """A statement asserted by the system.

    Claims are first-class citizens: they carry confidence, provenance,
    timestamps, source references, and both supporting and contradicting
    evidence. A claim with no provenance must not become trusted
    knowledge.
    """

    text: str
    subject: str = ""
    predicate: str = ""
    object: str = ""
    confidence: Confidence = Confidence(0.5)
    provenance: list[str] = field(default_factory=list)  # chunk ids
    source_ids: list[str] = field(default_factory=list)  # source ids
    supporting_evidence: list[str] = field(default_factory=list)  # evidence ids
    contradicting_evidence: list[str] = field(default_factory=list)
    verification_state: VerificationState = VerificationState.UNVERIFIED
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("claim"))
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    observed_at: str = ""

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Claim) and other.id == self.id


@dataclass(frozen=True, slots=True)
class Provenance:
    """Structured provenance chain for a claim or relation.

    Answers *"why does the system believe this?"* with machine-readable
    references at every level:

        claim/relation -> evidence -> chunk -> document -> source
    """

    entity_id: str = ""  # claim or relation id
    evidence_ids: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
