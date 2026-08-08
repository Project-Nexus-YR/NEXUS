"""Serializable evidence and investigation-result contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from nexus_runtime.models import new_id, utcnow

from .provenance import EvidenceProvenance


def _required(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


class EvidenceRole(StrEnum):
    """How an evidence item relates to its asserted claim."""

    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"


class InvestigationResultState(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ClaimStatement:
    """A structured assertion that can be fused without language guessing."""

    text: str
    subject: str
    predicate: str
    object: str
    claim_id: str = field(default_factory=lambda: new_id("claim"))

    def __post_init__(self) -> None:
        for value, name in (
            (self.text, "text"),
            (self.subject, "subject"),
            (self.predicate, "predicate"),
            (self.object, "object"),
            (self.claim_id, "claim_id"),
        ):
            _required(value, name)

    @property
    def identity(self) -> tuple[str, str, str]:
        """Canonical semantic identity used for deterministic fusion."""
        subject = " ".join(self.subject.casefold().split())
        predicate = " ".join(self.predicate.casefold().split())
        object_value = " ".join(self.object.casefold().split())
        return subject, predicate, object_value

    @property
    def contradiction_key(self) -> tuple[str, str]:
        return self.identity[:2]

    def to_dict(self) -> dict[str, str]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }


@dataclass(frozen=True, slots=True)
class Evidence:
    """A source-grounded unit returned by an investigation agent."""

    investigation_id: str
    source: str
    claim: ClaimStatement
    provenance: EvidenceProvenance
    confidence: float
    excerpt: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    role: EvidenceRole = EvidenceRole.SUPPORTING
    source_quality: float = 0.5
    supporting_entities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_id: str = field(default_factory=lambda: new_id("evidence"))
    timestamp: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for value, name in (
            (self.evidence_id, "evidence_id"),
            (self.investigation_id, "investigation_id"),
            (self.source, "source"),
        ):
            _required(value, name)
        if self.investigation_id != self.provenance.investigation_id:
            raise ValueError("evidence investigation_id does not match provenance")
        if self.source != self.provenance.source_reference:
            raise ValueError("evidence source does not match provenance source_reference")
        if not self.excerpt.strip() and not self.payload:
            raise ValueError("evidence requires an excerpt or payload")
        for numeric_value, name in (
            (self.confidence, "confidence"),
            (self.source_quality, "source_quality"),
        ):
            if not 0.0 <= numeric_value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")

    @property
    def fingerprint(self) -> str:
        """Stable duplicate key preserving independent-source corroboration."""
        content = {
            "claim": self.claim.identity,
            "source_id": self.provenance.source_id,
            "excerpt": " ".join(self.excerpt.casefold().split()),
            "payload": self.payload,
            "role": self.role.value,
        }
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "investigation_id": self.investigation_id,
            "source": self.source,
            "claim": self.claim.to_dict(),
            "excerpt": self.excerpt,
            "payload": self.payload,
            "provenance": self.provenance.to_dict(),
            "confidence": self.confidence,
            "source_quality": self.source_quality,
            "role": self.role.value,
            "timestamp": self.timestamp.isoformat(),
            "supporting_entities": list(self.supporting_entities),
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class EvidenceSet:
    """Evidence collected for one session, including partial task results."""

    session_id: str
    evidence: tuple[Evidence, ...] = ()
    evidence_set_id: str = field(default_factory=lambda: new_id("evidence_set"))
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        _required(self.session_id, "session_id")
        _required(self.evidence_set_id, "evidence_set_id")
        mismatches = [
            item.evidence_id
            for item in self.evidence
            if item.provenance.session_id != self.session_id
        ]
        if mismatches:
            raise ValueError(f"evidence belongs to another session: {', '.join(mismatches)}")
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_id values must be unique within an EvidenceSet")

    @property
    def investigation_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.investigation_id for item in self.evidence}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_set_id": self.evidence_set_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class InvestigationResult:
    """Track-B contract produced from one distributed task/agent run."""

    session_id: str
    investigation_id: str
    task_id: str
    attempt_id: str
    run_id: str
    state: InvestigationResultState
    evidence_set: EvidenceSet
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    result_id: str = field(default_factory=lambda: new_id("investigation_result"))
    completed_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for value, name in (
            (self.result_id, "result_id"),
            (self.session_id, "session_id"),
            (self.investigation_id, "investigation_id"),
            (self.task_id, "task_id"),
            (self.attempt_id, "attempt_id"),
            (self.run_id, "run_id"),
        ):
            _required(value, name)
        if self.evidence_set.session_id != self.session_id:
            raise ValueError("result and evidence set session ids do not match")
        for item in self.evidence_set.evidence:
            lineage = item.provenance
            expected = (
                self.investigation_id,
                self.task_id,
                self.attempt_id,
                self.run_id,
            )
            actual = (
                lineage.investigation_id,
                lineage.task_id,
                lineage.attempt_id,
                lineage.run_id,
            )
            if expected != actual:
                raise ValueError(f"evidence {item.evidence_id} has mismatched result lineage")
        if self.state == InvestigationResultState.COMPLETED and self.error is not None:
            raise ValueError("a completed result cannot contain an error")
        if self.state == InvestigationResultState.FAILED and not self.error:
            raise ValueError("a failed result requires an error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "session_id": self.session_id,
            "investigation_id": self.investigation_id,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "state": self.state.value,
            "evidence_set": self.evidence_set.to_dict(),
            "error": self.error,
            "metadata": self.metadata,
            "completed_at": self.completed_at.isoformat(),
        }
