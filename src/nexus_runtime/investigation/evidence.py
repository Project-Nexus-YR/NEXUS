"""Serializable evidence and investigation-result contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from nexus_runtime.models import new_id, utcnow

from .provenance import EvidenceProvenance


def _required(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _persisted_string(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _persisted_float(payload: Mapping[str, Any], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _persisted_bool(payload: Mapping[str, Any], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _persisted_strings(payload: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = payload.get(name)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return tuple(value)


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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClaimStatement:
        try:
            return cls(
                claim_id=_persisted_string(payload, "claim_id"),
                text=_persisted_string(payload, "text"),
                subject=_persisted_string(payload, "subject"),
                predicate=_persisted_string(payload, "predicate"),
                object=_persisted_string(payload, "object"),
            )
        except KeyError as exc:
            raise ValueError("malformed claim statement") from exc


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

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Evidence:
        claim = payload.get("claim")
        provenance = payload.get("provenance")
        item_payload = payload.get("payload", {})
        metadata = payload.get("metadata", {})
        entities = payload.get("supporting_entities", [])
        if (
            not isinstance(claim, dict)
            or not isinstance(provenance, dict)
            or not isinstance(item_payload, dict)
            or not isinstance(metadata, dict)
            or not isinstance(entities, list)
        ):
            raise ValueError("malformed evidence")
        excerpt = payload.get("excerpt", "")
        error_role = payload.get("role")
        if not isinstance(excerpt, str) or not isinstance(error_role, str):
            raise ValueError("malformed evidence")
        try:
            return cls(
                evidence_id=_persisted_string(payload, "evidence_id"),
                investigation_id=_persisted_string(payload, "investigation_id"),
                source=_persisted_string(payload, "source"),
                claim=ClaimStatement.from_dict(claim),
                excerpt=excerpt,
                payload=dict(item_payload),
                provenance=EvidenceProvenance.from_dict(provenance),
                confidence=_persisted_float(payload, "confidence"),
                source_quality=_persisted_float(payload, "source_quality"),
                role=EvidenceRole(error_role),
                timestamp=_parse_timestamp(payload["timestamp"]),
                supporting_entities=_persisted_strings(payload, "supporting_entities"),
                metadata=dict(metadata),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed evidence") from exc


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
        identities_by_claim_id: dict[str, tuple[str, str, str]] = {}
        for item in self.evidence:
            previous = identities_by_claim_id.setdefault(item.claim.claim_id, item.claim.identity)
            if previous != item.claim.identity:
                raise ValueError("a claim_id cannot represent multiple claim identities")

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

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EvidenceSet:
        evidence = payload.get("evidence")
        if not isinstance(evidence, list) or any(not isinstance(item, dict) for item in evidence):
            raise ValueError("malformed evidence set")
        try:
            return cls(
                evidence_set_id=_persisted_string(payload, "evidence_set_id"),
                session_id=_persisted_string(payload, "session_id"),
                created_at=_parse_timestamp(payload["created_at"]),
                evidence=tuple(Evidence.from_dict(item) for item in evidence),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed evidence set") from exc


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

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> InvestigationResult:
        evidence_set = payload.get("evidence_set")
        metadata = payload.get("metadata", {})
        if not isinstance(evidence_set, dict) or not isinstance(metadata, dict):
            raise ValueError("malformed investigation result")
        error = payload.get("error")
        state = payload.get("state")
        if (error is not None and not isinstance(error, str)) or not isinstance(state, str):
            raise ValueError("malformed investigation result")
        try:
            return cls(
                result_id=_persisted_string(payload, "result_id"),
                session_id=_persisted_string(payload, "session_id"),
                investigation_id=_persisted_string(payload, "investigation_id"),
                task_id=_persisted_string(payload, "task_id"),
                attempt_id=_persisted_string(payload, "attempt_id"),
                run_id=_persisted_string(payload, "run_id"),
                state=InvestigationResultState(state),
                evidence_set=EvidenceSet.from_dict(evidence_set),
                error=error,
                metadata=dict(metadata),
                completed_at=_parse_timestamp(payload["completed_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed investigation result") from exc


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be an ISO-8601 string") from exc
