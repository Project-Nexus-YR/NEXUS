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


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()[
        :24
    ]
    return f"{prefix}_{digest}"


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


class EvidenceGrade(StrEnum):
    """Discrete quality bucket derived from the evidentiary-strength composite."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


GRADE_THRESHOLDS: tuple[tuple[EvidenceGrade, float], ...] = (
    (EvidenceGrade.STRONG, 0.7),
    (EvidenceGrade.MODERATE, 0.4),
    (EvidenceGrade.WEAK, 0.0),
)


def grade_for_strength(strength: float) -> EvidenceGrade:
    """Map an evidentiary-strength score to its discrete grade."""
    for grade, threshold in GRADE_THRESHOLDS:
        if strength >= threshold:
            return grade
    return EvidenceGrade.WEAK


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
    claim_id: str = ""

    def __post_init__(self) -> None:
        for value, name in (
            (self.text, "text"),
            (self.subject, "subject"),
            (self.predicate, "predicate"),
            (self.object, "object"),
        ):
            _required(value, name)
        claim_id = self.claim_id.strip()
        if not claim_id:
            claim_id = _stable_id("claim", *self.identity)
        _required(claim_id, "claim_id")
        object.__setattr__(self, "claim_id", claim_id)

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
class ToolObservation:
    """A serializable record of one tool execution during an agent run.

    The ``observation_id`` is the runtime ``tool_call_id`` so that
    conclusions can reference observations and the lineage chain
    ``investigation -> run -> conclusion -> observation -> evidence -> source``
    remains reconstructable after a restart.
    """

    observation_id: str
    tool_name: str
    status: str
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    source_reference: str = ""
    timestamp: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in (
            (self.observation_id, "observation_id"),
            (self.tool_name, "tool_name"),
            (self.status, "status"),
        ):
            _required(value, name)
        if not isinstance(self.input, dict):
            raise ValueError("observation input must be an object")
        if self.output is not None and not isinstance(self.output, dict):
            raise ValueError("observation output must be an object or null")
        if not isinstance(self.metadata, dict):
            raise ValueError("observation metadata must be an object")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "input": dict(self.input),
            "output": None if self.output is None else dict(self.output),
            "source_reference": self.source_reference,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ToolObservation:
        observation_input = payload.get("input")
        output = payload.get("output")
        metadata = payload.get("metadata", {})
        source_reference = payload.get("source_reference", "")
        status = payload.get("status")
        tool_name = payload.get("tool_name")
        if (
            not isinstance(observation_input, dict)
            or not isinstance(output, dict | None)
            or not isinstance(metadata, dict)
            or not isinstance(source_reference, str)
            or not isinstance(status, str)
            or not isinstance(tool_name, str)
        ):
            raise ValueError("malformed tool observation")
        try:
            return cls(
                observation_id=_persisted_string(payload, "observation_id"),
                tool_name=tool_name,
                status=status,
                input=dict(observation_input),
                output=None if output is None else dict(output),
                source_reference=source_reference,
                timestamp=_parse_timestamp(payload["timestamp"]),
                metadata=dict(metadata),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed tool observation") from exc


@dataclass(frozen=True, slots=True)
class AgentConclusion:
    """A candidate assertion proposed by an agent, not yet trusted knowledge.

    A conclusion references observations by their ``observation_id``.  It
    becomes a candidate claim only after deterministic extraction, and it
    never enters the knowledge graph without passing verification.
    """

    claim: ClaimStatement
    supporting_observation_ids: tuple[str, ...]
    confidence: float
    conclusion_id: str = ""
    created_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError("conclusion confidence must be numeric")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("conclusion confidence must be between zero and one")
        if not isinstance(self.supporting_observation_ids, (tuple, list)):
            raise ValueError("supporting_observation_ids must be a sequence of strings")
        observation_ids = tuple(sorted({str(item) for item in self.supporting_observation_ids}))
        for observation_id in observation_ids:
            _required(observation_id, "supporting_observation_id")
        if not isinstance(self.metadata, dict):
            raise ValueError("conclusion metadata must be an object")
        object.__setattr__(self, "supporting_observation_ids", observation_ids)
        object.__setattr__(self, "metadata", dict(self.metadata))
        conclusion_id = self.conclusion_id.strip()
        if not conclusion_id:
            conclusion_id = _stable_id(
                "conclusion",
                self.claim.claim_id,
                *observation_ids,
            )
        _required(conclusion_id, "conclusion_id")
        object.__setattr__(self, "conclusion_id", conclusion_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conclusion_id": self.conclusion_id,
            "claim": self.claim.to_dict(),
            "supporting_observation_ids": list(self.supporting_observation_ids),
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AgentConclusion:
        claim = payload.get("claim")
        observation_ids = payload.get("supporting_observation_ids")
        metadata = payload.get("metadata", {})
        if (
            not isinstance(claim, dict)
            or not isinstance(observation_ids, list)
            or any(not isinstance(item, str) for item in observation_ids)
            or not isinstance(metadata, dict)
        ):
            raise ValueError("malformed agent conclusion")
        try:
            return cls(
                conclusion_id=_persisted_string(payload, "conclusion_id"),
                claim=ClaimStatement.from_dict(claim),
                supporting_observation_ids=tuple(observation_ids),
                confidence=_persisted_float(payload, "confidence"),
                created_at=_parse_timestamp(payload["created_at"]),
                metadata=dict(metadata),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed agent conclusion") from exc


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

    @property
    def evidentiary_strength(self) -> float:
        """Conservative composite of agent confidence and source quality.

        The geometric mean requires both signals to be strong before the
        evidence is graded STRONG, preventing a single inflated signal from
        carrying the item.
        """
        return float((self.confidence * self.source_quality) ** 0.5)

    @property
    def grade(self) -> EvidenceGrade:
        """Discrete quality bucket derived from :attr:`evidentiary_strength`."""
        return grade_for_strength(self.evidentiary_strength)

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
            "evidentiary_strength": self.evidentiary_strength,
            "grade": self.grade.value,
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

    @property
    def grade_counts(self) -> dict[str, int]:
        """Count of evidence items per grade, keyed by grade value."""
        counts = {grade.value: 0 for grade in EvidenceGrade}
        for item in self.evidence:
            counts[item.grade.value] += 1
        return counts

    @property
    def mean_evidentiary_strength(self) -> float:
        """Mean composite strength across the set (0.0 when empty)."""
        if not self.evidence:
            return 0.0
        return sum(item.evidentiary_strength for item in self.evidence) / len(self.evidence)

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
    final_answer: str = ""
    conclusions: tuple[AgentConclusion, ...] = ()
    observations: tuple[ToolObservation, ...] = ()
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
        if not isinstance(self.final_answer, str):
            raise ValueError("final_answer must be a string")
        if not isinstance(self.metadata, dict):
            raise ValueError("result metadata must be an object")
        observation_ids = [item.observation_id for item in self.observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation_id values must be unique within a result")
        conclusion_ids = [item.conclusion_id for item in self.conclusions]
        if len(conclusion_ids) != len(set(conclusion_ids)):
            raise ValueError("conclusion_id values must be unique within a result")
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
            "final_answer": self.final_answer,
            "conclusions": [item.to_dict() for item in self.conclusions],
            "observations": [item.to_dict() for item in self.observations],
            "metadata": self.metadata,
            "completed_at": self.completed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> InvestigationResult:
        evidence_set = payload.get("evidence_set")
        metadata = payload.get("metadata", {})
        conclusions = payload.get("conclusions", [])
        observations = payload.get("observations", [])
        final_answer = payload.get("final_answer", "")
        if (
            not isinstance(evidence_set, dict)
            or not isinstance(metadata, dict)
            or not isinstance(conclusions, list)
            or any(not isinstance(item, dict) for item in conclusions)
            or not isinstance(observations, list)
            or any(not isinstance(item, dict) for item in observations)
            or not isinstance(final_answer, str)
        ):
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
                final_answer=final_answer,
                conclusions=tuple(AgentConclusion.from_dict(item) for item in conclusions),
                observations=tuple(ToolObservation.from_dict(item) for item in observations),
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
