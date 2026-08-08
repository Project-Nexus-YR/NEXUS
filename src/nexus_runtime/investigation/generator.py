"""Deterministic transformation of measurable knowledge gaps into investigation work."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from importlib import import_module
from typing import Any, Protocol, cast

from nexus_runtime.models import DomainError

from .objective import (
    ResearchObjective,
    _string_tuple,
    _timestamp_from_text,
    _timestamp_to_text,
    _validate_json,
    _validate_timestamp,
)


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DomainError("numeric candidate field is malformed")
    return float(value)


class ExistingInvestigation(Protocol):
    id: str
    gap_id: str
    description: str
    target_entities: list[str]
    expected_information_gain: float
    uncertainty_reduction: float
    importance: float
    estimated_cost: float
    score: float
    metadata: dict[str, Any]
    created_at: str


class KnowledgeGapLike(Protocol):
    id: str
    kind: str
    description: str
    reason: str
    affected_entities: list[str]
    affected_relations: list[str]
    affected_claims: list[str]
    uncertainty: float
    importance: float
    estimated_cost: float
    candidate_investigations: list[ExistingInvestigation]
    metadata: dict[str, Any]
    created_at: str

    @property
    def priority(self) -> float: ...


class GainEstimatorLike(Protocol):
    def expected_information_gain(self, gap: KnowledgeGapLike) -> float: ...

    def uncertainty_reduction(self, gap: KnowledgeGapLike) -> float: ...


def _stable_id(prefix: str, *parts: object) -> str:
    """Call the knowledge subsystem's deterministic identifier boundary lazily."""
    identifier = cast("Any", import_module("nexus_knowledge.domain.ids").stable_id)
    return str(identifier(prefix, *parts))


MISSING_RELATION = "missing_relation"
LOW_CONFIDENCE = "low_confidence"
UNSUPPORTED_CLAIM = "unsupported_claim"
CONTRADICTION = "contradiction"
STALE = "stale"
DISCONNECTED_ENTITY = "disconnected_entity"
MISSING_EVIDENCE = "missing_evidence"
ANOMALOUS_REGION = "anomalous_region"
LOW_DIVERSITY = "low_diversity"


_CAPABILITIES: dict[str, tuple[str, ...]] = {
    MISSING_RELATION: ("graph_reasoning", "search"),
    LOW_CONFIDENCE: ("search", "verification"),
    UNSUPPORTED_CLAIM: ("search", "verification"),
    CONTRADICTION: ("criticism", "verification"),
    STALE: ("search", "verification"),
    DISCONNECTED_ENTITY: ("graph_reasoning", "search"),
    MISSING_EVIDENCE: ("document_analysis", "search"),
    ANOMALOUS_REGION: ("criticism", "graph_reasoning"),
    LOW_DIVERSITY: ("search", "verification"),
}

_REQUIRED_EVIDENCE: dict[str, tuple[str, ...]] = {
    MISSING_RELATION: ("direct relational evidence", "independent source"),
    LOW_CONFIDENCE: ("independent supporting or refuting evidence",),
    UNSUPPORTED_CLAIM: ("source-backed claim evidence",),
    CONTRADICTION: ("independent evidence for each conflicting claim",),
    STALE: ("current dated source",),
    DISCONNECTED_ENTITY: ("entity relationship evidence",),
    MISSING_EVIDENCE: ("traceable primary or secondary source",),
    ANOMALOUS_REGION: ("independent explanation of the anomaly",),
    LOW_DIVERSITY: ("evidence from a distinct source",),
}


@dataclass(frozen=True, slots=True)
class CandidateInvestigation:
    gap_id: str
    question: str
    hypothesis: str
    required_evidence: tuple[str, ...]
    constraints: tuple[str, ...]
    expected_information_gain: float
    uncertainty_reduction: float
    estimated_cost: float
    estimated_duration_seconds: float
    risk: float
    priority: float
    capabilities: tuple[str, ...]
    target_entities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    investigation_id: str = ""
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        gap_id = self.gap_id.strip()
        question = self.question.strip()
        hypothesis = self.hypothesis.strip()
        if not gap_id or not question or not hypothesis:
            raise DomainError("gap_id, question, and hypothesis are required")
        required_evidence = _string_tuple(
            self.required_evidence, "required_evidence", allow_empty=False
        )
        constraints = _string_tuple(self.constraints, "constraints")
        capabilities = tuple(sorted(set(_string_tuple(self.capabilities, "capabilities"))))
        if not capabilities:
            raise DomainError("candidate investigation requires at least one capability")
        target_entities = tuple(sorted(set(_string_tuple(self.target_entities, "target_entities"))))
        for name, value in (
            ("expected_information_gain", self.expected_information_gain),
            ("uncertainty_reduction", self.uncertainty_reduction),
            ("risk", self.risk),
            ("priority", self.priority),
        ):
            if not 0.0 <= value <= 1.0:
                raise DomainError(f"{name} must be between zero and one")
        if self.estimated_cost < 0:
            raise DomainError("estimated_cost cannot be negative")
        if self.estimated_duration_seconds <= 0:
            raise DomainError("estimated_duration_seconds must be positive")
        _validate_json(self.metadata, "candidate metadata")
        created_at = self.created_at
        if created_at is None:
            raise DomainError("candidate created_at is required for reproducibility")
        _validate_timestamp(created_at, "created_at")
        generated_id = _stable_id(
            "investigation",
            gap_id,
            question.casefold(),
            hypothesis.casefold(),
            required_evidence,
            constraints,
        )
        investigation_id = self.investigation_id.strip() or generated_id
        object.__setattr__(self, "gap_id", gap_id)
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "hypothesis", hypothesis)
        object.__setattr__(self, "required_evidence", required_evidence)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "target_entities", target_entities)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "investigation_id", investigation_id)

    @property
    def redundancy_key(self) -> str:
        explicit = self.metadata.get("redundancy_key")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        return _stable_id(
            "evidence-need",
            self.gap_id,
            self.required_evidence,
            self.target_entities,
            self.metadata.get("evidence_channel", ""),
        )

    def to_dict(self) -> dict[str, object]:
        assert self.created_at is not None
        return {
            "investigation_id": self.investigation_id,
            "gap_id": self.gap_id,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "required_evidence": list(self.required_evidence),
            "constraints": list(self.constraints),
            "expected_information_gain": self.expected_information_gain,
            "uncertainty_reduction": self.uncertainty_reduction,
            "estimated_cost": self.estimated_cost,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "risk": self.risk,
            "priority": self.priority,
            "capabilities": list(self.capabilities),
            "target_entities": list(self.target_entities),
            "created_at": _timestamp_to_text(self.created_at),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CandidateInvestigation:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise DomainError("candidate metadata must be an object")
        try:
            return cls(
                investigation_id=str(payload["investigation_id"]),
                gap_id=str(payload["gap_id"]),
                question=str(payload["question"]),
                hypothesis=str(payload["hypothesis"]),
                required_evidence=_string_tuple(
                    payload["required_evidence"], "required_evidence", allow_empty=False
                ),
                constraints=_string_tuple(payload["constraints"], "constraints"),
                expected_information_gain=_as_float(payload["expected_information_gain"]),
                uncertainty_reduction=_as_float(payload["uncertainty_reduction"]),
                estimated_cost=_as_float(payload["estimated_cost"]),
                estimated_duration_seconds=_as_float(payload["estimated_duration_seconds"]),
                risk=_as_float(payload["risk"]),
                priority=_as_float(payload["priority"]),
                capabilities=_string_tuple(payload["capabilities"], "capabilities"),
                target_entities=_string_tuple(payload["target_entities"], "target_entities"),
                created_at=_timestamp_from_text(payload["created_at"], "created_at"),
                metadata=dict(metadata),
            )
        except KeyError as exc:
            raise DomainError("malformed CandidateInvestigation") from exc


class InvestigationGenerator:
    """Enrich existing gap investigations; never invents ungrounded work."""

    def __init__(self, gain_estimator: GainEstimatorLike | None = None) -> None:
        if gain_estimator is None:
            scorer_module = import_module("nexus_knowledge.knowledge.scorer")
            gain_type = cast("Any", scorer_module.BaselineGainEstimator)
            gain_estimator = cast(GainEstimatorLike, gain_type())
        self._gain = gain_estimator

    def generate(
        self,
        objective: ResearchObjective,
        gaps: Sequence[KnowledgeGapLike],
    ) -> tuple[CandidateInvestigation, ...]:
        generated: dict[str, CandidateInvestigation] = {}
        for gap in sorted(gaps, key=lambda item: item.id):
            source_candidates: tuple[ExistingInvestigation | None, ...] = tuple(
                gap.candidate_investigations
            ) or (None,)
            for source in sorted(
                source_candidates,
                key=lambda item: ("", "") if item is None else (item.description, item.id),
            ):
                candidate = self._candidate(objective, gap, source)
                generated.setdefault(candidate.investigation_id, candidate)
        return tuple(generated[key] for key in sorted(generated))

    def _candidate(
        self,
        objective: ResearchObjective,
        gap: KnowledgeGapLike,
        source: ExistingInvestigation | None,
    ) -> CandidateInvestigation:
        description = gap.description if source is None else source.description
        source_metadata: Mapping[str, object] = {} if source is None else source.metadata
        question = description.strip().rstrip(".?") + "?"
        source_targets = [] if source is None else source.target_entities
        target_entities = tuple(source_targets or gap.affected_entities)
        duration = self._metadata_float(source_metadata, "estimated_duration_seconds", 60.0)
        risk = self._metadata_float(
            source_metadata,
            "risk",
            0.7 if gap.kind == CONTRADICTION else 0.2,
        )
        source_cost = 0.0 if source is None else source.estimated_cost
        cost = source_cost if source_cost > 0 else gap.estimated_cost
        metadata = {
            "objective_id": objective.objective_id,
            "gap_kind": gap.kind,
            "gap_reason": gap.reason,
            "legacy_investigation_id": None if source is None else source.id,
            "evidence_availability": self._metadata_float(
                source_metadata, "evidence_availability", 0.5
            ),
            **source_metadata,
        }
        return CandidateInvestigation(
            gap_id=gap.id,
            question=question,
            hypothesis=(
                f"Evidence addressing '{gap.description}' will reduce uncertainty for "
                f"'{objective.question}'."
            ),
            required_evidence=_REQUIRED_EVIDENCE.get(
                gap.kind, ("independent provenance-complete evidence",)
            ),
            constraints=objective.constraints,
            expected_information_gain=self._gain.expected_information_gain(gap),
            uncertainty_reduction=self._gain.uncertainty_reduction(gap),
            estimated_cost=max(0.0, cost),
            estimated_duration_seconds=duration,
            risk=min(1.0, max(0.0, risk)),
            priority=min(1.0, max(0.0, gap.priority)),
            capabilities=_CAPABILITIES.get(gap.kind, ("search",)),
            target_entities=target_entities,
            metadata=metadata,
            created_at=objective.created_at,
        )

    @staticmethod
    def _metadata_float(metadata: Mapping[str, object], key: str, default: float) -> float:
        value = metadata.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        return float(value)
