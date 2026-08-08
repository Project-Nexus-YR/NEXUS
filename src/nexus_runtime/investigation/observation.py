"""Structured knowledge observations consumed by investigation planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from importlib import import_module
from typing import Any, cast

from nexus_runtime.models import DomainError

from .generator import ExistingInvestigation, KnowledgeGapLike, _stable_id
from .objective import (
    ResearchObjective,
    _required_string,
    _string_tuple,
    _timestamp_from_text,
    _timestamp_to_text,
    _validate_json,
    _validate_timestamp,
)


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainError(f"{field_name} must be numeric")
    return float(value)


def _serialize_existing(investigation: ExistingInvestigation) -> dict[str, object]:
    return {
        "id": investigation.id,
        "gap_id": investigation.gap_id,
        "description": investigation.description,
        "target_entities": list(investigation.target_entities),
        "expected_information_gain": investigation.expected_information_gain,
        "uncertainty_reduction": investigation.uncertainty_reduction,
        "importance": investigation.importance,
        "estimated_cost": investigation.estimated_cost,
        "score": investigation.score,
        "metadata": dict(investigation.metadata),
        "created_at": investigation.created_at,
    }


def _serialize_gap(gap: KnowledgeGapLike) -> dict[str, object]:
    return {
        "id": gap.id,
        "kind": gap.kind,
        "description": gap.description,
        "reason": gap.reason,
        "affected_entities": list(gap.affected_entities),
        "affected_relations": list(gap.affected_relations),
        "affected_claims": list(gap.affected_claims),
        "uncertainty": gap.uncertainty,
        "importance": gap.importance,
        "estimated_cost": gap.estimated_cost,
        "candidate_investigations": [
            _serialize_existing(item) for item in gap.candidate_investigations
        ],
        "metadata": dict(gap.metadata),
        "created_at": gap.created_at,
    }


def _deserialize_gap(payload: object) -> KnowledgeGapLike:
    if not isinstance(payload, dict):
        raise DomainError("knowledge snapshot gap must be an object")
    investigation_type = cast(
        "Any",
        import_module("nexus_knowledge.domain.knowledge_gap").Investigation,
    )
    gap_type = cast(
        "Any",
        import_module("nexus_knowledge.domain.knowledge_gap").KnowledgeGap,
    )
    raw_candidates = payload.get("candidate_investigations", [])
    if not isinstance(raw_candidates, list):
        raise DomainError("candidate_investigations must be a list")
    candidates: list[object] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise DomainError("candidate investigation must be an object")
        candidates.append(
            investigation_type(
                id=_required_string(raw["id"], "candidate id"),
                gap_id=_required_string(raw["gap_id"], "candidate gap_id"),
                description=_required_string(raw["description"], "candidate description"),
                target_entities=list(
                    _string_tuple(raw.get("target_entities", []), "target_entities")
                ),
                expected_information_gain=_number(
                    raw.get("expected_information_gain", 0.0),
                    "expected_information_gain",
                ),
                uncertainty_reduction=_number(
                    raw.get("uncertainty_reduction", 0.0), "uncertainty_reduction"
                ),
                importance=_number(raw.get("importance", 0.0), "importance"),
                estimated_cost=_number(raw.get("estimated_cost", 0.0), "estimated_cost"),
                score=_number(raw.get("score", 0.0), "score"),
                metadata=dict(raw.get("metadata", {})),
                created_at=_required_string(raw["created_at"], "candidate created_at"),
            )
        )
    try:
        result = gap_type(
            id=_required_string(payload["id"], "gap id"),
            kind=_required_string(payload["kind"], "gap kind"),
            description=_required_string(payload["description"], "gap description"),
            reason=_required_string(payload["reason"], "gap reason"),
            affected_entities=list(
                _string_tuple(payload.get("affected_entities", []), "affected_entities")
            ),
            affected_relations=list(
                _string_tuple(payload.get("affected_relations", []), "affected_relations")
            ),
            affected_claims=list(
                _string_tuple(payload.get("affected_claims", []), "affected_claims")
            ),
            uncertainty=_number(payload.get("uncertainty", 0.0), "uncertainty"),
            importance=_number(payload.get("importance", 0.0), "importance"),
            estimated_cost=_number(payload.get("estimated_cost", 0.0), "estimated_cost"),
            candidate_investigations=candidates,
            metadata=dict(payload.get("metadata", {})),
            created_at=_required_string(payload["created_at"], "gap created_at"),
        )
    except KeyError as exc:
        raise DomainError("malformed knowledge snapshot gap") from exc
    return cast(KnowledgeGapLike, result)


@dataclass(frozen=True, slots=True)
class KnowledgeSnapshot:
    """The minimum inspectable knowledge state required by Track A planning."""

    objective_id: str
    query: str
    gaps: tuple[KnowledgeGapLike, ...]
    retrieval_refs: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    contradiction_ids: tuple[str, ...] = ()
    mean_uncertainty: float = 0.0
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    snapshot_id: str = ""
    observed_at: datetime | None = None

    def __post_init__(self) -> None:
        objective_id = self.objective_id.strip()
        query = self.query.strip()
        if not objective_id or not query:
            raise DomainError("snapshot objective_id and query are required")
        if not 0.0 <= self.mean_uncertainty <= 1.0:
            raise DomainError("mean_uncertainty must be between zero and one")
        observed_at = self.observed_at
        if observed_at is None:
            raise DomainError("observed_at is required for reproducibility")
        _validate_timestamp(observed_at, "observed_at")
        _validate_json(self.metadata, "snapshot metadata")
        gaps = tuple(sorted(self.gaps, key=lambda item: (-item.priority, item.id)))
        for gap in gaps:
            if not gap.id.strip():
                raise DomainError("snapshot gaps require identifiers")
        snapshot_id = self.snapshot_id.strip() or _stable_id(
            "snapshot",
            objective_id,
            query,
            tuple(gap.id for gap in gaps),
            tuple(sorted(self.contradiction_ids)),
            _timestamp_to_text(observed_at),
        )
        object.__setattr__(self, "objective_id", objective_id)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "gaps", gaps)
        object.__setattr__(self, "retrieval_refs", tuple(sorted(set(self.retrieval_refs))))
        object.__setattr__(self, "entity_ids", tuple(sorted(set(self.entity_ids))))
        object.__setattr__(self, "relation_ids", tuple(sorted(set(self.relation_ids))))
        object.__setattr__(self, "contradiction_ids", tuple(sorted(set(self.contradiction_ids))))
        object.__setattr__(self, "summary", self.summary.strip())
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "snapshot_id", snapshot_id)

    @classmethod
    def capture(
        cls,
        objective: ResearchObjective,
        gaps: Sequence[KnowledgeGapLike],
        *,
        observed_at: datetime,
        retrieval_refs: Sequence[str] = (),
        entity_ids: Sequence[str] = (),
        relation_ids: Sequence[str] = (),
        contradiction_ids: Sequence[str] = (),
        summary: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> KnowledgeSnapshot:
        mean_uncertainty = sum(gap.uncertainty for gap in gaps) / len(gaps) if gaps else 0.0
        return cls(
            objective_id=objective.objective_id,
            query=objective.question,
            gaps=tuple(gaps),
            retrieval_refs=tuple(retrieval_refs),
            entity_ids=tuple(entity_ids),
            relation_ids=tuple(relation_ids),
            contradiction_ids=tuple(contradiction_ids),
            mean_uncertainty=mean_uncertainty,
            summary=summary,
            metadata={} if metadata is None else dict(metadata),
            observed_at=observed_at,
        )

    def to_dict(self) -> dict[str, object]:
        assert self.observed_at is not None
        return {
            "snapshot_id": self.snapshot_id,
            "objective_id": self.objective_id,
            "query": self.query,
            "gaps": [_serialize_gap(gap) for gap in self.gaps],
            "retrieval_refs": list(self.retrieval_refs),
            "entity_ids": list(self.entity_ids),
            "relation_ids": list(self.relation_ids),
            "contradiction_ids": list(self.contradiction_ids),
            "mean_uncertainty": self.mean_uncertainty,
            "summary": self.summary,
            "metadata": dict(self.metadata),
            "observed_at": _timestamp_to_text(self.observed_at),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> KnowledgeSnapshot:
        raw_gaps = payload.get("gaps")
        metadata = payload.get("metadata")
        if not isinstance(raw_gaps, list) or not isinstance(metadata, dict):
            raise DomainError("malformed KnowledgeSnapshot")
        try:
            return cls(
                snapshot_id=_required_string(payload["snapshot_id"], "snapshot_id"),
                objective_id=_required_string(payload["objective_id"], "objective_id"),
                query=_required_string(payload["query"], "query"),
                gaps=tuple(_deserialize_gap(gap) for gap in raw_gaps),
                retrieval_refs=_string_tuple(payload["retrieval_refs"], "retrieval_refs"),
                entity_ids=_string_tuple(payload["entity_ids"], "entity_ids"),
                relation_ids=_string_tuple(payload["relation_ids"], "relation_ids"),
                contradiction_ids=_string_tuple(payload["contradiction_ids"], "contradiction_ids"),
                mean_uncertainty=_number(payload["mean_uncertainty"], "mean_uncertainty"),
                summary=_required_string(payload["summary"], "summary"),
                metadata=dict(metadata),
                observed_at=_timestamp_from_text(payload["observed_at"], "observed_at"),
            )
        except KeyError as exc:
            raise DomainError("malformed KnowledgeSnapshot") from exc
