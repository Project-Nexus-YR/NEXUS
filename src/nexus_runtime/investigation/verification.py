"""Configurable verification policy for evaluated claims."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from nexus_runtime.models import new_id, utcnow

from .evaluation import ClaimEvaluation, Evaluation
from .evidence import (
    ClaimStatement,
    _parse_timestamp,
    _persisted_bool,
    _persisted_float,
    _persisted_string,
    _persisted_strings,
)


class EpistemicStatus(StrEnum):
    CONFIRMED = "confirmed"
    PROBABLE = "probable"
    UNCERTAIN = "uncertain"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    min_independent_sources: int = 2
    confidence_threshold: float = 0.7
    probable_threshold: float = 0.5
    min_average_source_quality: float = 0.5
    allow_probable_updates: bool = False

    def __post_init__(self) -> None:
        if self.min_independent_sources < 1:
            raise ValueError("min_independent_sources must be positive")
        for value, name in (
            (self.confidence_threshold, "confidence_threshold"),
            (self.probable_threshold, "probable_threshold"),
            (self.min_average_source_quality, "min_average_source_quality"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.probable_threshold > self.confidence_threshold:
            raise ValueError("probable_threshold cannot exceed confidence_threshold")


@dataclass(frozen=True, slots=True)
class VerificationDecision:
    claim: ClaimStatement
    status: EpistemicStatus
    eligible_for_update: bool
    confidence: float
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    unresolved_conflict_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "status": self.status.value,
            "eligible_for_update": self.eligible_for_update,
            "confidence": self.confidence,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
            "unresolved_conflict_ids": list(self.unresolved_conflict_ids),
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VerificationDecision:
        claim = payload.get("claim")
        sequence_fields = (
            "supporting_evidence_ids",
            "contradicting_evidence_ids",
            "unresolved_conflict_ids",
            "reasons",
        )
        if not isinstance(claim, dict) or any(
            not isinstance(payload.get(name), list) for name in sequence_fields
        ):
            raise ValueError("malformed verification decision")
        status = payload.get("status")
        if not isinstance(status, str):
            raise ValueError("malformed verification decision")
        try:
            return cls(
                claim=ClaimStatement.from_dict(claim),
                status=EpistemicStatus(status),
                eligible_for_update=_persisted_bool(payload, "eligible_for_update"),
                confidence=_persisted_float(payload, "confidence"),
                supporting_evidence_ids=_persisted_strings(payload, "supporting_evidence_ids"),
                contradicting_evidence_ids=_persisted_strings(
                    payload, "contradicting_evidence_ids"
                ),
                unresolved_conflict_ids=_persisted_strings(payload, "unresolved_conflict_ids"),
                reasons=_persisted_strings(payload, "reasons"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed verification decision") from exc


@dataclass(frozen=True, slots=True)
class VerificationReport:
    session_id: str
    evaluation_id: str
    decisions: tuple[VerificationDecision, ...]
    verification_id: str = field(default_factory=lambda: new_id("verification"))
    verified_at: datetime = field(default_factory=utcnow)

    @property
    def eligible_claims(self) -> tuple[VerificationDecision, ...]:
        return tuple(decision for decision in self.decisions if decision.eligible_for_update)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "session_id": self.session_id,
            "evaluation_id": self.evaluation_id,
            "decisions": [item.to_dict() for item in self.decisions],
            "verified_at": self.verified_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VerificationReport:
        decisions = payload.get("decisions")
        if not isinstance(decisions, list) or any(not isinstance(item, dict) for item in decisions):
            raise ValueError("malformed verification report")
        try:
            return cls(
                verification_id=_persisted_string(payload, "verification_id"),
                session_id=_persisted_string(payload, "session_id"),
                evaluation_id=_persisted_string(payload, "evaluation_id"),
                decisions=tuple(VerificationDecision.from_dict(item) for item in decisions),
                verified_at=_parse_timestamp(payload["verified_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed verification report") from exc


class ClaimVerifier:
    """Require provenance, independent sources, quality, and no conflict."""

    def __init__(self, policy: VerificationPolicy | None = None) -> None:
        self._policy = policy or VerificationPolicy()

    def verify(self, evaluation: Evaluation) -> VerificationReport:
        return VerificationReport(
            session_id=evaluation.session_id,
            evaluation_id=evaluation.evaluation_id,
            decisions=tuple(self._decision(claim) for claim in evaluation.claims),
        )

    def _decision(self, evaluation: ClaimEvaluation) -> VerificationDecision:
        reasons: list[str] = []
        conflict_ids = tuple(conflict.conflict_id for conflict in evaluation.conflicts)
        if evaluation.unresolved_contradiction:
            reasons.append("unresolved contradiction")
            status = EpistemicStatus.CONTRADICTED
            eligible = False
        elif not evaluation.supporting:
            reasons.append("no acceptable supporting evidence")
            status = EpistemicStatus.INSUFFICIENT_EVIDENCE
            eligible = False
        elif evaluation.independent_source_count < self._policy.min_independent_sources:
            reasons.append(f"independent source count below {self._policy.min_independent_sources}")
            status = EpistemicStatus.INSUFFICIENT_EVIDENCE
            eligible = False
        elif evaluation.average_source_quality < self._policy.min_average_source_quality:
            reasons.append("average source quality below threshold")
            status = EpistemicStatus.UNCERTAIN
            eligible = False
        elif evaluation.aggregate_confidence >= self._policy.confidence_threshold:
            reasons.append("verification criteria satisfied")
            status = EpistemicStatus.CONFIRMED
            eligible = True
        elif evaluation.aggregate_confidence >= self._policy.probable_threshold:
            reasons.append("confidence supports a probable conclusion")
            status = EpistemicStatus.PROBABLE
            eligible = self._policy.allow_probable_updates
        else:
            reasons.append("confidence below verification threshold")
            status = EpistemicStatus.UNCERTAIN
            eligible = False

        return VerificationDecision(
            claim=evaluation.claim,
            status=status,
            eligible_for_update=eligible,
            confidence=evaluation.aggregate_confidence,
            supporting_evidence_ids=tuple(item.evidence_id for item in evaluation.supporting),
            contradicting_evidence_ids=tuple(item.evidence_id for item in evaluation.contradicting),
            unresolved_conflict_ids=conflict_ids,
            reasons=tuple(reasons),
        )
