"""Explicit claim acquisition lifecycle layered over the verification pipeline.

A candidate claim proposed by an agent moves through an explicit lifecycle:

    Candidate -> Evaluated -> Verified | Rejected | Deferred

This is an acquisition state, not a rename of the knowledge service's
``VerificationState``.  Verified claims proceed to the existing knowledge
update path; deferred claims are preserved with their evidence and reason and
surface as knowledge gaps for a future investigation; rejected claims never
become knowledge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nexus_runtime.models import new_id, utcnow

from .candidate_claims import (
    CandidateClaim,
    CandidateExtractionResult,
    CandidateStatus,
)
from .evidence import _parse_timestamp, _persisted_string
from .verification import EpistemicStatus, VerificationDecision, VerificationReport


@dataclass(frozen=True, slots=True)
class ClaimAcquisition:
    """One candidate's lifecycle outcome for a verification report."""

    candidate: CandidateClaim
    status: CandidateStatus
    reason: str
    decision: VerificationDecision | None = None
    acquisition_id: str = field(default_factory=lambda: new_id("acquisition"))
    created_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "acquisition_id": self.acquisition_id,
            "candidate": self.candidate.to_dict(),
            "status": self.status.value,
            "reason": self.reason,
            "decision": None if self.decision is None else self.decision.to_dict(),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ClaimAcquisition:
        candidate = payload.get("candidate")
        decision = payload.get("decision")
        if not isinstance(candidate, dict) or (
            decision is not None and not isinstance(decision, dict)
        ):
            raise ValueError("malformed claim acquisition")
        try:
            return cls(
                acquisition_id=_persisted_string(payload, "acquisition_id"),
                candidate=CandidateClaim.from_dict(candidate),
                status=CandidateStatus(_persisted_string(payload, "status")),
                reason=_persisted_string(payload, "reason"),
                decision=None if decision is None else VerificationDecision.from_dict(decision),
                created_at=_parse_timestamp(payload["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed claim acquisition") from exc


@dataclass(frozen=True, slots=True)
class AcquisitionReport:
    """Lifecycle outcomes for every extracted candidate claim in one report."""

    session_id: str
    verification_id: str
    acquisitions: tuple[ClaimAcquisition, ...]
    acquisition_id: str = field(default_factory=lambda: new_id("acquisition_report"))
    created_at: datetime = field(default_factory=utcnow)

    @property
    def verified(self) -> tuple[ClaimAcquisition, ...]:
        return tuple(item for item in self.acquisitions if item.status == CandidateStatus.VERIFIED)

    @property
    def rejected(self) -> tuple[ClaimAcquisition, ...]:
        return tuple(item for item in self.acquisitions if item.status == CandidateStatus.REJECTED)

    @property
    def deferred(self) -> tuple[ClaimAcquisition, ...]:
        return tuple(item for item in self.acquisitions if item.status == CandidateStatus.DEFERRED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "verification_id": self.verification_id,
            "acquisitions": [item.to_dict() for item in self.acquisitions],
            "acquisition_id": self.acquisition_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AcquisitionReport:
        acquisitions = payload.get("acquisitions")
        if not isinstance(acquisitions, list) or any(
            not isinstance(item, dict) for item in acquisitions
        ):
            raise ValueError("malformed acquisition report")
        try:
            return cls(
                session_id=_persisted_string(payload, "session_id"),
                verification_id=_persisted_string(payload, "verification_id"),
                acquisitions=tuple(ClaimAcquisition.from_dict(item) for item in acquisitions),
                acquisition_id=_persisted_string(payload, "acquisition_id"),
                created_at=_parse_timestamp(payload["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed acquisition report") from exc


class ClaimAcquisitionService:
    """Deterministic mapping from extracted candidates to lifecycle outcomes.

    The mapping preserves the existing verification policy: candidates whose
    claim is ``eligible_for_update`` are VERIFIED, contradicted or unsupported
    candidates are REJECTED, and under-supported candidates are DEFERRED with
    their evidence and reason preserved.
    """

    def acquire(
        self,
        extraction: CandidateExtractionResult,
        verification: VerificationReport,
    ) -> AcquisitionReport:
        if extraction.session_id != verification.session_id:
            raise ValueError("extraction and verification must belong to the same session")
        decisions = {item.claim.identity: item for item in verification.decisions}
        acquisitions: list[ClaimAcquisition] = []
        for candidate in sorted(extraction.candidates, key=lambda item: item.candidate_id):
            decision = decisions.get(candidate.claim.identity)
            if decision is None:
                acquisitions.append(
                    ClaimAcquisition(
                        candidate=candidate,
                        status=CandidateStatus.REJECTED,
                        reason="no verification decision was produced for the candidate claim",
                    )
                )
                continue
            status, reason = _lifecycle_for(decision)
            acquisitions.append(
                ClaimAcquisition(
                    candidate=candidate,
                    status=status,
                    reason=reason,
                    decision=decision,
                )
            )
        return AcquisitionReport(
            session_id=verification.session_id,
            verification_id=verification.verification_id,
            acquisitions=tuple(acquisitions),
        )


def _lifecycle_for(decision: VerificationDecision) -> tuple[CandidateStatus, str]:
    reasons = " ".join(decision.reasons)
    if decision.eligible_for_update:
        return CandidateStatus.VERIFIED, reasons or "claim passed verification policy"
    if decision.status == EpistemicStatus.CONTRADICTED:
        return CandidateStatus.REJECTED, reasons or "claim is contradicted by verified evidence"
    if decision.status == EpistemicStatus.PROBABLE:
        return CandidateStatus.DEFERRED, reasons or "claim is probable but not eligible for update"
    if decision.status == EpistemicStatus.UNCERTAIN:
        return (
            CandidateStatus.DEFERRED,
            reasons or "claim confidence is below the probable threshold",
        )
    return CandidateStatus.DEFERRED, reasons or "claim lacks sufficient supporting evidence"


__all__ = [
    "AcquisitionReport",
    "ClaimAcquisition",
    "ClaimAcquisitionService",
]
