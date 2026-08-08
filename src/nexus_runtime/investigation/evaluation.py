"""Evidence quality assessment built on deterministic evidence fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from nexus_runtime.models import new_id, utcnow

from .evidence import ClaimStatement, Evidence, EvidenceSet
from .fusion import EvidenceConflict, EvidenceFusion, FusedClaim, FusionResult


@dataclass(frozen=True, slots=True)
class EvidenceQualityPolicy:
    """Minimum item-level quality before evidence can support verification."""

    min_evidence_confidence: float = 0.5
    min_source_quality: float = 0.5

    def __post_init__(self) -> None:
        for value, name in (
            (self.min_evidence_confidence, "min_evidence_confidence"),
            (self.min_source_quality, "min_source_quality"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True, slots=True)
class ClaimEvaluation:
    claim: ClaimStatement
    supporting: tuple[Evidence, ...]
    contradicting: tuple[Evidence, ...]
    neutral: tuple[Evidence, ...]
    duplicate_evidence_ids: tuple[str, ...]
    low_quality_evidence_ids: tuple[str, ...]
    conflicts: tuple[EvidenceConflict, ...]
    aggregate_confidence: float
    average_source_quality: float
    independent_source_count: int

    @property
    def unresolved_contradiction(self) -> bool:
        return bool(self.conflicts or self.contradicting)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Structured assessment; agent responses are never concatenated."""

    session_id: str
    evidence_set_id: str
    claims: tuple[ClaimEvaluation, ...]
    duplicate_evidence_ids: tuple[str, ...]
    low_quality_evidence_ids: tuple[str, ...]
    conflict_ids: tuple[str, ...]
    evaluation_id: str = field(default_factory=lambda: new_id("evaluation"))
    evaluated_at: datetime = field(default_factory=utcnow)

    @property
    def accepted_evidence_count(self) -> int:
        accepted = {
            item.evidence_id
            for claim in self.claims
            for item in (*claim.supporting, *claim.contradicting, *claim.neutral)
        }
        return len(accepted)


class EvidenceEvaluator:
    """Classify supporting, conflicting, duplicate, and low-quality evidence."""

    def __init__(
        self,
        policy: EvidenceQualityPolicy | None = None,
        fusion: EvidenceFusion | None = None,
    ) -> None:
        self._policy = policy or EvidenceQualityPolicy()
        self._fusion = fusion or EvidenceFusion()

    def evaluate(self, evidence_set: EvidenceSet) -> Evaluation:
        fused = self._fusion.fuse(evidence_set)
        duplicate_ids = tuple(item.duplicate_evidence_id for item in fused.duplicates)
        conflicts = self._unresolved_conflicts(fused)
        claims = tuple(self._evaluate_claim(claim, fused, conflicts) for claim in fused.claims)
        low_quality_ids = tuple(
            sorted({item for claim in claims for item in claim.low_quality_evidence_ids})
        )
        return Evaluation(
            session_id=evidence_set.session_id,
            evidence_set_id=evidence_set.evidence_set_id,
            claims=claims,
            duplicate_evidence_ids=duplicate_ids,
            low_quality_evidence_ids=low_quality_ids,
            conflict_ids=tuple(conflict.conflict_id for conflict in conflicts),
        )

    def _evaluate_claim(
        self,
        fused_claim: FusedClaim,
        fusion: FusionResult,
        unresolved_conflicts: tuple[EvidenceConflict, ...],
    ) -> ClaimEvaluation:
        claim = fused_claim.claim
        all_evidence = fused_claim.all_evidence
        low_quality = tuple(item for item in all_evidence if not self._acceptable(item))
        accepted = tuple(item for item in all_evidence if self._acceptable(item))
        supporting = tuple(item for item in fused_claim.supporting if item in accepted)
        explicit_contra = tuple(item for item in fused_claim.contradicting if item in accepted)
        conflicts = tuple(
            conflict
            for conflict in unresolved_conflicts
            if claim.claim_id in (conflict.claim_a.claim_id, conflict.claim_b.claim_id)
        )
        opposing_ids = {
            evidence_id
            for conflict in conflicts
            for evidence_id in (
                conflict.evidence_b_ids
                if conflict.claim_a.claim_id == claim.claim_id
                else conflict.evidence_a_ids
            )
        }
        by_id = {
            item.evidence_id: item
            for candidate in fusion.claims
            for item in candidate.supporting
            if self._acceptable(item)
        }
        opposing = tuple(by_id[evidence_id] for evidence_id in sorted(opposing_ids))
        contradicting = explicit_contra + opposing
        neutral = tuple(item for item in fused_claim.neutral if item in accepted)
        duplicate_ids = tuple(
            duplicate.duplicate_evidence_id
            for duplicate in fusion.duplicates
            if duplicate.original_evidence_id in {item.evidence_id for item in all_evidence}
        )
        aggregate = self._aggregate_confidence(supporting, contradicting)
        source_quality = (
            sum(item.source_quality for item in supporting) / len(supporting) if supporting else 0.0
        )
        independent_sources = len({item.provenance.source_id for item in supporting})
        return ClaimEvaluation(
            claim=claim,
            supporting=supporting,
            contradicting=contradicting,
            neutral=neutral,
            duplicate_evidence_ids=duplicate_ids,
            low_quality_evidence_ids=tuple(item.evidence_id for item in low_quality),
            conflicts=conflicts,
            aggregate_confidence=aggregate,
            average_source_quality=source_quality,
            independent_source_count=independent_sources,
        )

    def _unresolved_conflicts(self, fusion: FusionResult) -> tuple[EvidenceConflict, ...]:
        by_id = {item.evidence_id: item for claim in fusion.claims for item in claim.supporting}
        unresolved: list[EvidenceConflict] = []
        for conflict in fusion.conflicts:
            left = tuple(
                evidence_id
                for evidence_id in conflict.evidence_a_ids
                if self._acceptable(by_id[evidence_id])
            )
            right = tuple(
                evidence_id
                for evidence_id in conflict.evidence_b_ids
                if self._acceptable(by_id[evidence_id])
            )
            if left and right:
                unresolved.append(
                    EvidenceConflict(
                        claim_a=conflict.claim_a,
                        claim_b=conflict.claim_b,
                        evidence_a_ids=left,
                        evidence_b_ids=right,
                        kind=conflict.kind,
                        conflict_id=conflict.conflict_id,
                        detected_at=conflict.detected_at,
                    )
                )
        return tuple(unresolved)

    def _acceptable(self, evidence: Evidence) -> bool:
        return (
            evidence.confidence >= self._policy.min_evidence_confidence
            and evidence.source_quality >= self._policy.min_source_quality
        )

    @staticmethod
    def _aggregate_confidence(
        supporting: tuple[Evidence, ...], contradicting: tuple[Evidence, ...]
    ) -> float:
        support = sum(item.confidence * item.source_quality for item in supporting)
        contradiction = sum(item.confidence * item.source_quality for item in contradicting)
        if support == 0.0:
            return 0.0
        evidence_weight = support + contradiction
        balance = support / evidence_weight if evidence_weight else 0.0
        mean_support = support / len(supporting)
        return min(1.0, max(0.0, mean_support * balance))
