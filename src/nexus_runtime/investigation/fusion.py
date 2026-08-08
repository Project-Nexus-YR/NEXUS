"""Deterministic evidence deduplication and conflict discovery."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations

from nexus_knowledge.domain.contradiction import ContradictionKind
from nexus_runtime.models import utcnow

from .evidence import ClaimStatement, Evidence, EvidenceRole, EvidenceSet


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


@dataclass(frozen=True, slots=True)
class DuplicateEvidence:
    original_evidence_id: str
    duplicate_evidence_id: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class EvidenceConflict:
    """An unresolved conflict with both claims and both evidence sets preserved."""

    claim_a: ClaimStatement
    claim_b: ClaimStatement
    evidence_a_ids: tuple[str, ...]
    evidence_b_ids: tuple[str, ...]
    kind: str = ContradictionKind.CONFLICTING_CLAIMS
    conflict_id: str = ""
    detected_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if not self.conflict_id:
            claim_ids = sorted((self.claim_a.claim_id, self.claim_b.claim_id))
            object.__setattr__(
                self,
                "conflict_id",
                _stable_id("evidence_conflict", self.kind, *claim_ids),
            )


@dataclass(frozen=True, slots=True)
class FusedClaim:
    claim: ClaimStatement
    supporting: tuple[Evidence, ...]
    contradicting: tuple[Evidence, ...]
    neutral: tuple[Evidence, ...]

    @property
    def all_evidence(self) -> tuple[Evidence, ...]:
        return self.supporting + self.contradicting + self.neutral


@dataclass(frozen=True, slots=True)
class FusionResult:
    evidence_set_id: str
    claims: tuple[FusedClaim, ...]
    duplicates: tuple[DuplicateEvidence, ...]
    conflicts: tuple[EvidenceConflict, ...]


class EvidenceFusion:
    """Fuse identical claims while retaining independent corroboration."""

    def fuse(self, evidence_set: EvidenceSet) -> FusionResult:
        unique: list[Evidence] = []
        duplicates: list[DuplicateEvidence] = []
        first_by_fingerprint: dict[str, Evidence] = {}
        for item in evidence_set.evidence:
            original = first_by_fingerprint.get(item.fingerprint)
            if original is not None:
                duplicates.append(
                    DuplicateEvidence(
                        original_evidence_id=original.evidence_id,
                        duplicate_evidence_id=item.evidence_id,
                        fingerprint=item.fingerprint,
                    )
                )
                continue
            first_by_fingerprint[item.fingerprint] = item
            unique.append(item)

        by_claim: dict[tuple[str, str, str], list[Evidence]] = {}
        for item in unique:
            by_claim.setdefault(item.claim.identity, []).append(item)

        fused: list[FusedClaim] = []
        for identity in sorted(by_claim):
            items = sorted(by_claim[identity], key=lambda item: item.evidence_id)
            canonical = min((item.claim for item in items), key=lambda claim: claim.claim_id)
            fused.append(
                FusedClaim(
                    claim=canonical,
                    supporting=tuple(
                        item for item in items if item.role == EvidenceRole.SUPPORTING
                    ),
                    contradicting=tuple(
                        item for item in items if item.role == EvidenceRole.CONTRADICTING
                    ),
                    neutral=tuple(item for item in items if item.role == EvidenceRole.NEUTRAL),
                )
            )

        conflicts = self._conflicts(fused)
        return FusionResult(
            evidence_set_id=evidence_set.evidence_set_id,
            claims=tuple(fused),
            duplicates=tuple(duplicates),
            conflicts=tuple(conflicts),
        )

    @staticmethod
    def _conflicts(claims: list[FusedClaim]) -> list[EvidenceConflict]:
        by_predicate: dict[tuple[str, str], list[FusedClaim]] = {}
        for claim in claims:
            by_predicate.setdefault(claim.claim.contradiction_key, []).append(claim)

        conflicts: list[EvidenceConflict] = []
        for key in sorted(by_predicate):
            candidates = by_predicate[key]
            for left, right in combinations(candidates, 2):
                if left.claim.identity[2] == right.claim.identity[2]:
                    continue
                left_evidence = tuple(item.evidence_id for item in left.supporting)
                right_evidence = tuple(item.evidence_id for item in right.supporting)
                if not left_evidence or not right_evidence:
                    continue
                conflicts.append(
                    EvidenceConflict(
                        claim_a=left.claim,
                        claim_b=right.claim,
                        evidence_a_ids=left_evidence,
                        evidence_b_ids=right_evidence,
                    )
                )
        return conflicts
