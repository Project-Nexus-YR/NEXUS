"""Deterministic contradiction detection.

Detects conflicting claims, mutually exclusive relations, stale claims
and source disagreement. Contradictions are recorded as first-class
objects; no information is ever deleted automatically — both sides are
preserved along with their evidence.
"""

from __future__ import annotations

from collections import defaultdict

from ..domain.claim import Claim
from ..domain.common import VerificationState
from ..domain.contradiction import Contradiction, ContradictionKind
from ..domain.entity import Relation
from ..port.repository import (
    ClaimRepository,
    ContradictionRepository,
    EvidenceRepository,
    RelationRepository,
)

__all__ = ["ContradictionDetector"]


class ContradictionDetector:
    """Detects explicit contradictions in the knowledge base."""

    def __init__(
        self,
        claims: ClaimRepository,
        relations: RelationRepository,
        evidence: EvidenceRepository,
        contradictions: ContradictionRepository,
        stale_age_days: float = 730.0,
    ) -> None:
        self._claims = claims
        self._relations = relations
        self._evidence = evidence
        self._contradictions = contradictions
        self._stale_age_days = stale_age_days

    def detect(self) -> list[Contradiction]:
        """Scan the knowledge base and persist every detected contradiction."""
        contradictions = self._conflicting_claims()
        contradictions += self._mutually_exclusive_relations()
        contradictions += self._stale_claims()
        for contradiction in contradictions:
            self._contradictions.save(contradiction)
        return contradictions

    # -- rules --------------------------------------------------------
    def _conflicting_claims(self) -> list[Contradiction]:
        """Two claims with the same subject+predicate but different object."""
        index: dict[tuple[str, str], list[Claim]] = defaultdict(list)
        for claim in self._claims.all():
            if not claim.subject or not claim.predicate or not claim.object:
                continue
            key = (claim.subject.lower(), claim.predicate)
            index[key].append(claim)
        contradictions: list[Contradiction] = []
        for (_, predicate), claims in index.items():
            by_object: dict[str, list[Claim]] = defaultdict(list)
            for claim in claims:
                by_object[claim.object.lower()].append(claim)
            if len(by_object) < 2:
                continue
            groups = list(by_object.values())
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    contradiction = self._pair_contradiction(groups[i][0], groups[j][0], predicate)
                    contradictions.append(contradiction)
        return contradictions

    def _mutually_exclusive_relations(self) -> list[Contradiction]:
        """Two relations with the same subject+predicate but different object."""
        index: dict[tuple[str, str], list[Relation]] = defaultdict(list)
        for relation in self._relations.all():
            key = (relation.subject_id, relation.predicate)
            index[key].append(relation)
        contradictions: list[Contradiction] = []
        for (_, predicate), relations in index.items():
            by_object: dict[str, list[Relation]] = defaultdict(list)
            for relation in relations:
                by_object[relation.object_id].append(relation)
            if len(by_object) < 2:
                continue
            groups = list(by_object.values())
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    a, b = groups[i][0], groups[j][0]
                    contradictions.append(
                        Contradiction(
                            kind=ContradictionKind.MUTUALLY_EXCLUSIVE_RELATIONS,
                            claim_a_id=a.id,
                            claim_b_id=b.id,
                            description=(
                                f"relation {a.subject_id[:8]} -{predicate}-> {a.object_id[:8]} "
                                f"conflicts with -> {b.object_id[:8]}"
                            ),
                            evidence_a=list(a.supporting_evidence),
                            evidence_b=list(b.supporting_evidence),
                            strength=self._pair_strength(a, b),
                        )
                    )
        return contradictions

    def _stale_claims(self) -> list[Contradiction]:
        """Claims flagged stale by the uncertainty model."""
        contradictions: list[Contradiction] = []
        for claim in self._claims.all():
            if claim.verification_state == VerificationState.STALE:
                contradictions.append(
                    Contradiction(
                        kind=ContradictionKind.STALE_CLAIM,
                        claim_a_id=claim.id,
                        claim_b_id=claim.id,
                        description=f"claim {claim.id[:8]} is stale",
                        evidence_a=list(claim.supporting_evidence),
                        evidence_b=list(claim.contradicting_evidence),
                        strength=0.5,
                    )
                )
        return contradictions

    # -- helpers ------------------------------------------------------
    def _pair_contradiction(
        self, a: Claim, b: Claim, predicate: str
    ) -> Contradiction:
        strength = self._pair_strength(a, b)
        return Contradiction(
            kind=ContradictionKind.CONFLICTING_CLAIMS,
            claim_a_id=a.id,
            claim_b_id=b.id,
            description=(
                f"claims '{a.text}' and '{b.text}' assert different "
                f"'{predicate}' values"
            ),
            evidence_a=list(a.supporting_evidence),
            evidence_b=list(b.supporting_evidence),
            strength=strength,
        )

    def _pair_strength(self, a: Claim | Relation, b: Claim | Relation) -> float:
        evidence_a = float(len(getattr(a, "supporting_evidence", [])))
        evidence_b = float(len(getattr(b, "supporting_evidence", [])))
        combined_confidence = float(a.confidence) * float(b.confidence)
        return min(1.0, combined_confidence * (1.0 + evidence_a + evidence_b) / 3.0)
