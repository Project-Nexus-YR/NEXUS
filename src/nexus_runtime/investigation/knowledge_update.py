"""Translate verified investigation claims through the knowledge service boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from nexus_knowledge.domain.claim import (
    Claim,
)
from nexus_knowledge.domain.claim import (
    Evidence as KnowledgeEvidence,
)
from nexus_knowledge.domain.claim import (
    EvidenceRole as KnowledgeEvidenceRole,
)
from nexus_knowledge.domain.common import Confidence, VerificationState
from nexus_knowledge.domain.contradiction import Contradiction
from nexus_knowledge.knowledge.uncertainty import UncertaintyAssessment
from nexus_knowledge.service.engine import KnowledgeUpdate, KnowledgeUpdateReceipt
from nexus_runtime.models import new_id, utcnow

from .evidence import (
    Evidence,
    EvidenceSet,
    _parse_timestamp,
    _persisted_string,
    _persisted_strings,
)
from .provenance import EvidenceProvenance
from .verification import VerificationDecision, VerificationReport


class KnowledgeUpdatePort(Protocol):
    """Only public knowledge-service operations used by Track B."""

    def commit_knowledge_update(self, update: KnowledgeUpdate) -> KnowledgeUpdateReceipt: ...

    def detect_contradictions(self) -> list[Contradiction]: ...

    def verify_claim(self, claim_id: str) -> UncertaintyAssessment: ...

    def validate_evidence_provenance(
        self,
        source_id: str,
        document_id: str,
        chunk_id: str,
        source_reference: str,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class PreparedClaim:
    decision: VerificationDecision
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        actual = {item.evidence_id for item in self.evidence}
        expected = set(self.decision.supporting_evidence_ids)
        if actual != expected:
            raise ValueError("prepared claim evidence does not match verification decision")
        if not self.decision.eligible_for_update:
            raise ValueError("ineligible claim cannot be prepared for knowledge update")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PreparedClaim:
        decision = payload.get("decision")
        evidence = payload.get("evidence")
        if not isinstance(decision, dict) or not isinstance(evidence, list):
            raise ValueError("malformed prepared claim")
        if any(not isinstance(item, dict) for item in evidence):
            raise ValueError("malformed prepared claim evidence")
        return cls(
            decision=VerificationDecision.from_dict(decision),
            evidence=tuple(Evidence.from_dict(item) for item in evidence),
        )


@dataclass(frozen=True, slots=True)
class InvestigationKnowledgeUpdate:
    """Inspectable update proposal retaining its complete evidence lineage."""

    session_id: str
    verification_id: str
    claims: tuple[PreparedClaim, ...]
    update_id: str = field(default_factory=lambda: new_id("knowledge_update"))
    created_at: datetime = field(default_factory=utcnow)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for claim in self.claims for item in claim.evidence)

    @property
    def provenance(self) -> tuple[EvidenceProvenance, ...]:
        return tuple(item.provenance for claim in self.claims for item in claim.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "session_id": self.session_id,
            "verification_id": self.verification_id,
            "claims": [item.to_dict() for item in self.claims],
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InvestigationKnowledgeUpdate:
        claims = payload.get("claims")
        if not isinstance(claims, list) or any(not isinstance(item, dict) for item in claims):
            raise ValueError("malformed investigation knowledge update")
        try:
            return cls(
                update_id=_persisted_string(payload, "update_id"),
                session_id=_persisted_string(payload, "session_id"),
                verification_id=_persisted_string(payload, "verification_id"),
                claims=tuple(PreparedClaim.from_dict(item) for item in claims),
                created_at=_parse_timestamp(payload["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed investigation knowledge update") from exc


@dataclass(frozen=True, slots=True)
class KnowledgeUpdateResult:
    update_id: str
    accepted_records: int
    rejected_records: int
    errors: tuple[str, ...]
    committed_claim_ids: tuple[str, ...]
    unresolved_contradiction_ids: tuple[str, ...]
    verification_states: dict[str, str]
    applied_at: datetime = field(default_factory=utcnow)

    @property
    def fully_applied(self) -> bool:
        return self.rejected_records == 0 and not self.unresolved_contradiction_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "accepted_records": self.accepted_records,
            "rejected_records": self.rejected_records,
            "errors": list(self.errors),
            "committed_claim_ids": list(self.committed_claim_ids),
            "unresolved_contradiction_ids": list(self.unresolved_contradiction_ids),
            "verification_states": dict(self.verification_states),
            "applied_at": self.applied_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> KnowledgeUpdateResult:
        sequence_fields = (
            "errors",
            "committed_claim_ids",
            "unresolved_contradiction_ids",
        )
        states = payload.get("verification_states")
        invalid_sequences = any(not isinstance(payload.get(name), list) for name in sequence_fields)
        if invalid_sequences or not isinstance(states, dict):
            raise ValueError("malformed knowledge update result")
        accepted = payload.get("accepted_records")
        rejected = payload.get("rejected_records")
        if (
            isinstance(accepted, bool)
            or not isinstance(accepted, int)
            or isinstance(rejected, bool)
            or not isinstance(rejected, int)
        ):
            raise ValueError("malformed knowledge update result")
        if any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in states.items()
        ):
            raise ValueError("malformed knowledge update result")
        try:
            return cls(
                update_id=_persisted_string(payload, "update_id"),
                accepted_records=accepted,
                rejected_records=rejected,
                errors=_persisted_strings(payload, "errors"),
                committed_claim_ids=_persisted_strings(payload, "committed_claim_ids"),
                unresolved_contradiction_ids=_persisted_strings(
                    payload, "unresolved_contradiction_ids"
                ),
                verification_states=dict(states),
                applied_at=_parse_timestamp(payload["applied_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed knowledge update result") from exc


class KnowledgeUpdateIntegrator:
    """Submit only verified claims through ``commit_knowledge_update``.

    Claims enter the knowledge service as ``UNVERIFIED``.  The existing
    contradiction analyzer then checks them against persisted knowledge.  Only
    conflict-free claims advance through the existing uncertainty verifier.
    Unresolved conflicts remain inspectable instead of being resolved by agent
    confidence or by direct storage mutation.
    """

    def __init__(self, knowledge: KnowledgeUpdatePort) -> None:
        self._knowledge = knowledge

    def prepare(
        self,
        report: VerificationReport,
        evidence_set: EvidenceSet,
    ) -> InvestigationKnowledgeUpdate:
        if report.session_id != evidence_set.session_id:
            raise ValueError("verification report and evidence set session ids differ")
        evidence_by_id = {item.evidence_id: item for item in evidence_set.evidence}
        prepared: list[PreparedClaim] = []
        for decision in report.eligible_claims:
            missing = [
                evidence_id
                for evidence_id in decision.supporting_evidence_ids
                if evidence_id not in evidence_by_id
            ]
            if missing:
                raise ValueError(f"verified evidence missing from set: {', '.join(missing)}")
            evidence = tuple(evidence_by_id[item] for item in decision.supporting_evidence_ids)
            if any(not item.provenance.is_complete for item in evidence):
                raise ValueError("knowledge update requires complete evidence provenance")
            invalid = [
                item.evidence_id
                for item in evidence
                if not self._knowledge.validate_evidence_provenance(
                    item.provenance.source_id,
                    item.provenance.document_id,
                    item.provenance.chunk_id,
                    item.provenance.source_reference,
                )
            ]
            if invalid:
                raise ValueError(
                    "knowledge update provenance does not resolve: " + ", ".join(invalid)
                )
            prepared.append(PreparedClaim(decision=decision, evidence=evidence))
        return InvestigationKnowledgeUpdate(
            session_id=report.session_id,
            verification_id=report.verification_id,
            claims=tuple(prepared),
        )

    def apply(self, submission: InvestigationKnowledgeUpdate) -> KnowledgeUpdateResult:
        if not submission.claims:
            return KnowledgeUpdateResult(
                update_id=submission.update_id,
                accepted_records=0,
                rejected_records=0,
                errors=(),
                committed_claim_ids=(),
                unresolved_contradiction_ids=(),
                verification_states={},
            )

        update = self._to_engine_update(submission)
        receipt = self._knowledge.commit_knowledge_update(update)
        claim_ids = tuple(claim.decision.claim.claim_id for claim in submission.claims)
        claim_id_set = set(claim_ids)
        contradictions = self._knowledge.detect_contradictions()
        relevant = tuple(
            contradiction
            for contradiction in contradictions
            if contradiction.claim_a_id in claim_id_set or contradiction.claim_b_id in claim_id_set
        )
        conflicted_claims = {
            claim_id
            for contradiction in relevant
            for claim_id in (contradiction.claim_a_id, contradiction.claim_b_id)
            if claim_id in claim_id_set
        }

        states: dict[str, str] = {}
        for claim_id in claim_ids:
            if claim_id in conflicted_claims:
                states[claim_id] = VerificationState.CONTRADICTED.value
                continue
            assessment = self._knowledge.verify_claim(claim_id)
            states[claim_id] = assessment.verification_state.value

        return KnowledgeUpdateResult(
            update_id=submission.update_id,
            accepted_records=receipt.accepted,
            rejected_records=receipt.rejected,
            errors=tuple(receipt.errors),
            committed_claim_ids=claim_ids,
            unresolved_contradiction_ids=tuple(contradiction.id for contradiction in relevant),
            verification_states=states,
        )

    @staticmethod
    def _to_engine_update(submission: InvestigationKnowledgeUpdate) -> KnowledgeUpdate:
        claims: list[Claim] = []
        evidence: list[KnowledgeEvidence] = []
        for prepared in submission.claims:
            decision = prepared.decision
            lineages = [item.provenance.to_dict() for item in prepared.evidence]
            source_ids = sorted({item.provenance.source_id for item in prepared.evidence})
            chunk_ids = sorted({item.provenance.chunk_id for item in prepared.evidence})
            claims.append(
                Claim(
                    text=decision.claim.text,
                    subject=decision.claim.subject,
                    predicate=decision.claim.predicate,
                    object=decision.claim.object,
                    confidence=Confidence(decision.confidence),
                    provenance=chunk_ids,
                    source_ids=source_ids,
                    supporting_evidence=list(decision.supporting_evidence_ids),
                    contradicting_evidence=list(decision.contradicting_evidence_ids),
                    verification_state=VerificationState.UNVERIFIED,
                    metadata={
                        "investigation_session_id": submission.session_id,
                        "verification_id": submission.verification_id,
                        "epistemic_status": decision.status.value,
                        "evidence_lineage": lineages,
                    },
                    id=decision.claim.claim_id,
                )
            )
            for item in prepared.evidence:
                text = item.excerpt or json.dumps(item.payload, sort_keys=True, default=str)
                role = (
                    KnowledgeEvidenceRole.CONTRADICT
                    if item.evidence_id in decision.contradicting_evidence_ids
                    else KnowledgeEvidenceRole.SUPPORT
                )
                evidence.append(
                    KnowledgeEvidence(
                        claim_id=decision.claim.claim_id,
                        chunk_id=item.provenance.chunk_id,
                        document_id=item.provenance.document_id,
                        text=text,
                        role=role,
                        quality=item.source_quality,
                        id=item.evidence_id,
                        extracted_at=item.timestamp.isoformat(),
                    )
                )
        return KnowledgeUpdate(claims=claims, evidence=evidence)
