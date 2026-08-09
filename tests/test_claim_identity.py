"""Section 6: deterministic identity across claim, evidence, and fusion keys.

The system must derive identical identifiers from identical semantics so that
replayed runs, retried extractions, and fused sessions cannot create phantom
claim identities.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from nexus_runtime.investigation.candidate_claims import CandidateClaimExtractor
from nexus_runtime.investigation.evidence import (
    AgentConclusion,
    ClaimStatement,
    Evidence,
    EvidenceRole,
    EvidenceSet,
    InvestigationResult,
    InvestigationResultState,
    ToolObservation,
    _stable_id,
)
from nexus_runtime.investigation.provenance import EvidenceProvenance

SESSION = "session-id1"
INVESTIGATION = "investigation-id1"
TASK = "task-id1"
ATTEMPT = "attempt-id1"
RUN = "run-id1"


def evidence(
    claim_statement: ClaimStatement,
    source_id: str,
    excerpt: str,
    *,
    confidence: float = 0.8,
    role: EvidenceRole = EvidenceRole.SUPPORTING,
    payload: dict | None = None,
) -> Evidence:
    return Evidence(
        investigation_id=INVESTIGATION,
        source=f"source://{source_id}",
        claim=claim_statement,
        provenance=EvidenceProvenance(
            session_id=SESSION,
            investigation_id=INVESTIGATION,
            task_id=TASK,
            attempt_id=ATTEMPT,
            run_id=RUN,
            tool_call_id=f"observation-{source_id}",
            source_id=source_id,
            document_id=f"document-{source_id}",
            chunk_id=f"chunk-{source_id}",
            source_reference=f"source://{source_id}",
        ),
        confidence=confidence,
        source_quality=0.9,
        excerpt=excerpt,
        payload=payload or {},
        role=role,
        evidence_id=_stable_id("evidence", claim_statement.claim_id, f"observation-{source_id}"),
    )


def active_claim() -> ClaimStatement:
    return ClaimStatement(
        text="Acme is active", subject="Acme", predicate="status", object="active"
    )


def test_claim_identity_normalizes_case_and_whitespace() -> None:
    left = active_claim()
    right = ClaimStatement(
        text="Acme is Active",
        subject="  Acme ",
        predicate="  status ",
        object=" Active",
    )

    assert left.identity == right.identity
    assert left.identity == ("acme", "status", "active")


def test_claim_id_is_deterministic_from_identity() -> None:
    first = active_claim()
    second = ClaimStatement(
        text="Acme is Active", subject="Acme", predicate="status", object="Active"
    )

    assert first.claim_id == second.claim_id
    assert first.claim_id == _stable_id("claim", *first.identity)


def test_different_identity_produces_different_claim_id() -> None:
    active = active_claim()
    inactive = ClaimStatement(
        text="Acme is inactive", subject="Acme", predicate="status", object="inactive"
    )

    assert active.identity != inactive.identity
    assert active.claim_id != inactive.claim_id


def test_explicit_claim_id_is_preserved() -> None:
    statement = ClaimStatement(
        text="Acme is active",
        subject="Acme",
        predicate="status",
        object="active",
        claim_id="claim-explicit",
    )

    assert statement.claim_id == "claim-explicit"


def test_claim_id_cannot_represent_multiple_identities_in_one_set() -> None:
    first = active_claim()
    impostor = ClaimStatement(
        text="Acme is dormant", subject="Acme", predicate="status", object="dormant"
    )
    forged = replace(impostor, claim_id=first.claim_id)

    with pytest.raises(ValueError, match="cannot represent multiple claim identities"):
        EvidenceSet(
            session_id=SESSION,
            evidence=(
                evidence(first, "source-a", "report one"),
                evidence(forged, "source-b", "report two"),
            ),
        )


def test_contradiction_key_is_subject_and_predicate() -> None:
    active = active_claim()
    inactive = ClaimStatement(
        text="Acme is inactive", subject="Acme", predicate="status", object="inactive"
    )
    other_subject = ClaimStatement(
        text="Beta is active", subject="Beta", predicate="status", object="active"
    )

    assert active.contradiction_key == inactive.contradiction_key == ("acme", "status")
    assert active.contradiction_key != other_subject.contradiction_key


def test_evidence_id_is_deterministic_per_claim_and_observation() -> None:
    statement = ClaimStatement(
        text="Acme is active", subject="Acme", predicate="status", object="active"
    )

    first = evidence(statement, "source-a", "report")
    second = evidence(statement, "source-a", "report")
    other_source = evidence(statement, "source-b", "report")

    assert first.evidence_id == second.evidence_id
    assert first.evidence_id == _stable_id("evidence", statement.claim_id, "observation-source-a")
    assert first.evidence_id != other_source.evidence_id


def test_candidate_id_is_deterministic_regardless_of_observation_order() -> None:
    statement = ClaimStatement(
        text="Acme is active", subject="Acme", predicate="status", object="active"
    )

    first = AgentConclusion(
        claim=statement,
        supporting_observation_ids=("observation-a", "observation-b"),
        confidence=0.8,
    )
    second = AgentConclusion(
        claim=statement,
        supporting_observation_ids=("observation-b", "observation-a"),
        confidence=0.8,
    )

    assert first.supporting_observation_ids == second.supporting_observation_ids
    assert first.conclusion_id == second.conclusion_id

    obs = tuple(
        ToolObservation(
            observation_id=observation_id,
            tool_name="search",
            status="SUCCEEDED",
            input={},
            output={"text": "report"},
            source_reference=f"source://{observation_id}",
            metadata={
                "source_id": observation_id,
                "document_id": f"doc-{observation_id}",
                "chunk_id": f"chunk-{observation_id}",
            },
        )
        for observation_id in ("observation-a", "observation-b")
    )

    def _candidate_id(conclusion_statement: AgentConclusion) -> str:
        investigation_result = InvestigationResult(
            session_id=SESSION,
            investigation_id=INVESTIGATION,
            task_id=TASK,
            attempt_id=ATTEMPT,
            run_id=RUN,
            state=InvestigationResultState.COMPLETED,
            evidence_set=EvidenceSet(session_id=SESSION, evidence=()),
            conclusions=(conclusion_statement,),
            observations=obs,
        )
        return CandidateClaimExtractor().extract(investigation_result).candidates[0].candidate_id

    assert _candidate_id(first) == _candidate_id(second)


def test_evidence_fingerprint_is_stable_for_identical_content() -> None:
    statement = ClaimStatement(
        text="Acme is active", subject="Acme", predicate="status", object="active"
    )

    assert evidence(statement, "source-a", "report one").fingerprint == evidence(
        statement, "source-a", "report one"
    ).fingerprint


def test_evidence_fingerprint_is_source_sensitive() -> None:
    statement = ClaimStatement(
        text="Acme is active", subject="Acme", predicate="status", object="active"
    )

    assert evidence(statement, "source-a", "report one").fingerprint != evidence(
        statement, "source-b", "report one"
    ).fingerprint


def test_evidence_fingerprint_is_excerpt_sensitive() -> None:
    statement = ClaimStatement(
        text="Acme is active", subject="Acme", predicate="status", object="active"
    )

    assert evidence(statement, "source-a", "report one").fingerprint != evidence(
        statement, "source-a", "report two"
    ).fingerprint


def test_evidence_fingerprint_is_role_sensitive() -> None:
    statement = ClaimStatement(
        text="Acme is active", subject="Acme", predicate="status", object="active"
    )

    assert evidence(statement, "source-a", "report one").fingerprint != evidence(
        statement, "source-a", "report one", role=EvidenceRole.CONTRADICTING
    ).fingerprint


def test_evidence_fingerprint_ignores_agent_confidence() -> None:
    statement = ClaimStatement(
        text="Acme is active", subject="Acme", predicate="status", object="active"
    )

    low = evidence(statement, "source-a", "report one", confidence=0.3)
    high = evidence(statement, "source-a", "report one", confidence=0.95)

    assert low.fingerprint == high.fingerprint


def test_claim_identity_survives_serialization() -> None:
    statement = active_claim()

    restored = ClaimStatement.from_dict(statement.to_dict())

    assert restored.identity == statement.identity
    assert restored.claim_id == statement.claim_id
