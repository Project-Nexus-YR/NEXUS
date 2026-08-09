from __future__ import annotations

from dataclasses import replace

import pytest

from nexus_runtime.investigation.candidate_claims import (
    CandidateClaim,
    CandidateClaimExtractor,
)
from nexus_runtime.investigation.evidence import (
    AgentConclusion,
    ClaimStatement,
    EvidenceSet,
    InvestigationResult,
    InvestigationResultState,
    ToolObservation,
    _stable_id,
)
from nexus_runtime.investigation.knowledge_update import KnowledgeUpdateIntegrator
from nexus_runtime.investigation.provenance import EvidenceProvenance
from nexus_runtime.investigation.verification import (
    EpistemicStatus,
    VerificationDecision,
    VerificationReport,
)

SESSION = "session-g1"
INVESTIGATION = "investigation-g1"
TASK = "task-g1"
ATTEMPT = "attempt-g1"
RUN = "run-g1"


def claim(
    claim_id: str = "claim-g1",
    subject: str = "Acme",
    predicate: str = "status",
    object_value: str = "active",
) -> ClaimStatement:
    return ClaimStatement(
        text=f"{subject} is {object_value}",
        subject=subject,
        predicate=predicate,
        object=object_value,
        claim_id=claim_id,
    )


def observation(
    observation_id: str,
    source: str = "source-1",
    excerpt: str = "independent report",
    quality: float = 0.9,
) -> ToolObservation:
    return ToolObservation(
        observation_id=observation_id,
        tool_name="search",
        status="SUCCEEDED",
        input={"source": source},
        output={"excerpt": excerpt},
        source_reference=f"source://{source}",
        metadata={
            "source_id": f"source-{source}",
            "document_id": f"document-{source}",
            "chunk_id": f"chunk-{source}",
            "source_reference": f"source://{source}",
            "source_quality": quality,
        },
    )


def conclusion(
    observation_ids: tuple[str, ...],
    claim_statement: ClaimStatement | None = None,
    conclusion_id: str | None = None,
    confidence: float = 0.8,
) -> AgentConclusion:
    kwargs = {"conclusion_id": conclusion_id} if conclusion_id is not None else {}
    return AgentConclusion(
        claim=claim_statement or claim(),
        supporting_observation_ids=observation_ids,
        confidence=confidence,
        **kwargs,
    )


def result(
    observations: tuple[ToolObservation, ...],
    conclusions: tuple[AgentConclusion, ...] = (),
    *,
    session_id: str = SESSION,
    investigation_id: str = INVESTIGATION,
    task_id: str = TASK,
    attempt_id: str = ATTEMPT,
    run_id: str = RUN,
    metadata: dict | None = None,
) -> InvestigationResult:
    return InvestigationResult(
        session_id=session_id,
        investigation_id=investigation_id,
        task_id=task_id,
        attempt_id=attempt_id,
        run_id=run_id,
        state=InvestigationResultState.COMPLETED,
        evidence_set=EvidenceSet(session_id=session_id, evidence=()),
        conclusions=conclusions,
        observations=observations,
        final_answer="ok",
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Section 4: grounding invariants -- every candidate claim maps 1:1 to the
# observations it cites, and dangling references are rejected explicitly.
# ---------------------------------------------------------------------------


def test_valid_conclusion_produces_one_evidence_per_supporting_observation() -> None:
    observations = (observation("observation-a"), observation("observation-b"))
    investigation_result = result(observations, (conclusion(("observation-a", "observation-b")),))

    extraction = CandidateClaimExtractor().extract(investigation_result)

    assert len(extraction.candidates) == 1
    candidate = extraction.candidates[0]
    assert candidate.claim.claim_id == "claim-g1"
    expected = tuple(
        _stable_id("evidence", "claim-g1", observation_id)
        for observation_id in ("observation-a", "observation-b")
    )
    assert candidate.evidence_ids == expected
    assert {item.provenance.tool_call_id for item in candidate.evidence} == {
        "observation-a",
        "observation-b",
    }
    assert not extraction.diagnostics

    second_pass = CandidateClaimExtractor().extract(investigation_result)
    assert second_pass.candidates[0].evidence_ids == candidate.evidence_ids


def test_every_evidence_lineage_matches_the_result_correlation_chain() -> None:
    observations = (observation("observation-a"), observation("observation-b"))
    investigation_result = result(observations, (conclusion(("observation-a", "observation-b")),))

    extraction = CandidateClaimExtractor().extract(investigation_result)

    for item in extraction.evidence_set.evidence:
        lineage = item.provenance
        assert lineage.correlation_ids in (
            (SESSION, INVESTIGATION, TASK, ATTEMPT, RUN, "observation-a"),
            (SESSION, INVESTIGATION, TASK, ATTEMPT, RUN, "observation-b"),
        )
        assert item.investigation_id == INVESTIGATION
        assert item.source == lineage.source_reference
        source = lineage.source_id.split("-", 1)[1]
        assert lineage.source_id == f"source-{source}"
        assert lineage.document_id == f"document-{source}"
        assert lineage.chunk_id == f"chunk-{source}"


def test_evidence_carries_observation_excerpt_and_input_payload() -> None:
    observations = (
        observation("observation-a", excerpt="reported by regulator"),
    )
    investigation_result = result(observations, (conclusion(("observation-a",)),))

    item = CandidateClaimExtractor().extract(investigation_result).evidence_set.evidence[0]

    assert item.excerpt == "reported by regulator"
    assert item.payload == {"source": "source-1"}
    assert item.role.value == "supporting"


def test_conclusion_without_supporting_observations_is_rejected() -> None:
    investigation_result = result((), (conclusion(()),))

    extraction = CandidateClaimExtractor().extract(investigation_result)

    assert not extraction.candidates
    assert not extraction.evidence_set.evidence
    codes = {diagnostic.code for diagnostic in extraction.diagnostics}
    assert "no_supporting_observations" in codes
    message = next(
        diagnostic.message
        for diagnostic in extraction.diagnostics
        if diagnostic.code == "no_supporting_observations"
    )
    assert "does not reference any observation" in message


def test_unknown_observation_reference_is_rejected_and_named() -> None:
    investigation_result = result(
        (observation("observation-a"),),
        (conclusion(("observation-a", "observation-missing")),),
    )

    extraction = CandidateClaimExtractor().extract(investigation_result)

    assert not extraction.candidates
    codes = {diagnostic.code for diagnostic in extraction.diagnostics}
    assert "unknown_observation_reference" in codes
    message = next(
        diagnostic.message
        for diagnostic in extraction.diagnostics
        if diagnostic.code == "unknown_observation_reference"
    )
    assert "observation-missing" in message


def test_partially_grounded_conclusion_produces_no_partial_evidence() -> None:
    investigation_result = result(
        (observation("observation-a"),),
        (conclusion(("observation-a", "observation-missing")),),
    )

    extraction = CandidateClaimExtractor().extract(investigation_result)

    assert not extraction.candidates
    assert not extraction.evidence_set.evidence


def test_reference_to_observation_from_another_result_is_unknown() -> None:
    other_result = result(
        (observation("observation-other-run"),),
        (conclusion(("observation-other-run",)),),
        run_id="run-other",
    )
    assert other_result.observations[0].observation_id == "observation-other-run"

    investigation_result = result(
        (observation("observation-a"),),
        (conclusion(("observation-other-run",)),),
    )

    extraction = CandidateClaimExtractor().extract(investigation_result)
    assert not extraction.candidates
    assert any(
        diagnostic.code == "unknown_observation_reference"
        and "observation-other-run" in diagnostic.message
        for diagnostic in extraction.diagnostics
    )


def test_reference_to_observation_from_another_investigation_is_unknown() -> None:
    investigation_result = result(
        (observation("observation-a"),),
        (conclusion(("observation-other-investigation",)),),
        investigation_id=INVESTIGATION,
    )

    extraction = CandidateClaimExtractor().extract(investigation_result)

    assert not extraction.candidates
    assert any(
        diagnostic.code == "unknown_observation_reference"
        and "observation-other-investigation" in diagnostic.message
        for diagnostic in extraction.diagnostics
    )


def test_malformed_conclusions_surface_as_diagnostics() -> None:
    investigation_result = result(
        (observation("observation-a"),),
        (conclusion(("observation-a",)),),
        metadata={
            "malformed_conclusions": [
                {"conclusion_id": "conclusion-broken", "reason": "claim not an object"}
            ]
        },
    )

    extraction = CandidateClaimExtractor().extract(investigation_result)

    assert len(extraction.candidates) == 1
    assert any(
        diagnostic.code == "malformed_conclusion"
        and diagnostic.conclusion_id == "conclusion-broken"
        for diagnostic in extraction.diagnostics
    )


def _unchecked_result(
    observations: tuple[ToolObservation, ...],
    conclusions: tuple[AgentConclusion, ...],
    **kwargs: object,
) -> InvestigationResult:
    payload = {
        "session_id": SESSION,
        "investigation_id": INVESTIGATION,
        "task_id": TASK,
        "attempt_id": ATTEMPT,
        "run_id": RUN,
        "state": InvestigationResultState.COMPLETED,
        "evidence_set": EvidenceSet(session_id=SESSION, evidence=()),
        "error": None,
        "final_answer": "ok",
        "conclusions": conclusions,
        "observations": observations,
        "metadata": {},
    }
    payload.update(kwargs)
    instance = object.__new__(InvestigationResult)
    for name, value in payload.items():
        object.__setattr__(instance, name, value)
    return instance


def test_result_rejects_duplicate_conclusion_ids_at_construction() -> None:
    with pytest.raises(ValueError, match="conclusion_id values must be unique"):
        result(
            (observation("observation-a"),),
            (
                conclusion(("observation-a",), conclusion_id="conclusion-duplicate"),
                conclusion(("observation-a",), conclusion_id="conclusion-duplicate"),
            ),
        )


def test_duplicate_conclusion_deduped_with_recovered_diagnostic() -> None:
    investigation_result = _unchecked_result(
        (observation("observation-a"),),
        (
            conclusion(("observation-a",), conclusion_id="conclusion-duplicate"),
            conclusion(("observation-a",), conclusion_id="conclusion-duplicate"),
        ),
    )

    extraction = CandidateClaimExtractor().extract(investigation_result)

    assert len(extraction.candidates) == 1
    assert any(
        diagnostic.code == "duplicate_conclusion"
        and diagnostic.conclusion_id == "conclusion-duplicate"
        and diagnostic.recovered
        for diagnostic in extraction.diagnostics
    )


def test_same_candidate_from_two_conclusions_is_deduped() -> None:
    investigation_result = result(
        (observation("observation-a"),),
        (
            conclusion(("observation-a",), conclusion_id="conclusion-one"),
            conclusion(("observation-a",), conclusion_id="conclusion-two"),
        ),
    )

    extraction = CandidateClaimExtractor().extract(investigation_result)

    assert len(extraction.candidates) == 1
    assert any(
        diagnostic.code == "duplicate_candidate" and diagnostic.recovered
        for diagnostic in extraction.diagnostics
    )


def test_extraction_does_not_mutate_the_input_result() -> None:
    observations = (observation("observation-a"),)
    investigation_result = result(observations, (conclusion(("observation-a",)),))

    extraction = CandidateClaimExtractor().extract(investigation_result)

    assert investigation_result.evidence_set.evidence == ()
    assert extraction.evidence_set.session_id == investigation_result.session_id
    assert len(extraction.evidence_set.evidence) == 1


def test_candidate_claim_requires_supporting_evidence() -> None:
    with pytest.raises(ValueError, match="requires supporting evidence"):
        CandidateClaim(
            claim=claim(),
            evidence=(),
            conclusion_id="conclusion-g1",
            candidate_id="candidate-g1",
            confidence=0.8,
        )


# ---------------------------------------------------------------------------
# Section 5: provenance completeness -- the objective-to-source lineage must
# survive serialization and every component is enforced end to end.
# ---------------------------------------------------------------------------


def test_provenance_survives_round_trip_preserving_full_lineage() -> None:
    investigation_result = result(
        (observation("observation-a"),),
        (conclusion(("observation-a",)),),
    )

    extraction = CandidateClaimExtractor().extract(investigation_result)
    restored = extraction.from_dict(extraction.to_dict())

    assert restored.candidates[0].evidence_ids == extraction.candidates[0].evidence_ids
    for original, recovered in zip(
        extraction.evidence_set.evidence, restored.evidence_set.evidence, strict=True
    ):
        assert recovered.provenance.correlation_ids == original.provenance.correlation_ids
        assert recovered.provenance.to_dict() == original.provenance.to_dict()


def test_provenance_rejects_any_empty_component() -> None:
    valid = {
        "session_id": SESSION,
        "investigation_id": INVESTIGATION,
        "task_id": TASK,
        "attempt_id": ATTEMPT,
        "run_id": RUN,
        "tool_call_id": "observation-a",
        "source_id": "source-1",
        "document_id": "document-1",
        "chunk_id": "chunk-1",
        "source_reference": "source://source-1",
    }
    for component in ("session_id", "tool_call_id", "chunk_id", "source_reference"):
        corrupted = dict(valid)
        corrupted[component] = "   "
        with pytest.raises(ValueError, match=component):
            EvidenceProvenance(**corrupted)


def test_provenance_from_dict_rejects_missing_component() -> None:
    payload = {
        "session_id": SESSION,
        "investigation_id": INVESTIGATION,
        "task_id": TASK,
        "attempt_id": ATTEMPT,
        "run_id": RUN,
        "tool_call_id": "observation-a",
        "source_id": "source-1",
        "document_id": "document-1",
        "chunk_id": "chunk-1",
        "source_reference": "source://source-1",
    }
    del payload["run_id"]
    with pytest.raises(ValueError, match="malformed evidence provenance field: run_id"):
        EvidenceProvenance.from_dict(payload)


def test_provenance_is_complete_for_valid_evidence() -> None:
    extraction = CandidateClaimExtractor().extract(
        result((observation("observation-a"),), (conclusion(("observation-a",)),))
    )
    item = extraction.evidence_set.evidence[0]

    assert item.provenance.is_complete
    assert len(item.provenance.correlation_ids) == 6


def test_evidence_rejects_mismatched_investigation_id() -> None:
    extraction = CandidateClaimExtractor().extract(
        result((observation("observation-a"),), (conclusion(("observation-a",)),))
    )
    evidence = extraction.evidence_set.evidence[0]

    with pytest.raises(ValueError, match="does not match provenance"):
        replace(evidence, investigation_id="investigation-other")


def test_evidence_rejects_mismatched_source_reference() -> None:
    extraction = CandidateClaimExtractor().extract(
        result((observation("observation-a"),), (conclusion(("observation-a",)),))
    )
    evidence = extraction.evidence_set.evidence[0]

    with pytest.raises(ValueError, match="does not match provenance"):
        replace(evidence, source="source://unrelated")


def test_result_rejects_evidence_with_mismatched_run_lineage() -> None:
    extraction = CandidateClaimExtractor().extract(
        result((observation("observation-a"),), (conclusion(("observation-a",)),))
    )
    misplaced = replace(
        extraction.evidence_set.evidence[0],
        provenance=replace(
            extraction.evidence_set.evidence[0].provenance,
            run_id="run-other",
        ),
    )

    with pytest.raises(ValueError, match="mismatched result lineage"):
        InvestigationResult(
            session_id=SESSION,
            investigation_id=INVESTIGATION,
            task_id=TASK,
            attempt_id=ATTEMPT,
            run_id=RUN,
            state=InvestigationResultState.COMPLETED,
            evidence_set=EvidenceSet(session_id=SESSION, evidence=(misplaced,)),
            observations=(observation("observation-a"),),
            conclusions=(conclusion(("observation-a",)),),
        )


def test_evidence_set_rejects_evidence_from_another_session() -> None:
    extraction = CandidateClaimExtractor().extract(
        result((observation("observation-a"),), (conclusion(("observation-a",)),))
    )
    foreign = replace(
        extraction.evidence_set.evidence[0],
        provenance=replace(
            extraction.evidence_set.evidence[0].provenance,
            session_id="session-other",
        ),
    )

    with pytest.raises(ValueError, match="evidence belongs to another session"):
        EvidenceSet(session_id=SESSION, evidence=(foreign,))


def _verification_report(evidence_ids: tuple[str, ...]) -> VerificationReport:
    decision = VerificationDecision(
        claim=claim(),
        status=EpistemicStatus.CONFIRMED,
        eligible_for_update=True,
        confidence=0.9,
        supporting_evidence_ids=evidence_ids,
        contradicting_evidence_ids=(),
        unresolved_conflict_ids=(),
        reasons=("verification criteria satisfied",),
    )
    return VerificationReport(
        session_id=SESSION,
        evaluation_id="evaluation-g1",
        decisions=(decision,),
    )


class _ResolvingKnowledge:
    def __init__(self, resolve: bool = True) -> None:
        self.resolve = resolve

    def validate_evidence_provenance(
        self,
        source_id: str,
        document_id: str,
        chunk_id: str,
        source_reference: str,
    ) -> bool:
        return self.resolve


class _IncompleteProvenance:
    is_complete = False
    session_id = SESSION
    investigation_id = INVESTIGATION
    task_id = TASK
    attempt_id = ATTEMPT
    run_id = RUN
    tool_call_id = "observation-a"
    source_id = "source-1"
    document_id = "document-1"
    chunk_id = "chunk-1"
    source_reference = "source://source-1"


def test_prepare_gate_blocks_incomplete_provenance() -> None:
    extraction = CandidateClaimExtractor().extract(
        result((observation("observation-a"),), (conclusion(("observation-a",)),))
    )
    evidence = extraction.evidence_set.evidence[0]
    corrupted_evidence = replace(evidence, provenance=_IncompleteProvenance())
    corrupted_set = EvidenceSet(session_id=SESSION, evidence=(corrupted_evidence,))

    report = _verification_report((evidence.evidence_id,))
    with pytest.raises(ValueError, match="complete evidence provenance"):
        KnowledgeUpdateIntegrator(_ResolvingKnowledge()).prepare(report, corrupted_set)


def test_prepare_gate_requires_resolvable_provenance() -> None:
    extraction = CandidateClaimExtractor().extract(
        result((observation("observation-a"),), (conclusion(("observation-a",)),))
    )
    evidence = extraction.evidence_set.evidence[0]
    evidence_set = EvidenceSet(session_id=SESSION, evidence=(evidence,))

    report = _verification_report((evidence.evidence_id,))
    with pytest.raises(ValueError, match="provenance does not resolve"):
        KnowledgeUpdateIntegrator(_ResolvingKnowledge(resolve=False)).prepare(report, evidence_set)


def test_prepare_gate_accepts_complete_resolvable_provenance() -> None:
    extraction = CandidateClaimExtractor().extract(
        result((observation("observation-a"),), (conclusion(("observation-a",)),))
    )
    evidence = extraction.evidence_set.evidence[0]
    evidence_set = EvidenceSet(session_id=SESSION, evidence=(evidence,))

    report = _verification_report((evidence.evidence_id,))
    submission = KnowledgeUpdateIntegrator(_ResolvingKnowledge()).prepare(report, evidence_set)

    assert submission.evidence_ids == (evidence.evidence_id,)
    chain = (SESSION, INVESTIGATION, TASK, ATTEMPT, RUN, "observation-a")
    assert submission.provenance[0].correlation_ids == chain
