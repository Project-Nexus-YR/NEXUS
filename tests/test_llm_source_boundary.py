"""Section 17: low-quality LLM sources cannot bypass the acquisition gates.

An LLM-synthesized observation reports its own ``source_quality``; that self
report is never trusted to carry a claim.  Low-quality LLM evidence collapses
to WEAK, cannot self-corroborate, cannot borrow a real document's credibility,
and an LLM-invented citation fails provenance resolution at the knowledge
boundary.
"""

from __future__ import annotations

import pytest

from nexus_knowledge.domain.source import Source, SourceKind
from nexus_runtime.investigation.candidate_claims import CandidateClaimExtractor
from nexus_runtime.investigation.evaluation import EvidenceEvaluator
from nexus_runtime.investigation.evidence import (
    AgentConclusion,
    ClaimStatement,
    Evidence,
    EvidenceSet,
    InvestigationResult,
    InvestigationResultState,
    ToolObservation,
)
from nexus_runtime.investigation.knowledge_update import KnowledgeUpdateIntegrator
from nexus_runtime.investigation.provenance import EvidenceProvenance
from nexus_runtime.investigation.verification import ClaimVerifier

SESSION = "session-llm-source"
INVESTIGATION = "investigation-llm-source"
TASK = "task-llm-source"
ATTEMPT = "attempt-llm-source"
RUN = "run-llm-source"

LLM_CLAIM_ID = "claim-llm-source"


def claim_statement() -> ClaimStatement:
    return ClaimStatement(
        text="Nebula is profitable",
        subject="Nebula",
        predicate="status",
        object="profitable",
        claim_id=LLM_CLAIM_ID,
    )


def _observation(
    observation_id: str,
    source_reference: str,
    source_quality: float,
    tool_name: str = "llm_synthesis",
    confidence: float = 0.99,
) -> ToolObservation:
    return ToolObservation(
        observation_id=observation_id,
        tool_name=tool_name,
        status="SUCCEEDED",
        input={},
        output={"text": "model states Nebula is profitable"},
        source_reference=source_reference,
        metadata={
            "source_id": f"source-{observation_id}",
            "document_id": f"document-{observation_id}",
            "chunk_id": f"chunk-{observation_id}",
            "source_reference": source_reference,
            "source_quality": source_quality,
        },
    )


def _result(observations: tuple[ToolObservation, ...]) -> InvestigationResult:
    conclusion = AgentConclusion(
        claim=claim_statement(),
        supporting_observation_ids=tuple(item.observation_id for item in observations),
        confidence=0.99,
        conclusion_id="conclusion-llm-source",
    )
    return InvestigationResult(
        session_id=SESSION,
        investigation_id=INVESTIGATION,
        task_id=TASK,
        attempt_id=ATTEMPT,
        run_id=RUN,
        state=InvestigationResultState.COMPLETED,
        evidence_set=EvidenceSet(session_id=SESSION, evidence=()),
        conclusions=(conclusion,),
        observations=observations,
    )


def _pipeline(observations: tuple[ToolObservation, ...]):
    extraction = CandidateClaimExtractor().extract(_result(observations))
    report = ClaimVerifier().verify(EvidenceEvaluator().evaluate(extraction.evidence_set))
    return extraction, report


def test_single_low_quality_llm_source_cannot_verify() -> None:
    observations = (_observation("obs-llm-one", "llm://consensus-one", source_quality=0.2),)

    _, report = _pipeline(observations)

    assert report.decisions[0].status.value == "insufficient_evidence"
    assert not report.decisions[0].eligible_for_update
    assert "no acceptable supporting evidence" in report.decisions[0].reasons


def test_low_quality_llm_sources_cannot_self_corroborate() -> None:
    observations = (
        _observation("obs-llm-two-a", "llm://consensus-a", source_quality=0.2),
        _observation("obs-llm-two-b", "llm://consensus-b", source_quality=0.2),
    )

    _, report = _pipeline(observations)

    assert report.decisions[0].status.value == "insufficient_evidence"
    assert not report.decisions[0].eligible_for_update


def test_low_quality_llm_source_cannot_borrow_a_real_document() -> None:
    observations = (
        _observation("obs-llm-parasite", "llm://consensus", source_quality=0.2),
        _observation(
            "obs-doc-real",
            "doc://audited-filing",
            source_quality=0.9,
            tool_name="search",
        ),
    )

    extraction, report = _pipeline(observations)
    evidence = {item.source: item for item in extraction.evidence_set.evidence}

    assert report.decisions[0].status.value == "insufficient_evidence"
    llm_evidence = evidence["llm://consensus"].evidence_id
    real_evidence = evidence["doc://audited-filing"].evidence_id
    assert llm_evidence not in report.decisions[0].supporting_evidence_ids
    assert real_evidence in report.decisions[0].supporting_evidence_ids
    assert len(report.decisions[0].supporting_evidence_ids) == 1
    assert not report.decisions[0].eligible_for_update


def test_sub_threshold_llm_quality_is_excluded_even_at_max_confidence() -> None:
    observations = (
        _observation("obs-llm-avg-a", "llm://avg-a", source_quality=0.45),
        _observation("obs-llm-avg-b", "llm://avg-b", source_quality=0.45),
    )

    _, report = _pipeline(observations)

    assert report.decisions[0].status.value == "insufficient_evidence"
    assert report.decisions[0].supporting_evidence_ids == ()
    assert not report.decisions[0].eligible_for_update
    assert "no acceptable supporting evidence" in report.decisions[0].reasons


def test_llm_source_is_deferred_not_submitted(ingested_engine) -> None:
    observations = (_observation("obs-llm-deferred", "llm://consensus", source_quality=0.2),)
    extraction, report = _pipeline(observations)
    count_before = ingested_engine.repository.claims.count()

    integrator = KnowledgeUpdateIntegrator(ingested_engine)

    assert report.eligible_claims == ()
    result = integrator.apply(integrator.prepare(report, extraction.evidence_set))
    assert result.committed_claim_ids == ()
    assert ingested_engine.repository.claims.count() == count_before


def test_llm_invented_citation_fails_provenance_resolution(ingested_engine) -> None:
    real = ingested_engine.ingest(
        Source("llm-real", SourceKind.TEXT, "llm://real-doc"),
        "The audited filing reports Nebula is profitable.",
    )
    evidence = EvidenceSet(
        session_id=SESSION,
        evidence=(
            _engine_evidence(real, "evidence-real", real.source.id, "doc-real", "chunk-real"),
            _engine_evidence(
                real,
                "evidence-invented",
                "src-llm-invented",
                "doc-llm-invented",
                "chunk-llm-invented",
            ),
        ),
    )
    report = ClaimVerifier().verify(EvidenceEvaluator().evaluate(evidence))
    assert report.eligible_claims

    integrator = KnowledgeUpdateIntegrator(ingested_engine)
    with pytest.raises(ValueError, match="does not resolve"):
        integrator.prepare(report, evidence)


def _engine_evidence(
    ingested,
    evidence_id: str,
    source_id: str,
    document_id: str,
    chunk_id: str,
) -> Evidence:
    provenance = EvidenceProvenance(
        session_id=SESSION,
        investigation_id=INVESTIGATION,
        task_id=TASK,
        attempt_id=ATTEMPT,
        run_id=RUN,
        tool_call_id=f"tool-{evidence_id}",
        source_id=source_id,
        document_id=document_id,
        chunk_id=chunk_id,
        source_reference="llm://invented",
    )
    return Evidence(
        investigation_id=INVESTIGATION,
        source="llm://invented",
        claim=claim_statement(),
        provenance=provenance,
        confidence=0.95,
        source_quality=0.9,
        excerpt=ingested.chunks[0].text,
        payload={},
        evidence_id=evidence_id,
    )
