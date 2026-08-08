"""Focused evidence, verification, knowledge-update, and progress scenarios."""

from __future__ import annotations

from dataclasses import replace

import pytest

from nexus_knowledge.domain.claim import Claim
from nexus_knowledge.domain.common import VerificationState
from nexus_knowledge.domain.source import Source, SourceKind
from nexus_knowledge.service.engine import KnowledgeUpdate
from nexus_runtime.investigation.evaluation import (
    EvidenceEvaluator,
    EvidenceQualityPolicy,
)
from nexus_runtime.investigation.evidence import (
    ClaimStatement,
    Evidence,
    EvidenceRole,
    EvidenceSet,
    InvestigationResult,
    InvestigationResultState,
)
from nexus_runtime.investigation.knowledge_update import KnowledgeUpdateIntegrator
from nexus_runtime.investigation.progress import GapState, ProgressMeasurer
from nexus_runtime.investigation.provenance import EvidenceProvenance
from nexus_runtime.investigation.verification import (
    ClaimVerifier,
    EpistemicStatus,
    VerificationPolicy,
)


def _provenance(
    *,
    source_id: str = "source-a",
    source_reference: str = "https://example.test/a",
    investigation_id: str = "investigation-a",
    document_id: str = "document-a",
    chunk_id: str = "chunk-a",
) -> EvidenceProvenance:
    return EvidenceProvenance(
        session_id="session-1",
        investigation_id=investigation_id,
        task_id="task-1",
        attempt_id="attempt-1",
        run_id="run-1",
        tool_call_id=f"tool-{source_id}",
        source_id=source_id,
        document_id=document_id,
        chunk_id=chunk_id,
        source_reference=source_reference,
    )


def _evidence(
    *,
    claim: ClaimStatement | None = None,
    source_id: str = "source-a",
    source_reference: str = "https://example.test/a",
    excerpt: str = "The primary record states London.",
    confidence: float = 0.95,
    source_quality: float = 0.9,
    role: EvidenceRole = EvidenceRole.SUPPORTING,
    evidence_id: str | None = None,
    document_id: str = "document-a",
    chunk_id: str = "chunk-a",
) -> Evidence:
    kwargs = {}
    if evidence_id is not None:
        kwargs["evidence_id"] = evidence_id
    return Evidence(
        investigation_id="investigation-a",
        source=source_reference,
        claim=claim
        or ClaimStatement(
            text="Atlas is headquartered in London",
            subject="Atlas",
            predicate="headquartered_in",
            object="London",
        ),
        provenance=_provenance(
            source_id=source_id,
            source_reference=source_reference,
            document_id=document_id,
            chunk_id=chunk_id,
        ),
        confidence=confidence,
        excerpt=excerpt,
        source_quality=source_quality,
        role=role,
        **kwargs,
    )


class TestEvidenceContracts:
    def test_provenance_captures_complete_lineage(self):
        provenance = _provenance()
        assert provenance.is_complete
        assert provenance.correlation_ids == (
            "session-1",
            "investigation-a",
            "task-1",
            "attempt-1",
            "run-1",
            "tool-source-a",
        )
        assert provenance.to_dict()["chunk_id"] == "chunk-a"

    def test_incomplete_provenance_is_rejected(self):
        with pytest.raises(ValueError, match="tool_call_id"):
            replace(_provenance(), tool_call_id="")

    def test_anonymous_evidence_is_rejected(self):
        with pytest.raises(ValueError, match="excerpt or payload"):
            replace(_evidence(), excerpt="", payload={})

    def test_result_requires_matching_runtime_correlation_ids(self):
        evidence_set = EvidenceSet(session_id="session-1", evidence=(_evidence(),))
        result = InvestigationResult(
            session_id="session-1",
            investigation_id="investigation-a",
            task_id="task-1",
            attempt_id="attempt-1",
            run_id="run-1",
            state=InvestigationResultState.COMPLETED,
            evidence_set=evidence_set,
        )
        assert result.to_dict()["run_id"] == "run-1"
        with pytest.raises(ValueError, match="mismatched result lineage"):
            replace(result, run_id="another-run")


class TestEvidenceEvaluation:
    def test_independent_support_is_fused_without_being_deduplicated(self):
        claim = ClaimStatement(
            text="Atlas is headquartered in London",
            subject="Atlas",
            predicate="headquartered_in",
            object="London",
        )
        evidence_set = EvidenceSet(
            session_id="session-1",
            evidence=(
                _evidence(claim=claim, evidence_id="evidence-a"),
                _evidence(
                    claim=claim,
                    source_id="source-b",
                    source_reference="https://example.test/b",
                    excerpt="The registry lists London.",
                    evidence_id="evidence-b",
                    document_id="document-b",
                    chunk_id="chunk-b",
                ),
            ),
        )
        evaluation = EvidenceEvaluator().evaluate(evidence_set)
        assert len(evaluation.claims) == 1
        assert evaluation.claims[0].independent_source_count == 2
        assert evaluation.claims[0].aggregate_confidence > 0.8
        assert evaluation.duplicate_evidence_ids == ()

    def test_duplicate_and_low_quality_evidence_are_classified(self):
        claim = ClaimStatement("Atlas in London", "Atlas", "located_in", "London")
        original = _evidence(claim=claim, evidence_id="evidence-original")
        duplicate = _evidence(claim=claim, evidence_id="evidence-duplicate")
        weak = _evidence(
            claim=claim,
            source_id="weak",
            source_reference="https://example.test/weak",
            excerpt="An unattributed post says London.",
            confidence=0.2,
            source_quality=0.2,
            evidence_id="evidence-weak",
        )
        evaluation = EvidenceEvaluator().evaluate(
            EvidenceSet(session_id="session-1", evidence=(original, duplicate, weak))
        )
        assert evaluation.duplicate_evidence_ids == ("evidence-duplicate",)
        assert evaluation.low_quality_evidence_ids == ("evidence-weak",)
        assert evaluation.accepted_evidence_count == 1

    def test_conflicting_claims_preserve_both_sides(self):
        london = ClaimStatement("Atlas is in London", "Atlas", "located_in", "London")
        paris = ClaimStatement("Atlas is in Paris", "Atlas", "located_in", "Paris")
        evaluation = EvidenceEvaluator().evaluate(
            EvidenceSet(
                session_id="session-1",
                evidence=(
                    _evidence(claim=london, evidence_id="evidence-london"),
                    _evidence(
                        claim=paris,
                        source_id="source-b",
                        source_reference="https://example.test/b",
                        excerpt="A second registry lists Paris.",
                        evidence_id="evidence-paris",
                    ),
                ),
            )
        )
        assert len(evaluation.conflict_ids) == 1
        assert {claim.claim.object for claim in evaluation.claims} == {"London", "Paris"}
        assert all(claim.unresolved_contradiction for claim in evaluation.claims)


class TestVerification:
    def test_sufficient_independent_evidence_is_confirmed(self):
        claim = ClaimStatement("Atlas in London", "Atlas", "located_in", "London")
        evidence = (
            _evidence(claim=claim, evidence_id="evidence-a"),
            _evidence(
                claim=claim,
                source_id="source-b",
                source_reference="https://example.test/b",
                excerpt="Independent registry record.",
                evidence_id="evidence-b",
            ),
        )
        report = ClaimVerifier().verify(
            EvidenceEvaluator().evaluate(EvidenceSet("session-1", evidence))
        )
        assert report.decisions[0].status == EpistemicStatus.CONFIRMED
        assert report.decisions[0].eligible_for_update

    def test_single_source_is_insufficient_by_default(self):
        report = ClaimVerifier().verify(
            EvidenceEvaluator().evaluate(EvidenceSet("session-1", (_evidence(),)))
        )
        assert report.decisions[0].status == EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert not report.decisions[0].eligible_for_update

    def test_unresolved_conflict_never_uses_agent_confidence_as_tiebreaker(self):
        london = ClaimStatement("Atlas in London", "Atlas", "located_in", "London")
        paris = ClaimStatement("Atlas in Paris", "Atlas", "located_in", "Paris")
        evidence = (
            _evidence(claim=london, confidence=0.99),
            _evidence(
                claim=paris,
                source_id="source-b",
                source_reference="https://example.test/b",
                excerpt="Paris registry",
                confidence=0.6,
            ),
        )
        report = ClaimVerifier(VerificationPolicy(min_independent_sources=1)).verify(
            EvidenceEvaluator().evaluate(EvidenceSet("session-1", evidence))
        )
        assert {decision.status for decision in report.decisions} == {EpistemicStatus.CONTRADICTED}
        assert not report.eligible_claims

    def test_quality_threshold_is_configurable(self):
        evaluation = EvidenceEvaluator(
            EvidenceQualityPolicy(min_evidence_confidence=0.9, min_source_quality=0.9)
        ).evaluate(EvidenceSet("session-1", (_evidence(confidence=0.8),)))
        assert evaluation.low_quality_evidence_ids


class TestKnowledgeUpdate:
    def _verified_set(self, ingested_engine):
        first = ingested_engine.ingest(
            Source("registry-a", SourceKind.TEXT, "registry://a"),
            "Atlas has a registered office in London.",
        )
        second = ingested_engine.ingest(
            Source("registry-b", SourceKind.TEXT, "registry://b"),
            "Atlas maintains its headquarters in London.",
        )
        claim = ClaimStatement("Atlas in London", "Atlas", "located_in", "London")
        evidence = (
            _evidence(
                claim=claim,
                source_id=first.source.id,
                source_reference=first.source.reference,
                excerpt=first.chunks[0].text,
                evidence_id="investigation-evidence-a",
                document_id=first.documents[0].id,
                chunk_id=first.chunks[0].id,
            ),
            _evidence(
                claim=claim,
                source_id=second.source.id,
                source_reference=second.source.reference,
                excerpt=second.chunks[0].text,
                evidence_id="investigation-evidence-b",
                document_id=second.documents[0].id,
                chunk_id=second.chunks[0].id,
            ),
        )
        return claim, EvidenceSet("session-1", evidence)

    def test_valid_update_uses_public_engine_and_preserves_provenance(self, ingested_engine):
        claim, evidence_set = self._verified_set(ingested_engine)
        report = ClaimVerifier().verify(EvidenceEvaluator().evaluate(evidence_set))
        integrator = KnowledgeUpdateIntegrator(ingested_engine)
        submission = integrator.prepare(report, evidence_set)
        result = integrator.apply(submission)

        assert result.fully_applied
        assert result.verification_states[claim.claim_id] == VerificationState.VERIFIED.value
        stored = ingested_engine.repository.claims.get(claim.claim_id)
        assert stored is not None
        assert stored.metadata["investigation_session_id"] == "session-1"
        assert len(stored.metadata["evidence_lineage"]) == 2
        assert set(stored.supporting_evidence) == {
            "investigation-evidence-a",
            "investigation-evidence-b",
        }

    def test_invalid_update_is_not_submitted(self, ingested_engine):
        evidence_set = EvidenceSet("session-1", (_evidence(),))
        report = ClaimVerifier().verify(EvidenceEvaluator().evaluate(evidence_set))
        integrator = KnowledgeUpdateIntegrator(ingested_engine)
        submission = integrator.prepare(report, evidence_set)
        before = ingested_engine.repository.claims.count()
        result = integrator.apply(submission)
        assert result.accepted_records == 0
        assert ingested_engine.repository.claims.count() == before

    def test_existing_contradiction_is_preserved_and_not_auto_verified(self, ingested_engine):
        claim, evidence_set = self._verified_set(ingested_engine)
        existing = Claim(
            text="Atlas in Paris",
            subject="Atlas",
            predicate="located_in",
            object="Paris",
            id="existing-paris",
        )
        ingested_engine.commit_knowledge_update(KnowledgeUpdate(claims=[existing]))
        report = ClaimVerifier().verify(EvidenceEvaluator().evaluate(evidence_set))
        integrator = KnowledgeUpdateIntegrator(ingested_engine)
        result = integrator.apply(integrator.prepare(report, evidence_set))

        assert result.unresolved_contradiction_ids
        assert result.verification_states[claim.claim_id] == VerificationState.CONTRADICTED.value
        stored = ingested_engine.repository.claims.get(claim.claim_id)
        assert stored is not None
        assert stored.verification_state == VerificationState.UNVERIFIED
        assert stored.metadata["evidence_lineage"]


class TestProgress:
    def test_measures_resolved_new_gaps_uncertainty_and_cost(self):
        report = ProgressMeasurer().measure(
            session_id="session-1",
            iteration=2,
            before_gaps=(GapState("gap-a", 0.8), GapState("gap-b", 0.6)),
            after_gaps=(GapState("gap-b", 0.2), GapState("gap-c", 0.1)),
            before_contradiction_ids=("conflict-old",),
            after_contradiction_ids=("conflict-new",),
            evidence_collected=4,
            knowledge_updates=1,
            cost=6.0,
        )
        assert report.resolved_gap_ids == ("gap-a",)
        assert report.new_gap_ids == ("gap-c",)
        assert report.uncertainty_reduced == pytest.approx(1.2)
        assert report.contradictions_resolved == ("conflict-old",)
        assert report.contradictions_introduced == ("conflict-new",)
        assert report.cost_per_resolved_gap == 6.0
        assert report.information_gain > 0

    def test_no_resolved_gap_has_no_cost_ratio(self):
        report = ProgressMeasurer().measure(
            session_id="session-1",
            iteration=1,
            before_gaps=(GapState("gap-a", 0.8),),
            after_gaps=(GapState("gap-a", 0.7),),
            cost=2.0,
        )
        assert report.cost_per_resolved_gap is None
        assert report.information_gain == pytest.approx(0.1)
