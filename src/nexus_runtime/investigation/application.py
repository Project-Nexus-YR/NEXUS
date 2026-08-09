"""Explicit application services for the closed-loop investigation lifecycle."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol, cast

from nexus_runtime.events import Event, EventBus, InMemoryEventBus
from nexus_runtime.models import DomainError, utcnow

from .acquisition import AcquisitionReport, ClaimAcquisitionService
from .candidate_claims import (
    CandidateClaim,
    CandidateClaimExtractor,
    CandidateExtractionResult,
    ExtractionDiagnostic,
)
from .evaluation import Evaluation, EvidenceEvaluator
from .evidence import EvidenceSet, InvestigationResult, InvestigationResultState
from .execution import ExecutionStatus, PlanExecution, PlanExecutionController
from .generator import CandidateInvestigation, InvestigationGenerator, KnowledgeGapLike
from .knowledge_update import (
    InvestigationKnowledgeUpdate,
    KnowledgeUpdateIntegrator,
    KnowledgeUpdatePort,
    KnowledgeUpdateResult,
)
from .metrics import InMemoryInvestigationMetrics, InvestigationMetrics
from .objective import ResearchObjective
from .observation import KnowledgeSnapshot
from .planner import InvestigationPlan, InvestigationPlanner
from .progress import GapState, ProgressMeasurer, ProgressReport
from .repository import (
    InMemoryInvestigationRepository,
    InvestigationArtifact,
    InvestigationRecord,
    InvestigationRepository,
)
from .results import RuntimeResultCollector
from .scoring import InvestigationScore, InvestigationScoringModel
from .selector import InvestigationSelector, SelectionResult
from .session import InvestigationBudget, InvestigationSession, SessionState, TerminationReason
from .termination import TerminationContext, TerminationDecision, TerminationPolicy
from .verification import ClaimVerifier, VerificationReport


class KnowledgeObservationPort(Protocol):
    """Read-only public knowledge operations used to construct a snapshot."""

    def retrieve(self, query: str, top_k: int = 10) -> Any: ...

    def graphrag(self, query: str, top_k: int = 8, depth: int = 2) -> Any: ...

    def find_knowledge_gaps(self) -> list[Any]: ...

    def detect_contradictions(self) -> list[Any]: ...


class KnowledgeObserver:
    """Observe retrieval, GraphRAG, uncertainty gaps, and contradictions via one boundary."""

    def __init__(self, knowledge: KnowledgeObservationPort) -> None:
        self._knowledge = knowledge

    def observe(
        self, objective: ResearchObjective, *, at: datetime | None = None
    ) -> KnowledgeSnapshot:
        retrieval = self._knowledge.retrieve(objective.question, top_k=10)
        evidence_graph = self._knowledge.graphrag(objective.question, top_k=8, depth=2)
        gaps = cast("Sequence[KnowledgeGapLike]", self._knowledge.find_knowledge_gaps())
        contradictions = self._knowledge.detect_contradictions()
        retrieval_refs = tuple(
            str(getattr(item, "chunk_id", ""))
            for item in getattr(retrieval, "candidates", ())
            if getattr(item, "chunk_id", "")
        )
        entity_ids = tuple(
            str(getattr(item, "id", ""))
            for item in getattr(evidence_graph, "entities", ())
            if getattr(item, "id", "")
        )
        relation_ids = tuple(
            str(getattr(item, "id", ""))
            for item in getattr(evidence_graph, "relations", ())
            if getattr(item, "id", "")
        )
        contradiction_keys = tuple(self._contradiction_key(item) for item in contradictions)
        confidence = float(getattr(evidence_graph, "confidence", 0.0))
        return KnowledgeSnapshot.capture(
            objective,
            gaps,
            observed_at=at or utcnow(),
            retrieval_refs=retrieval_refs,
            entity_ids=entity_ids,
            relation_ids=relation_ids,
            contradiction_ids=contradiction_keys,
            summary=(
                f"{len(gaps)} gaps, {len(contradictions)} contradictions, "
                f"GraphRAG confidence {confidence:.3f}"
            ),
            metadata={
                "retrieval_request_id": str(getattr(retrieval, "request_id", "")),
                "graphrag_confidence": confidence,
            },
        )

    @staticmethod
    def _contradiction_key(contradiction: Any) -> str:
        kind = str(getattr(contradiction, "kind", "contradiction"))
        pair = sorted(
            (
                str(getattr(contradiction, "claim_a_id", "unknown-a")),
                str(getattr(contradiction, "claim_b_id", "unknown-b")),
            )
        )
        return f"{kind}:{pair[0]}:{pair[1]}"


@dataclass(frozen=True, slots=True)
class PlanningOutcome:
    snapshot: KnowledgeSnapshot
    candidates: tuple[CandidateInvestigation, ...]
    scores: tuple[InvestigationScore, ...]
    selection: SelectionResult
    plan: InvestigationPlan | None
    termination: TerminationDecision | None = None


@dataclass(frozen=True, slots=True)
class IterationOutcome:
    evidence_set: EvidenceSet
    evaluation: Evaluation
    verification: VerificationReport
    knowledge_update: InvestigationKnowledgeUpdate
    update_result: KnowledgeUpdateResult
    progress: ProgressReport
    termination: TerminationDecision


class InvestigationApplication:
    """Transport-independent API; each epistemic stage remains separately callable."""

    def __init__(
        self,
        knowledge: KnowledgeObservationPort,
        *,
        knowledge_updates: KnowledgeUpdatePort | None = None,
        repository: InvestigationRepository | None = None,
        execution: PlanExecutionController | None = None,
        result_collector: RuntimeResultCollector | None = None,
        event_bus: EventBus | None = None,
        metrics: InvestigationMetrics | None = None,
        generator: InvestigationGenerator | None = None,
        scoring: InvestigationScoringModel | None = None,
        selector: InvestigationSelector | None = None,
        planner: InvestigationPlanner | None = None,
        evaluator: EvidenceEvaluator | None = None,
        verifier: ClaimVerifier | None = None,
        termination: TerminationPolicy | None = None,
        extractor: CandidateClaimExtractor | None = None,
        acquisition: ClaimAcquisitionService | None = None,
    ) -> None:
        self._observer = KnowledgeObserver(knowledge)
        self._repository = repository or InMemoryInvestigationRepository()
        self._execution = execution
        self._result_collector = result_collector
        self._events = event_bus or InMemoryEventBus()
        self._metrics = metrics or InMemoryInvestigationMetrics()
        self._generator = generator or InvestigationGenerator()
        self._scoring = scoring or InvestigationScoringModel()
        self._selector = selector or InvestigationSelector(minimum_score=0.05)
        self._planner = planner or InvestigationPlanner()
        self._evaluator = evaluator or EvidenceEvaluator()
        self._verifier = verifier or ClaimVerifier()
        self._termination = termination or TerminationPolicy()
        update_boundary = knowledge_updates
        if update_boundary is None:
            update_boundary = cast(KnowledgeUpdatePort, knowledge)
        self._updates = KnowledgeUpdateIntegrator(update_boundary)
        self._extractor = extractor or CandidateClaimExtractor()
        self._acquisition = acquisition or ClaimAcquisitionService()

    def create(
        self, objective: ResearchObjective, budget: InvestigationBudget
    ) -> InvestigationSession:
        session = InvestigationSession(objective_id=objective.objective_id, budget=budget)
        record = InvestigationRecord(objective, session)
        self._repository.save(record)
        self._metrics.increment("investigation_sessions")
        self._emit("investigation.session_created", record, {"question": objective.question})
        return session

    def status(self, session_id: str) -> InvestigationRecord:
        return self._record(session_id)

    def observe(self, session_id: str) -> KnowledgeSnapshot:
        record = self._record(session_id)
        if record.session.state not in {SessionState.PLANNING, SessionState.UPDATING}:
            raise DomainError("knowledge can be observed only while PLANNING or UPDATING")
        self._emit("investigation.planning_started", record, {})
        snapshot = self._observer.observe(record.objective)
        record.append("knowledge_snapshot", snapshot.to_dict())
        self._repository.save(record)
        self._metrics.increment("gaps_discovered", len(snapshot.gaps))
        self._emit(
            "investigation.gaps_identified",
            record,
            {"snapshot_id": snapshot.snapshot_id, "count": len(snapshot.gaps)},
        )
        return snapshot

    def generate(
        self, session_id: str, snapshot: KnowledgeSnapshot
    ) -> tuple[CandidateInvestigation, ...]:
        record = self._record(session_id)
        self._require_snapshot(record, snapshot)
        candidates = self._generator.generate(record.objective, snapshot.gaps)
        record.append(
            "candidate_investigations", {"items": [item.to_dict() for item in candidates]}
        )
        self._repository.save(record)
        self._metrics.increment("investigations_generated", len(candidates))
        self._emit("investigation.candidates_generated", record, {"count": len(candidates)})
        return candidates

    def score(
        self,
        session_id: str,
        snapshot: KnowledgeSnapshot,
        candidates: Sequence[CandidateInvestigation],
    ) -> tuple[InvestigationScore, ...]:
        record = self._record(session_id)
        self._require_snapshot(record, snapshot)
        candidate_artifact = self._required_artifact(record, "candidate_investigations")
        if (
            candidate_artifact.iteration != record.session.iteration
            or candidate_artifact.payload.get("items") != [item.to_dict() for item in candidates]
        ):
            raise DomainError("candidate investigations are not active for this iteration")
        scores = self._scoring.score_all(candidates, snapshot.gaps)
        forecast = self._scoring.forecast(scores)
        record.append(
            "investigation_scores",
            {
                "items": [item.to_dict() for item in scores],
                "forecast": forecast.to_dict(),
            },
        )
        self._repository.save(record)
        return scores

    def select(
        self,
        session_id: str,
        scores: Sequence[InvestigationScore],
        *,
        worker_capacity: int,
        top_k: int | None = None,
    ) -> SelectionResult:
        record = self._record(session_id)
        score_artifact = self._required_artifact(record, "investigation_scores")
        if score_artifact.iteration != record.session.iteration or score_artifact.payload.get(
            "items"
        ) != [item.to_dict() for item in scores]:
            raise DomainError("investigation scores are not active for this iteration")
        selection = self._selector.select(
            scores,
            budget=record.session.budget,
            usage=record.session.usage,
            worker_capacity=worker_capacity,
            top_k=top_k,
        )
        record.append("investigation_selection", selection.to_dict())
        self._repository.save(record)
        self._emit(
            "investigation.selected",
            record,
            {
                "selected": [item.candidate.investigation_id for item in selection.selected],
                "rejected": selection.rejected,
            },
        )
        return selection

    def build_plan(
        self,
        session_id: str,
        selection: SelectionResult,
        *,
        dependencies: Mapping[str, tuple[str, ...]] | None = None,
    ) -> InvestigationPlan:
        record = self._record(session_id)
        selection_artifact = self._required_artifact(record, "investigation_selection")
        if (
            selection_artifact.iteration != record.session.iteration
            or selection_artifact.payload != selection.to_dict()
        ):
            raise DomainError("investigation selection is not active for this iteration")
        plan = self._planner.build(record.session, selection, dependencies=dependencies)
        record.append("investigation_plan", plan.to_dict())
        self._repository.save(record)
        self._emit("investigation.plan_created", record, {"plan_id": plan.plan_id})
        return plan

    def plan_iteration(
        self,
        session_id: str,
        *,
        worker_capacity: int,
        top_k: int | None = None,
        dependencies: Mapping[str, tuple[str, ...]] | None = None,
    ) -> PlanningOutcome:
        """Thin composition of the independently callable planning stages."""
        snapshot = self.observe(session_id)
        candidates = self.generate(session_id, snapshot)
        scores = self.score(session_id, snapshot, candidates)
        selection = self.select(session_id, scores, worker_capacity=worker_capacity, top_k=top_k)
        if selection.selected:
            plan = self.build_plan(session_id, selection, dependencies=dependencies)
            return PlanningOutcome(snapshot, candidates, scores, selection, plan)
        record = self._record(session_id)
        decision = self._termination.evaluate(
            record.session,
            TerminationContext(
                objective_satisfied=not snapshot.gaps,
                objective_confidence=float(snapshot.metadata.get("graphrag_confidence", 0.0)),
                remaining_gap_count=len(snapshot.gaps),
                best_candidate_score=None,
            ),
        )
        self._apply_termination(record, decision)
        return PlanningOutcome(snapshot, candidates, scores, selection, None, decision)

    def start_execution(
        self,
        session_id: str,
        plan: InvestigationPlan,
        run_ids: Mapping[str, str],
    ) -> ExecutionStatus:
        record = self._record(session_id)
        if self._execution is None:
            raise DomainError("distributed execution adapter is not configured")
        if record.session.state != SessionState.PLANNING or plan.session_id != session_id:
            raise DomainError("session is not ready to execute this plan")
        plan_artifact = self._required_artifact(record, "investigation_plan")
        if (
            plan_artifact.iteration != record.session.iteration
            or plan_artifact.payload != plan.to_dict()
        ):
            raise DomainError("only the current iteration plan can be executed")
        if any(
            artifact.kind == "plan_execution"
            and artifact.iteration == record.session.iteration
            and artifact.payload.get("plan_id") == plan.plan_id
            for artifact in record.artifacts
        ):
            raise DomainError("investigation plan has already been started")
        investigation_cost = sum(item.estimated_cost for item in plan.investigations)
        remaining = record.session.remaining_budget()
        if (
            len(plan.investigations) > remaining["investigations"]
            or len(run_ids) > remaining["agent_runs"]
            or investigation_cost > remaining["cost"]
        ):
            raise DomainError("investigation plan exceeds remaining session budget")
        execution = self._execution.prepare(plan, run_ids)
        record.session.transition(SessionState.EXECUTING)
        record.session.record_usage(
            investigations=len(plan.investigations),
            agent_runs=len(run_ids),
            cost=investigation_cost,
        )
        record.append("plan_execution", execution.to_dict())
        self._repository.save(record)
        status = self._execution.advance(plan, execution)
        record.append("plan_execution", status.execution.to_dict())
        self._repository.save(record)
        self._metrics.increment("investigations_executed", len(plan.investigations))
        self._emit(
            "investigation.execution_started",
            record,
            {"plan_id": plan.plan_id, "task_ids": status.execution.task_ids},
        )
        return status

    def advance_execution(self, session_id: str) -> ExecutionStatus:
        record = self._record(session_id)
        if self._execution is None:
            raise DomainError("distributed execution adapter is not configured")
        if record.session.state != SessionState.EXECUTING:
            raise DomainError("distributed execution can advance only while EXECUTING")
        plan_artifact = self._required_artifact(record, "investigation_plan")
        execution_artifact = self._required_artifact(record, "plan_execution")
        plan = InvestigationPlan.from_dict(plan_artifact.payload)
        execution = PlanExecution.from_dict(execution_artifact.payload)
        status = self._execution.advance(plan, execution)
        record.append("plan_execution", status.execution.to_dict())
        self._repository.save(record)
        return status

    def collect_execution_results(self, session_id: str) -> tuple[InvestigationResult, ...]:
        if self._result_collector is None:
            raise DomainError("distributed result collector is not configured")
        record = self._record(session_id)
        execution = PlanExecution.from_dict(
            self._required_artifact(record, "plan_execution").payload
        )
        return self._result_collector.collect(execution)

    def collect_evidence(
        self,
        session_id: str,
        results: Sequence[InvestigationResult],
    ) -> EvidenceSet:
        record = self._record(session_id)
        if record.session.state != SessionState.EXECUTING:
            raise DomainError("evidence can be collected only while EXECUTING")
        if any(item.session_id != session_id for item in results):
            raise DomainError("investigation result belongs to another session")
        if any(
            item.kind == "evidence_set" and item.iteration == record.session.iteration
            for item in record.artifacts
        ):
            raise DomainError("evidence has already been collected for this iteration")
        execution = PlanExecution.from_dict(
            self._required_artifact(record, "plan_execution").payload
        )
        self._validate_results(execution, results)
        if self._result_collector is None:
            raise DomainError("distributed result collector is required for lineage validation")
        authoritative = {item.result_id: item for item in self._result_collector.collect(execution)}
        if any(
            item.result_id not in authoritative
            or item.to_dict() != authoritative[item.result_id].to_dict()
            for item in results
        ):
            raise DomainError("investigation result is not the durable runtime outcome")
        completed = tuple(
            item
            for item in results
            if item.state in {InvestigationResultState.COMPLETED, InvestigationResultState.PARTIAL}
        )
        successful = sum(item.state == InvestigationResultState.COMPLETED for item in results)
        failed = sum(
            item.state in {InvestigationResultState.FAILED, InvestigationResultState.CANCELLED}
            for item in results
        )
        evidence = tuple(item for result in completed for item in result.evidence_set.evidence)
        evidence_set = EvidenceSet(session_id=session_id, evidence=evidence)
        record.append("investigation_results", {"items": [item.to_dict() for item in results]})
        record.append("evidence_set", evidence_set.to_dict())
        candidate_by_id: dict[str, CandidateClaim] = {}
        diagnostics: list[ExtractionDiagnostic] = []
        for result in completed:
            extraction = self._extractor.extract(result)
            for candidate in extraction.candidates:
                candidate_by_id.setdefault(candidate.candidate_id, candidate)
            diagnostics.extend(extraction.diagnostics)
        extraction = CandidateExtractionResult(
            session_id=session_id,
            candidates=tuple(sorted(candidate_by_id.values(), key=lambda item: item.candidate_id)),
            diagnostics=tuple(
                sorted(diagnostics, key=lambda item: (item.conclusion_id, item.code))
            ),
            evidence_set=evidence_set,
        )
        record.append(
            "candidate_claims",
            {
                "conclusions": sum(len(result.conclusions) for result in completed),
                "items": [extraction.to_dict()],
            },
        )
        self._metrics.increment("candidates_extracted", len(extraction.candidates))
        self._metrics.increment("extraction_diagnostics", len(extraction.diagnostics))
        if results:
            completed_at = max(item.completed_at for item in results)
            elapsed = max(timedelta(0), completed_at - record.session.updated_at)
            record.session.record_usage(execution_time=elapsed)
        self._repository.save(record)
        self._metrics.increment("evidence_collected", len(evidence))
        self._metrics.increment("investigation_successes", successful)
        self._metrics.increment("investigation_failures", failed)
        self._emit(
            "investigation.evidence_collected",
            record,
            {"evidence_set_id": evidence_set.evidence_set_id, "count": len(evidence)},
        )
        return evidence_set

    def stored_results(self, session_id: str) -> tuple[InvestigationResult, ...]:
        """Rehydrate durable worker results after an application restart."""
        record = self._record(session_id)
        artifact = self._required_artifact(record, "investigation_results")
        items = artifact.payload.get("items")
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise DomainError("persisted investigation results are malformed")
        try:
            return tuple(InvestigationResult.from_dict(item) for item in items)
        except ValueError as exc:
            raise DomainError("persisted investigation results are malformed") from exc

    def evaluate(self, session_id: str, evidence_set: EvidenceSet) -> Evaluation:
        record = self._record(session_id)
        if evidence_set.session_id != session_id:
            raise DomainError("evidence set belongs to another session")
        persisted_evidence = self._required_artifact(record, "evidence_set")
        if (
            persisted_evidence.iteration != record.session.iteration
            or persisted_evidence.payload != evidence_set.to_dict()
        ):
            raise DomainError("evidence set is not active for this iteration")
        if record.session.state == SessionState.EXECUTING:
            record.session.transition(SessionState.EVALUATING)
        elif record.session.state != SessionState.EVALUATING:
            raise DomainError("evidence can be evaluated only after execution")
        evaluation = self._evaluator.evaluate(evidence_set)
        record.append("evaluation", _jsonable(evaluation))
        self._repository.save(record)
        self._metrics.increment("contradictions_detected", len(evaluation.conflict_ids))
        self._emit(
            "investigation.evaluated",
            record,
            {
                "evaluation_id": evaluation.evaluation_id,
                "accepted_evidence": evaluation.accepted_evidence_count,
                "conflicts": len(evaluation.conflict_ids),
            },
        )
        return evaluation

    def verify(self, session_id: str, evaluation: Evaluation) -> VerificationReport:
        record = self._record(session_id)
        if record.session.state != SessionState.EVALUATING:
            raise DomainError("claim verification requires an EVALUATING session")
        if evaluation.session_id != session_id:
            raise DomainError("evaluation belongs to another session")
        evaluation_artifact = self._required_artifact(record, "evaluation")
        if (
            evaluation_artifact.iteration != record.session.iteration
            or evaluation_artifact.payload != _jsonable(evaluation)
        ):
            raise DomainError("evaluation is not active for this iteration")
        self._emit("investigation.verification_started", record, {})
        verification = self._verifier.verify(evaluation)
        record.append("verification", verification.to_dict())
        self._repository.save(record)
        return verification

    @staticmethod
    def _session_extraction(
        candidate_artifact: InvestigationArtifact,
        session_id: str,
        evidence_set: EvidenceSet,
    ) -> CandidateExtractionResult:
        items = candidate_artifact.payload.get("items")
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise DomainError("persisted candidate claims are malformed")
        try:
            extractions = tuple(CandidateExtractionResult.from_dict(item) for item in items)
        except ValueError as exc:
            raise DomainError("persisted candidate claims are malformed") from exc
        candidates = tuple(
            sorted(
                (candidate for item in extractions for candidate in item.candidates),
                key=lambda item: item.candidate_id,
            )
        )
        diagnostics = tuple(
            sorted(
                (diagnostic for item in extractions for diagnostic in item.diagnostics),
                key=lambda item: (item.conclusion_id, item.code),
            )
        )
        return CandidateExtractionResult(
            session_id=session_id,
            candidates=candidates,
            diagnostics=diagnostics,
            evidence_set=evidence_set,
        )

    @staticmethod
    def _verified_report(
        acquisition: AcquisitionReport,
        verification: VerificationReport,
    ) -> VerificationReport:
        return VerificationReport(
            verification_id=verification.verification_id,
            session_id=verification.session_id,
            evaluation_id=verification.evaluation_id,
            decisions=tuple(
                item.decision for item in acquisition.verified if item.decision is not None
            ),
            verified_at=verification.verified_at,
        )

    def _record_acquisition_metrics(
        self,
        extraction: CandidateExtractionResult,
        acquisition: AcquisitionReport,
    ) -> None:
        self._metrics.increment("claims_acquired", len(acquisition.acquisitions))
        self._metrics.increment("claims_verified", len(acquisition.verified))
        self._metrics.increment("claims_deferred", len(acquisition.deferred))
        self._metrics.increment("claims_rejected", len(acquisition.rejected))
        self._metrics.increment("extraction_diagnostics", len(extraction.diagnostics))

    def update_knowledge(
        self,
        session_id: str,
        verification: VerificationReport,
        evidence_set: EvidenceSet,
    ) -> tuple[InvestigationKnowledgeUpdate, KnowledgeUpdateResult]:
        record = self._record(session_id)
        if record.session.state != SessionState.EVALUATING:
            raise DomainError("knowledge update requires an EVALUATING session")
        if verification.session_id != session_id or evidence_set.session_id != session_id:
            raise DomainError("verification or evidence belongs to another session")
        evidence_artifact = self._required_artifact(record, "evidence_set")
        if (
            evidence_artifact.iteration != record.session.iteration
            or evidence_artifact.payload != evidence_set.to_dict()
        ):
            raise DomainError("evidence set is not active for this iteration")
        verification_artifact = self._required_artifact(record, "verification")
        if (
            verification_artifact.iteration != record.session.iteration
            or verification_artifact.payload != verification.to_dict()
        ):
            raise DomainError("verification is not active for this iteration")
        candidate_artifact = self._required_artifact(record, "candidate_claims")
        if candidate_artifact.iteration != record.session.iteration:
            raise DomainError("candidate claims are not active for this iteration")
        extraction = self._session_extraction(candidate_artifact, session_id, evidence_set)
        acquisition = self._acquisition.acquire(extraction, verification)
        record.append("claim_acquisition", acquisition.to_dict())
        self._repository.save(record)
        self._record_acquisition_metrics(extraction, acquisition)
        verified = self._verified_report(acquisition, verification)
        record.session.transition(SessionState.UPDATING)
        update = self._updates.prepare(verified, evidence_set)
        record.append("knowledge_update", update.to_dict())
        self._repository.save(record)
        result = self._updates.apply(update)
        record.append("knowledge_update_result", result.to_dict())
        self._repository.save(record)
        self._metrics.increment("knowledge_updates", len(result.committed_claim_ids))
        self._emit(
            "investigation.knowledge_updated",
            record,
            {
                "update_id": update.update_id,
                "committed_claim_ids": list(result.committed_claim_ids),
                "unresolved_contradiction_ids": list(result.unresolved_contradiction_ids),
            },
        )
        return update, result

    def finish_iteration(
        self,
        session_id: str,
        *,
        before: KnowledgeSnapshot,
        evidence_set: EvidenceSet,
        verification: VerificationReport,
        update_result: KnowledgeUpdateResult,
        objective_satisfied: bool = False,
        contradictions_resolvable: bool = True,
    ) -> tuple[ProgressReport, TerminationDecision]:
        record = self._record(session_id)
        if record.session.state != SessionState.UPDATING:
            raise DomainError("iteration can finish only while UPDATING")
        snapshot_artifact = self._required_artifact(record, "knowledge_snapshot")
        if (
            snapshot_artifact.iteration != record.session.iteration
            or snapshot_artifact.payload != before.to_dict()
        ):
            raise DomainError("before snapshot is not active for this iteration")
        evidence_artifact = self._required_artifact(record, "evidence_set")
        verification_artifact = self._required_artifact(record, "verification")
        update_result_artifact = self._required_artifact(record, "knowledge_update_result")
        if (
            evidence_artifact.iteration != record.session.iteration
            or evidence_artifact.payload != evidence_set.to_dict()
            or verification_artifact.iteration != record.session.iteration
            or verification_artifact.payload != verification.to_dict()
            or update_result_artifact.iteration != record.session.iteration
            or update_result_artifact.payload != update_result.to_dict()
        ):
            raise DomainError("iteration inputs do not match active persisted artifacts")
        after = self._observer.observe(record.objective)
        record.append("knowledge_snapshot", after.to_dict())
        progress = ProgressMeasurer().measure(
            session_id=session_id,
            iteration=record.session.iteration + 1,
            before_gaps=(GapState(item.id, item.uncertainty) for item in before.gaps),
            after_gaps=(GapState(item.id, item.uncertainty) for item in after.gaps),
            before_contradiction_ids=before.contradiction_ids,
            after_contradiction_ids=after.contradiction_ids,
            evidence_collected=len(evidence_set.evidence),
            knowledge_updates=len(update_result.committed_claim_ids),
            cost=record.session.usage.cost,
        )
        record.append("progress_report", _jsonable(progress))
        record.session.complete_iteration()
        confidence = (
            min(item.confidence for item in verification.decisions)
            if verification.decisions and not after.gaps
            else 0.0
        )
        latest_scores = record.latest("investigation_scores")
        score_items = [] if latest_scores is None else latest_scores.payload.get("items", [])
        best_score = None
        if isinstance(score_items, list) and score_items:
            best_score = max(float(item["score"]) for item in score_items if isinstance(item, dict))
        decision = self._termination.evaluate(
            record.session,
            TerminationContext(
                objective_satisfied=objective_satisfied,
                objective_confidence=confidence,
                remaining_gap_count=len(after.gaps),
                best_candidate_score=best_score,
                unresolved_contradictions=len(update_result.unresolved_contradiction_ids),
                contradictions_resolvable=contradictions_resolvable,
            ),
        )
        record.append("termination_decision", decision.to_dict())
        self._metrics.increment("iterations")
        self._metrics.increment("gaps_resolved", len(progress.resolved_gap_ids))
        self._metrics.increment("contradictions_resolved", len(progress.contradictions_resolved))
        self._metrics.observe("average_information_gain", progress.information_gain)
        if progress.cost_per_resolved_gap is not None:
            self._metrics.observe("cost_per_resolved_gap", progress.cost_per_resolved_gap)
        self._emit(
            "investigation.iteration_completed",
            record,
            {"iteration": record.session.iteration, "information_gain": progress.information_gain},
        )
        self._apply_termination(record, decision)
        if not decision.terminate:
            self._repository.save(record)
        return progress, decision

    def process_results(
        self,
        session_id: str,
        results: Sequence[InvestigationResult],
        *,
        objective_satisfied: bool = False,
        contradictions_resolvable: bool = True,
    ) -> IterationOutcome:
        """Thin composition of collection, evaluation, verification, update, and progress."""
        record = self._record(session_id)
        before_artifact = self._required_artifact(record, "knowledge_snapshot")
        before = KnowledgeSnapshot.from_dict(before_artifact.payload)
        evidence_set = self.collect_evidence(session_id, results)
        evaluation = self.evaluate(session_id, evidence_set)
        verification = self.verify(session_id, evaluation)
        update, update_result = self.update_knowledge(session_id, verification, evidence_set)
        progress, termination = self.finish_iteration(
            session_id,
            before=before,
            evidence_set=evidence_set,
            verification=verification,
            update_result=update_result,
            objective_satisfied=objective_satisfied,
            contradictions_resolvable=contradictions_resolvable,
        )
        return IterationOutcome(
            evidence_set,
            evaluation,
            verification,
            update,
            update_result,
            progress,
            termination,
        )

    def resume_collected_iteration(
        self,
        session_id: str,
        *,
        objective_satisfied: bool = False,
        contradictions_resolvable: bool = True,
    ) -> IterationOutcome:
        """Resume after durable collection without rerunning agents or duplicating evidence."""
        record = self._record(session_id)
        if record.session.state != SessionState.EXECUTING:
            raise DomainError("collected iteration can resume only from EXECUTING")
        before = KnowledgeSnapshot.from_dict(
            self._required_artifact(record, "knowledge_snapshot").payload
        )
        evidence_set = EvidenceSet.from_dict(
            self._required_artifact(record, "evidence_set").payload
        )
        evaluation = self.evaluate(session_id, evidence_set)
        verification = self.verify(session_id, evaluation)
        update, update_result = self.update_knowledge(session_id, verification, evidence_set)
        progress, termination = self.finish_iteration(
            session_id,
            before=before,
            evidence_set=evidence_set,
            verification=verification,
            update_result=update_result,
            objective_satisfied=objective_satisfied,
            contradictions_resolvable=contradictions_resolvable,
        )
        return IterationOutcome(
            evidence_set,
            evaluation,
            verification,
            update,
            update_result,
            progress,
            termination,
        )

    def resume_iteration(
        self,
        session_id: str,
        *,
        objective_satisfied: bool = False,
        contradictions_resolvable: bool = True,
    ) -> IterationOutcome:
        """Resume from the latest durable stage without rerunning completed work."""
        state = self._record(session_id).session.state
        if state == SessionState.EXECUTING:
            return self.resume_collected_iteration(
                session_id,
                objective_satisfied=objective_satisfied,
                contradictions_resolvable=contradictions_resolvable,
            )
        if state == SessionState.EVALUATING:
            return self._resume_evaluating(
                session_id,
                objective_satisfied=objective_satisfied,
                contradictions_resolvable=contradictions_resolvable,
            )
        if state == SessionState.UPDATING:
            return self._resume_updating(
                session_id,
                objective_satisfied=objective_satisfied,
                contradictions_resolvable=contradictions_resolvable,
            )
        raise DomainError("session is not at a resumable iteration stage")

    def _resume_evaluating(
        self,
        session_id: str,
        *,
        objective_satisfied: bool,
        contradictions_resolvable: bool,
    ) -> IterationOutcome:
        record = self._record(session_id)
        before = KnowledgeSnapshot.from_dict(
            self._required_artifact(record, "knowledge_snapshot").payload
        )
        evidence_set = EvidenceSet.from_dict(
            self._required_artifact(record, "evidence_set").payload
        )
        verification_artifact = record.latest("verification")
        if (
            verification_artifact is not None
            and verification_artifact.iteration == record.session.iteration
        ):
            verification = VerificationReport.from_dict(verification_artifact.payload)
            evaluation = self._evaluator.evaluate(evidence_set)
        else:
            evaluation = self.evaluate(session_id, evidence_set)
            verification = self.verify(session_id, evaluation)
        update, update_result = self.update_knowledge(session_id, verification, evidence_set)
        progress, termination = self.finish_iteration(
            session_id,
            before=before,
            evidence_set=evidence_set,
            verification=verification,
            update_result=update_result,
            objective_satisfied=objective_satisfied,
            contradictions_resolvable=contradictions_resolvable,
        )
        return IterationOutcome(
            evidence_set,
            evaluation,
            verification,
            update,
            update_result,
            progress,
            termination,
        )

    def _resume_updating(
        self,
        session_id: str,
        *,
        objective_satisfied: bool,
        contradictions_resolvable: bool,
    ) -> IterationOutcome:
        record = self._record(session_id)
        before = KnowledgeSnapshot.from_dict(
            self._required_artifact(record, "knowledge_snapshot").payload
        )
        evidence_set = EvidenceSet.from_dict(
            self._required_artifact(record, "evidence_set").payload
        )
        verification = VerificationReport.from_dict(
            self._required_artifact(record, "verification").payload
        )
        update = InvestigationKnowledgeUpdate.from_dict(
            self._required_artifact(record, "knowledge_update").payload
        )
        result_artifact = record.latest("knowledge_update_result")
        if result_artifact is not None and result_artifact.iteration == record.session.iteration:
            update_result = KnowledgeUpdateResult.from_dict(result_artifact.payload)
        else:
            update_result = self._updates.apply(update)
            record.append("knowledge_update_result", update_result.to_dict())
            self._repository.save(record)
        evaluation = self._evaluator.evaluate(evidence_set)
        progress, termination = self.finish_iteration(
            session_id,
            before=before,
            evidence_set=evidence_set,
            verification=verification,
            update_result=update_result,
            objective_satisfied=objective_satisfied,
            contradictions_resolvable=contradictions_resolvable,
        )
        return IterationOutcome(
            evidence_set,
            evaluation,
            verification,
            update,
            update_result,
            progress,
            termination,
        )

    def pause(self, session_id: str) -> InvestigationSession:
        record = self._record(session_id)
        record.session.pause()
        assert record.session.paused_from is not None
        record.append("session_paused", {"paused_from": record.session.paused_from.value})
        self._repository.save(record)
        return record.session

    def resume(self, session_id: str) -> InvestigationSession:
        record = self._record(session_id)
        record.session.resume()
        record.append("session_resumed", {"state": record.session.state.value})
        self._repository.save(record)
        return record.session

    def cancel(self, session_id: str, principal: str = "user") -> InvestigationSession:
        record = self._record(session_id)
        execution_artifact = record.latest("plan_execution")
        if execution_artifact is not None and self._execution is not None:
            execution = PlanExecution.from_dict(execution_artifact.payload)
            self._execution.cancel(execution, principal)
        record.session.transition(
            SessionState.CANCELLED, reason=TerminationReason.USER_CANCELLATION
        )
        record.append("termination_decision", {"reason": TerminationReason.USER_CANCELLATION.value})
        self._repository.save(record)
        self._metrics.increment("investigation_sessions_completed")
        self._emit("investigation.completed", record, {"reason": "user_cancellation"})
        return record.session

    def explain(self, session_id: str) -> dict[str, object]:
        record = self._record(session_id)
        return {
            "objective": record.objective.to_dict(),
            "session": record.session.to_dict(),
            "why_investigated": [
                item.payload for item in record.artifacts if item.kind == "investigation_scores"
            ],
            "why_believed": [
                item.payload for item in record.artifacts if item.kind == "verification"
            ],
            "why_stopped": [
                item.payload for item in record.artifacts if item.kind == "termination_decision"
            ],
        }

    def _record(self, session_id: str) -> InvestigationRecord:
        record = self._repository.get(session_id)
        if record is None:
            raise DomainError(f"unknown investigation session: {session_id}")
        return record

    @staticmethod
    def _required_artifact(record: InvestigationRecord, kind: str) -> Any:
        artifact = record.latest(kind)
        if artifact is None:
            raise DomainError(f"session has no {kind} artifact")
        return artifact

    @staticmethod
    def _require_snapshot(record: InvestigationRecord, snapshot: KnowledgeSnapshot) -> None:
        if snapshot.objective_id != record.objective.objective_id:
            raise DomainError("knowledge snapshot belongs to another objective")
        if record.session.state != SessionState.PLANNING:
            raise DomainError("investigation planning requires a PLANNING session")
        artifact = record.latest("knowledge_snapshot")
        if (
            artifact is None
            or artifact.iteration != record.session.iteration
            or artifact.payload != snapshot.to_dict()
        ):
            raise DomainError("knowledge snapshot is not active for this iteration")

    @staticmethod
    def _validate_results(execution: PlanExecution, results: Sequence[InvestigationResult]) -> None:
        result_ids = [item.result_id for item in results]
        investigation_ids = [item.investigation_id for item in results]
        if len(result_ids) != len(set(result_ids)):
            raise DomainError("investigation result identifiers must be unique")
        if len(investigation_ids) != len(set(investigation_ids)):
            raise DomainError("only one result per investigation can be collected")
        for result in results:
            run_id = execution.run_ids.get(result.investigation_id)
            if run_id is None or result.run_id != run_id:
                raise DomainError("investigation result does not belong to the active plan")
            task_id = execution.task_ids.get(result.investigation_id)
            if task_id is not None and result.task_id != task_id:
                raise DomainError("investigation result task does not match the active plan")
            if result.investigation_id in execution.blocked_investigations:
                prefix = f"blocked:{execution.plan_id}:{result.investigation_id}"
                if result.task_id != prefix or result.state != InvestigationResultState.FAILED:
                    raise DomainError("blocked investigation result is malformed")
            elif task_id is None:
                raise DomainError("result was produced for an investigation that was not submitted")

    def _apply_termination(
        self, record: InvestigationRecord, decision: TerminationDecision
    ) -> None:
        if not decision.terminate:
            self._repository.save(record)
            return
        assert decision.target_state is not None and decision.reason is not None
        record.session.transition(decision.target_state, reason=decision.reason)
        self._repository.save(record)
        event_type = (
            "investigation.failed"
            if decision.target_state == SessionState.FAILED
            else "investigation.completed"
        )
        self._metrics.increment(
            "investigation_sessions_failed"
            if decision.target_state == SessionState.FAILED
            else "investigation_sessions_completed"
        )
        self._emit(event_type, record, {"reason": decision.reason.value})

    def _emit(
        self, event_type: str, record: InvestigationRecord, payload: Mapping[str, Any]
    ) -> None:
        self._events.publish(
            Event(
                event_type=event_type,
                payload={
                    "session_id": record.session.session_id,
                    "objective_id": record.objective.objective_id,
                    "iteration": record.session.iteration,
                    **dict(payload),
                },
                producer="investigation-application",
                trace_id=record.session.session_id,
                correlation_id=record.objective.objective_id,
            )
        )


def _jsonable(value: Any) -> dict[str, Any]:
    def default(item: object) -> object:
        if isinstance(item, datetime):
            return item.isoformat()
        if isinstance(item, Enum):
            return item.value
        if is_dataclass(item):
            return asdict(cast(Any, item))
        if isinstance(item, tuple | set | frozenset):
            return list(item)
        raise TypeError(f"cannot serialize {type(item).__name__}")

    encoded = json.dumps(value, default=default, sort_keys=True, allow_nan=False)
    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise DomainError("investigation artifact must serialize as an object")
    return cast(dict[str, Any], payload)
