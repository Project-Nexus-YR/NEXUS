from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus_knowledge.domain.common import VerificationState
from nexus_knowledge.domain.knowledge_gap import Investigation, KnowledgeGap
from nexus_knowledge.knowledge.uncertainty import UncertaintyAssessment
from nexus_knowledge.service.engine import KnowledgeUpdate, KnowledgeUpdateReceipt
from nexus_runtime.agent import AgentExecutor
from nexus_runtime.distributed.model import FailureClass, RetryPolicy
from nexus_runtime.distributed.service import RuntimeApplication
from nexus_runtime.distributed.simulator import LocalDistributedSimulator
from nexus_runtime.distributed.worker import (
    HarnessExecutionContext,
    HarnessOutcome,
    HarnessStatus,
)
from nexus_runtime.events import InMemoryEventBus
from nexus_runtime.investigation.agent_harness import AgentHarness
from nexus_runtime.investigation.application import InvestigationApplication
from nexus_runtime.investigation.evidence import (
    ClaimStatement,
    Evidence,
    EvidenceSet,
    InvestigationResult,
    InvestigationResultState,
)
from nexus_runtime.investigation.execution import PlanExecution, PlanExecutionController
from nexus_runtime.investigation.metrics import InMemoryInvestigationMetrics
from nexus_runtime.investigation.objective import ResearchObjective
from nexus_runtime.investigation.provenance import EvidenceProvenance
from nexus_runtime.investigation.repository import (
    InMemoryInvestigationRepository,
    SQLiteInvestigationRepository,
)
from nexus_runtime.investigation.results import (
    InMemoryInvestigationResultRepository,
    RuntimeResultCollector,
    SQLiteInvestigationResultRepository,
)
from nexus_runtime.investigation.session import (
    InvestigationBudget,
    SessionState,
    TerminationReason,
)
from nexus_runtime.models import Agent, Budget, DomainError
from nexus_runtime.policy import PolicyEngine
from nexus_runtime.tools import ToolRegistry

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def budget(max_iterations: int = 3) -> InvestigationBudget:
    return InvestigationBudget(max_iterations, 20, 20, 100.0, timedelta(hours=1))


def objective() -> ResearchObjective:
    return ResearchObjective(
        question="What is Acme's verified status?",
        success_criteria=("status has two independent sources",),
        objective_id="objective-app",
        created_at=NOW,
    )


def gap(index: int = 0) -> KnowledgeGap:
    item = KnowledgeGap(
        id=f"gap-{index}",
        kind="missing_evidence",
        description=f"verify Acme status {index}",
        reason="no independent evidence",
        affected_entities=["acme"],
        uncertainty=0.9,
        importance=0.9,
        estimated_cost=1.0,
        created_at="2026-01-01T00:00:00Z",
    )
    item.candidate_investigations = [
        Investigation(
            id=f"legacy-{index}",
            gap_id=item.id,
            description=f"find independent evidence for Acme {index}",
            target_entities=["acme"],
            estimated_cost=1.0,
            created_at="2026-01-01T00:00:00Z",
        )
    ]
    return item


class FakeKnowledge:
    def __init__(self, gaps: list[KnowledgeGap] | None = None) -> None:
        self.gaps = [gap()] if gaps is None else gaps
        self.updated = False
        self.committed: list[KnowledgeUpdate] = []
        self.fail_commit_once = False

    def retrieve(self, query: str, top_k: int = 10) -> object:
        candidate = SimpleNamespace(chunk_id="chunk-existing")
        return SimpleNamespace(request_id="request-1", candidates=[candidate])

    def graphrag(self, query: str, top_k: int = 8, depth: int = 2) -> object:
        entity = SimpleNamespace(id="acme")
        return SimpleNamespace(entities=[entity], relations=[], confidence=0.4)

    def find_knowledge_gaps(self) -> list[KnowledgeGap]:
        return [] if self.updated else self.gaps

    def detect_contradictions(self) -> list[object]:
        return []

    def commit_knowledge_update(self, update: KnowledgeUpdate) -> KnowledgeUpdateReceipt:
        if self.fail_commit_once:
            self.fail_commit_once = False
            raise RuntimeError("injected knowledge commit failure")
        self.committed.append(update)
        self.updated = bool(update.claims)
        return KnowledgeUpdateReceipt(
            accepted=len(update.claims) + len(update.evidence), rejected=0, errors=[]
        )

    def verify_claim(self, claim_id: str) -> UncertaintyAssessment:
        return UncertaintyAssessment(
            claim_id=claim_id,
            confidence=0.9,
            uncertainty=0.1,
            verification_state=VerificationState.VERIFIED,
            supporting_evidence_count=2,
            contradicting_evidence_count=0,
            source_quality=0.9,
            source_diversity=1.0,
            recency=1.0,
            components={},
        )

    def validate_evidence_provenance(
        self,
        source_id: str,
        document_id: str,
        chunk_id: str,
        source_reference: str,
    ) -> bool:
        return all((source_id, document_id, chunk_id, source_reference))


class EvidenceHarness:
    def __init__(
        self,
        results: InMemoryInvestigationResultRepository,
        fail_once: bool = False,
    ) -> None:
        self.result_repository = results
        self.fail_once = fail_once
        self.calls = 0
        self.results: list[InvestigationResult] = []

    def execute_or_resume(
        self,
        context: HarnessExecutionContext,
        cancellation_requested,
    ):
        self.calls += 1
        if self.fail_once and self.calls == 1:
            return HarnessOutcome(
                HarnessStatus.FAILED,
                checkpoint_ref="checkpoint://evidence-agent",
                failure_class=FailureClass.TRANSIENT,
                error="temporary source failure",
            )
        task = context
        investigation_id = str(context.metadata["investigation_id"])
        claim = ClaimStatement(
            text="Acme is active",
            subject="Acme",
            predicate="status",
            object="active",
            claim_id="claim-acme-active",
        )
        evidence = tuple(
            Evidence(
                investigation_id=investigation_id,
                source=f"source://{source}",
                claim=claim,
                provenance=EvidenceProvenance(
                    session_id=context.correlation_id,
                    investigation_id=investigation_id,
                    task_id=task.task_id,
                    attempt_id=context.attempt_id,
                    run_id=context.run_id,
                    tool_call_id=f"tool-{source}",
                    source_id=f"source-{source}",
                    document_id=f"document-{source}",
                    chunk_id=f"chunk-{source}",
                    source_reference=f"source://{source}",
                ),
                confidence=0.95,
                source_quality=0.9,
                excerpt=f"independent report {source}",
                evidence_id=f"evidence-{context.task_id}-{source}",
            )
            for source in ("a", "b")
        )
        evidence_set = EvidenceSet(session_id=context.correlation_id, evidence=evidence)
        result = InvestigationResult(
            session_id=context.correlation_id,
            investigation_id=investigation_id,
            task_id=context.task_id,
            attempt_id=context.attempt_id,
            run_id=context.run_id,
            state=InvestigationResultState.COMPLETED,
            evidence_set=evidence_set,
        )
        self.results.append(result)
        return HarnessOutcome(HarnessStatus.SUCCEEDED, self.result_repository.save(result))

    def cancel_run(self, run_id: str) -> None:
        return None


def make_runtime(fail_once: bool = False):
    simulator = LocalDistributedSimulator(start=NOW)
    results = InMemoryInvestigationResultRepository()
    harness = EvidenceHarness(results, fail_once=fail_once)
    simulator.add_worker(
        "evidence-worker",
        frozenset({"document_analysis", "search"}),
        harness,
    )
    runtime = RuntimeApplication(simulator.coordinator)
    return (
        simulator,
        harness,
        PlanExecutionController(runtime),
        RuntimeResultCollector(runtime, results),
    )


def test_closed_loop_resolves_gap_and_completes_objective() -> None:
    knowledge = FakeKnowledge()
    simulator, _, execution, collector = make_runtime()
    events = InMemoryEventBus()
    metrics = InMemoryInvestigationMetrics()
    app = InvestigationApplication(
        knowledge,
        repository=InMemoryInvestigationRepository(),
        execution=execution,
        result_collector=collector,
        event_bus=events,
        metrics=metrics,
    )
    session = app.create(objective(), budget())
    planning = app.plan_iteration(session.session_id, worker_capacity=1)
    assert planning.plan is not None
    investigation_id = planning.plan.investigations[0].investigation_id
    app.start_execution(
        session.session_id,
        planning.plan,
        {investigation_id: "agent-run-acme"},
    )

    report = simulator.run_until_terminal()
    results = app.collect_execution_results(session.session_id)
    outcome = app.process_results(
        session.session_id,
        results,
        objective_satisfied=True,
    )

    record = app.status(session.session_id)
    assert report.succeeded == 1
    assert outcome.verification.eligible_claims
    assert outcome.update_result.committed_claim_ids == ("claim-acme-active",)
    assert outcome.progress.resolved_gap_ids == ("gap-0",)
    assert record.session.state == SessionState.COMPLETED
    assert record.session.iteration == 1
    assert record.session.termination_reason == TerminationReason.OBJECTIVE_SATISFIED
    event_types = {item.event_type for item in events.events}
    assert {
        "investigation.session_created",
        "investigation.plan_created",
        "investigation.execution_started",
        "investigation.evidence_collected",
        "investigation.knowledge_updated",
        "investigation.completed",
    } <= event_types
    snapshot = metrics.snapshot()
    assert snapshot["counters"]["gaps_resolved"] == 1
    assert snapshot["gauges"]["investigation_success_rate"] == 1.0


def test_distributed_retry_preserves_attempt_lineage() -> None:
    knowledge = FakeKnowledge()
    simulator, harness, execution, _ = make_runtime(fail_once=True)
    app = InvestigationApplication(knowledge, execution=execution)
    session = app.create(objective(), budget())
    planning = app.plan_iteration(session.session_id, worker_capacity=1)
    assert planning.plan is not None
    investigation_id = planning.plan.investigations[0].investigation_id
    status = app.start_execution(
        session.session_id,
        planning.plan,
        {investigation_id: "agent-run-retry"},
    )
    task_id = status.execution.task_ids[investigation_id]
    task = simulator.coordinator.require_task(task_id)
    task.retry_policy = RetryPolicy(
        max_attempts=2,
        initial_backoff=timedelta(0),
        max_backoff=timedelta(0),
    )

    report = simulator.run_until_terminal()

    completed = simulator.coordinator.require_task(task_id)
    assert report.retries == 1
    assert completed.attempt == 2
    assert harness.results[0].attempt_id == completed.attempts[-1].attempt_id
    evidence_attempt = harness.results[0].evidence_set.evidence[0].provenance.attempt_id
    assert evidence_attempt == completed.attempts[-1].attempt_id


def test_worker_loss_reassigns_investigation_without_corrupting_session() -> None:
    knowledge = FakeKnowledge()
    simulator, harness, execution, collector = make_runtime()
    app = InvestigationApplication(
        knowledge,
        execution=execution,
        result_collector=collector,
    )
    session = app.create(objective(), budget())
    planning = app.plan_iteration(session.session_id, worker_capacity=1)
    assert planning.plan is not None
    investigation_id = planning.plan.investigations[0].investigation_id
    started = app.start_execution(
        session.session_id,
        planning.plan,
        {investigation_id: "agent-run-worker-loss"},
    )
    task_id = started.execution.task_ids[investigation_id]
    crashed = simulator.workers[0]
    claimed = simulator.coordinator.claim_task(crashed.identity)
    assert claimed is not None and claimed.lease_id is not None
    simulator.coordinator.start_task(crashed.identity, task_id, claimed.lease_id)
    crashed.crash()

    simulator.clock.advance(timedelta(seconds=31))
    simulator.coordinator.recover()
    simulator.clock.advance(timedelta(seconds=1))
    simulator.coordinator.recover()
    replacement = simulator.add_worker(
        "replacement-worker",
        frozenset({"document_analysis", "search"}),
        harness,
    )
    completed = replacement.poll_once()
    assert completed is not None
    results = app.collect_execution_results(session.session_id)
    outcome = app.process_results(
        session.session_id,
        results,
        objective_satisfied=True,
    )

    assert completed.attempt == 2
    assert completed.attempts[0].error == "lease expired"
    assert results[0].attempt_id == completed.attempts[-1].attempt_id
    assert outcome.update_result.committed_claim_ids == ("claim-acme-active",)
    assert app.status(session.session_id).session.state == SessionState.COMPLETED


def test_plan_dependencies_submit_in_ready_waves() -> None:
    knowledge = FakeKnowledge([gap(0), gap(1)])
    simulator, _, execution, _ = make_runtime()
    app = InvestigationApplication(knowledge, execution=execution)
    session = app.create(objective(), budget())
    snapshot = app.observe(session.session_id)
    candidates = app.generate(session.session_id, snapshot)
    scores = app.score(session.session_id, snapshot, candidates)
    selection = app.select(session.session_id, scores, worker_capacity=2)
    ids = [item.candidate.investigation_id for item in selection.selected]
    plan = app.build_plan(session.session_id, selection, dependencies={ids[1]: (ids[0],)})

    started = app.start_execution(
        session.session_id,
        plan,
        {ids[0]: "run-root", ids[1]: "run-child"},
    )
    assert set(started.execution.task_ids) == {ids[0]}
    simulator.workers[0].poll_once()
    advanced = app.advance_execution(session.session_id)
    assert set(advanced.execution.task_ids) == set(ids)


def test_ready_wave_recovery_reuses_task_after_lost_local_ack() -> None:
    knowledge = FakeKnowledge()
    simulator = LocalDistributedSimulator(start=NOW)
    runtime = RuntimeApplication(simulator.coordinator)
    controller = PlanExecutionController(runtime)
    app = InvestigationApplication(knowledge)
    session = app.create(objective(), budget())
    planning = app.plan_iteration(session.session_id, worker_capacity=1)
    assert planning.plan is not None
    investigation_id = planning.plan.investigations[0].investigation_id
    run_ids = {investigation_id: "run-idempotent-submit"}
    lost_ack_execution = controller.prepare(planning.plan, run_ids)
    first = controller.advance(planning.plan, lost_ack_execution)
    recovered_intent = controller.prepare(planning.plan, run_ids)

    recovered = controller.advance(planning.plan, recovered_intent)

    assert recovered.execution.task_ids == first.execution.task_ids
    assert len(runtime.list_tasks()) == 1


def test_blocked_investigation_becomes_explicit_failed_result() -> None:
    simulator = LocalDistributedSimulator(start=NOW)
    runtime = RuntimeApplication(simulator.coordinator)
    collector = RuntimeResultCollector(runtime, InMemoryInvestigationResultRepository())
    execution = PlanExecution(
        plan_id="plan-blocked",
        session_id="session-blocked",
        run_ids={"investigation-blocked": "run-blocked"},
        blocked_investigations={"investigation-blocked": "dependency failed: parent"},
    )

    results = collector.collect(execution)

    assert results[0].state == InvestigationResultState.FAILED
    assert results[0].metadata == {"blocked": True}
    assert results[0].error == "dependency failed: parent"


def test_cancelled_task_yields_partial_iteration_then_replans() -> None:
    knowledge = FakeKnowledge()
    simulator = LocalDistributedSimulator(start=NOW)
    runtime = RuntimeApplication(simulator.coordinator)
    result_repository = InMemoryInvestigationResultRepository()
    app = InvestigationApplication(
        knowledge,
        execution=PlanExecutionController(runtime),
        result_collector=RuntimeResultCollector(runtime, result_repository),
    )
    session = app.create(objective(), budget())
    planning = app.plan_iteration(session.session_id, worker_capacity=1)
    assert planning.plan is not None
    investigation_id = planning.plan.investigations[0].investigation_id
    started = app.start_execution(
        session.session_id,
        planning.plan,
        {investigation_id: "run-cancelled-task"},
    )
    task_id = started.execution.task_ids[investigation_id]
    runtime.cancel_task(task_id, "failure-injection")

    results = app.collect_execution_results(session.session_id)
    first_iteration = app.process_results(session.session_id, results)
    second_planning = app.plan_iteration(session.session_id, worker_capacity=1)

    assert results[0].state == InvestigationResultState.CANCELLED
    assert not first_iteration.termination.terminate
    assert second_planning.plan is not None
    assert app.status(session.session_id).session.iteration == 1
    with pytest.raises(DomainError, match="current iteration plan"):
        app.start_execution(
            session.session_id,
            planning.plan,
            {investigation_id: "stale-run"},
        )


def test_sqlite_pause_restart_resume_preserves_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "investigations.sqlite"
    first_repository = SQLiteInvestigationRepository(path)
    first = InvestigationApplication(FakeKnowledge(), repository=first_repository)
    session = first.create(objective(), budget())
    first.observe(session.session_id)
    first.pause(session.session_id)
    artifact_count = len(first.status(session.session_id).artifacts)
    first_repository.close()

    resumed_repository = SQLiteInvestigationRepository(path)
    resumed = InvestigationApplication(FakeKnowledge(), repository=resumed_repository)
    state = resumed.resume(session.session_id)

    assert state.state == SessionState.PLANNING
    assert len(resumed.status(session.session_id).artifacts) == artifact_count + 1
    resumed_repository.close()


def test_restart_after_collection_resumes_without_rerunning_agents(tmp_path: Path) -> None:
    path = tmp_path / "collected.sqlite"
    knowledge = FakeKnowledge()
    simulator, harness, execution, collector = make_runtime()
    first_repository = SQLiteInvestigationRepository(path)
    first = InvestigationApplication(
        knowledge,
        repository=first_repository,
        execution=execution,
        result_collector=collector,
    )
    session = first.create(objective(), budget())
    planning = first.plan_iteration(session.session_id, worker_capacity=1)
    assert planning.plan is not None
    investigation_id = planning.plan.investigations[0].investigation_id
    first.start_execution(
        session.session_id,
        planning.plan,
        {investigation_id: "agent-run-resume"},
    )
    simulator.run_until_terminal()
    first.collect_evidence(session.session_id, first.collect_execution_results(session.session_id))
    calls_before_restart = harness.calls
    first_repository.close()

    resumed_repository = SQLiteInvestigationRepository(path)
    resumed = InvestigationApplication(knowledge, repository=resumed_repository)
    stored = resumed.stored_results(session.session_id)
    outcome = resumed.resume_collected_iteration(
        session.session_id,
        objective_satisfied=True,
    )

    assert stored[0].result_id == harness.results[0].result_id
    assert outcome.update_result.committed_claim_ids == ("claim-acme-active",)
    assert resumed.status(session.session_id).session.iteration == 1
    assert harness.calls == calls_before_restart
    resumed_repository.close()


def test_result_reference_survives_repository_restart(tmp_path: Path) -> None:
    simulator, harness, execution, _ = make_runtime()
    knowledge = FakeKnowledge()
    app = InvestigationApplication(knowledge, execution=execution)
    session = app.create(objective(), budget())
    planning = app.plan_iteration(session.session_id, worker_capacity=1)
    assert planning.plan is not None
    investigation_id = planning.plan.investigations[0].investigation_id
    app.start_execution(
        session.session_id,
        planning.plan,
        {investigation_id: "agent-run-durable-result"},
    )
    simulator.run_until_terminal()
    result = harness.results[0]
    path = tmp_path / "results.sqlite"
    first = SQLiteInvestigationResultRepository(path)
    result_ref = first.save(result)
    first.close()

    resumed = SQLiteInvestigationResultRepository(path)
    restored = resumed.get(result_ref)

    assert restored is not None
    assert restored.to_dict() == result.to_dict()
    resumed.close()


def test_restart_from_evaluating_reuses_verified_evidence(tmp_path: Path) -> None:
    path = tmp_path / "evaluating.sqlite"
    knowledge = FakeKnowledge()
    simulator, _, execution, collector = make_runtime()
    first_repository = SQLiteInvestigationRepository(path)
    first = InvestigationApplication(
        knowledge,
        repository=first_repository,
        execution=execution,
        result_collector=collector,
    )
    session = first.create(objective(), budget())
    planning = first.plan_iteration(session.session_id, worker_capacity=1)
    assert planning.plan is not None
    investigation_id = planning.plan.investigations[0].investigation_id
    first.start_execution(
        session.session_id,
        planning.plan,
        {investigation_id: "agent-run-evaluating-resume"},
    )
    simulator.run_until_terminal()
    evidence_set = first.collect_evidence(
        session.session_id,
        first.collect_execution_results(session.session_id),
    )
    evaluation = first.evaluate(session.session_id, evidence_set)
    first.verify(session.session_id, evaluation)
    assert first.status(session.session_id).session.state == SessionState.EVALUATING
    first_repository.close()

    resumed_repository = SQLiteInvestigationRepository(path)
    resumed = InvestigationApplication(knowledge, repository=resumed_repository)
    outcome = resumed.resume_iteration(session.session_id, objective_satisfied=True)

    assert outcome.update_result.committed_claim_ids == ("claim-acme-active",)
    assert resumed.status(session.session_id).session.state == SessionState.COMPLETED
    resumed_repository.close()


def test_restart_from_pending_update_replays_idempotent_claim_ids(tmp_path: Path) -> None:
    path = tmp_path / "updating.sqlite"
    knowledge = FakeKnowledge()
    simulator, _, execution, collector = make_runtime()
    first_repository = SQLiteInvestigationRepository(path)
    first = InvestigationApplication(
        knowledge,
        repository=first_repository,
        execution=execution,
        result_collector=collector,
    )
    session = first.create(objective(), budget())
    planning = first.plan_iteration(session.session_id, worker_capacity=1)
    assert planning.plan is not None
    investigation_id = planning.plan.investigations[0].investigation_id
    first.start_execution(
        session.session_id,
        planning.plan,
        {investigation_id: "agent-run-update-resume"},
    )
    simulator.run_until_terminal()
    evidence_set = first.collect_evidence(
        session.session_id,
        first.collect_execution_results(session.session_id),
    )
    evaluation = first.evaluate(session.session_id, evidence_set)
    verification = first.verify(session.session_id, evaluation)
    knowledge.fail_commit_once = True
    with pytest.raises(RuntimeError, match="injected"):
        first.update_knowledge(session.session_id, verification, evidence_set)
    assert first.status(session.session_id).session.state == SessionState.UPDATING
    assert first.status(session.session_id).latest("knowledge_update") is not None
    first_repository.close()

    resumed_repository = SQLiteInvestigationRepository(path)
    resumed = InvestigationApplication(knowledge, repository=resumed_repository)
    outcome = resumed.resume_iteration(session.session_id, objective_satisfied=True)

    assert outcome.update_result.committed_claim_ids == ("claim-acme-active",)
    assert resumed.status(session.session_id).session.state == SessionState.COMPLETED
    resumed_repository.close()


def test_no_gaps_terminates_without_spawning_agents() -> None:
    app = InvestigationApplication(FakeKnowledge([]))
    session = app.create(objective(), budget())
    planning = app.plan_iteration(session.session_id, worker_capacity=2)

    assert planning.plan is None
    assert planning.termination is not None
    assert app.status(session.session_id).session.state == SessionState.COMPLETED


def test_planning_rejects_modified_copy_of_persisted_snapshot() -> None:
    app = InvestigationApplication(FakeKnowledge())
    session = app.create(objective(), budget())
    snapshot = app.observe(session.session_id)

    with pytest.raises(DomainError, match="not active"):
        app.generate(session.session_id, replace(snapshot, summary="forged summary"))


def test_paused_session_can_be_cancelled() -> None:
    app = InvestigationApplication(FakeKnowledge())
    session = app.create(objective(), budget())
    app.pause(session.session_id)

    cancelled = app.cancel(session.session_id)

    assert cancelled.state == SessionState.CANCELLED
    assert cancelled.termination_reason == TerminationReason.USER_CANCELLATION


def test_cancelled_session_cannot_submit_more_task_waves() -> None:
    knowledge = FakeKnowledge([gap(0), gap(1)])
    simulator = LocalDistributedSimulator(start=NOW)
    runtime = RuntimeApplication(simulator.coordinator)
    app = InvestigationApplication(
        knowledge,
        execution=PlanExecutionController(runtime),
    )
    session = app.create(objective(), budget())
    snapshot = app.observe(session.session_id)
    candidates = app.generate(session.session_id, snapshot)
    scores = app.score(session.session_id, snapshot, candidates)
    selection = app.select(session.session_id, scores, worker_capacity=2)
    ids = [item.candidate.investigation_id for item in selection.selected]
    plan = app.build_plan(session.session_id, selection, dependencies={ids[1]: (ids[0],)})
    app.start_execution(
        session.session_id,
        plan,
        {ids[0]: "run-cancel-root", ids[1]: "run-cancel-child"},
    )
    app.cancel(session.session_id)

    with pytest.raises(DomainError, match="only while EXECUTING"):
        app.advance_execution(session.session_id)

    assert len(runtime.list_tasks()) == 1


class QueueModel:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)

    def complete(self, prompt: str, response_schema: dict[str, object]) -> object:
        return self.responses.pop(0)


class Memory:
    def recall(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        return []


class SearchTool:
    name = "search"
    description = "mock search"
    capability = "search.execute"
    input_schema = {"required": ["query"]}
    output_schema = {"required": ["results"]}
    permissions = frozenset({"search.execute"})
    timeout_seconds = 1.0
    side_effect = "none"
    idempotency = "keyed"

    def execute(self, input: dict[str, object], idempotency_key: str | None) -> dict[str, object]:
        return {"results": [input["query"]]}


def test_agent_harness_runs_production_distributed_investigation() -> None:
    knowledge = FakeKnowledge()
    simulator = LocalDistributedSimulator(start=NOW)
    results = InMemoryInvestigationResultRepository()
    agent = Agent("Researcher", "Researcher", frozenset({"search.execute"}))
    registry = ToolRegistry(PolicyEngine({agent.agent_id: frozenset({"search.execute"})}))
    registry.register(SearchTool())
    executor = AgentExecutor(
        QueueModel(
            [
                {
                    "action": "tool",
                    "tool": "search",
                    "input": {
                        "query": "acme",
                        "document_id": "doc-acme",
                        "chunk_id": "chunk-acme",
                    },
                },
                {"action": "finish", "output": {"summary": "verified"}},
            ]
        ),
        Memory(),
        registry,
    )
    harness = AgentHarness(
        executor,
        results,
        agent,
        budget=Budget(100, timedelta(minutes=1), 5, 1, 1),
    )
    simulator.add_worker(
        "agent-worker",
        frozenset({"document_analysis", "search"}),
        harness,
    )
    runtime = RuntimeApplication(simulator.coordinator)
    app = InvestigationApplication(
        knowledge,
        execution=PlanExecutionController(runtime),
        result_collector=RuntimeResultCollector(runtime, results),
    )
    session = app.create(objective(), budget())
    planning = app.plan_iteration(session.session_id, worker_capacity=1)
    assert planning.plan is not None
    investigation_id = planning.plan.investigations[0].investigation_id
    started = app.start_execution(
        session.session_id,
        planning.plan,
        {investigation_id: "agent-run-prod"},
    )
    task_id = started.execution.task_ids[investigation_id]

    report = simulator.run_until_terminal()
    collected = app.collect_execution_results(session.session_id)

    assert report.succeeded == 1
    assert len(collected) == 1
    result = collected[0]
    assert result.state == InvestigationResultState.COMPLETED
    assert result.run_id == "agent-run-prod"
    assert result.investigation_id == investigation_id
    assert result.task_id == task_id
    assert len(result.evidence_set.evidence) == 1
    evidence = result.evidence_set.evidence[0]
    assert evidence.provenance.investigation_id == investigation_id
    assert evidence.provenance.run_id == "agent-run-prod"
    assert evidence.provenance.task_id == result.task_id
    assert evidence.provenance.attempt_id == result.attempt_id
    assert evidence.provenance.document_id == "doc-acme"
    assert evidence.provenance.chunk_id == "chunk-acme"
    assert evidence.source == "tool://search"

    outcome = app.process_results(
        session.session_id,
        collected,
        objective_satisfied=True,
    )
    assert outcome.termination.terminate
    assert app.status(session.session_id).session.state == SessionState.COMPLETED
    assert app.status(session.session_id).session.termination_reason == (
        TerminationReason.OBJECTIVE_SATISFIED
    )
