"""Focused tests for Track A autonomous-investigation intelligence."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from nexus_knowledge.domain.knowledge_gap import GapKind, Investigation, KnowledgeGap
from nexus_runtime.investigation.generator import CandidateInvestigation, InvestigationGenerator
from nexus_runtime.investigation.objective import ResearchObjective
from nexus_runtime.investigation.observation import KnowledgeSnapshot
from nexus_runtime.investigation.planner import InvestigationPlan, InvestigationPlanner
from nexus_runtime.investigation.scoring import (
    InvestigationScore,
    InvestigationScoringModel,
)
from nexus_runtime.investigation.selector import InvestigationSelector, SelectionResult
from nexus_runtime.investigation.session import (
    InvestigationBudget,
    InvestigationSession,
    InvestigationUsage,
    SessionState,
    TerminationReason,
)
from nexus_runtime.investigation.termination import TerminationContext, TerminationPolicy
from nexus_runtime.models import DomainError, InvalidTransition

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def objective() -> ResearchObjective:
    return ResearchObjective(
        objective_id="objective-1",
        question="Does method X improve metric Y?",
        scope=("controlled benchmarks",),
        constraints=("offline sources only",),
        success_criteria=("two independent sources", "confidence >= 0.8"),
        created_at=NOW,
        metadata={"owner": "research"},
    )


def budget(**overrides: object) -> InvestigationBudget:
    values: dict[str, object] = {
        "max_iterations": 3,
        "max_investigations": 6,
        "max_agent_runs": 12,
        "max_cost": 20.0,
        "max_execution_time": timedelta(minutes=10),
    }
    values.update(overrides)
    return InvestigationBudget(**values)  # type: ignore[arg-type]


def gap(
    gap_id: str = "gap-1",
    *,
    kind: str = GapKind.LOW_CONFIDENCE,
    cost: float = 2.0,
    question: str = "verify claim",
) -> KnowledgeGap:
    item = KnowledgeGap(
        id=gap_id,
        kind=kind,
        description="claim confidence is low",
        reason="confidence below threshold",
        affected_entities=["entity-1"],
        uncertainty=0.8,
        importance=0.7,
        estimated_cost=cost,
        created_at="2026-01-01T00:00:00Z",
    )
    item.candidate_investigations = [
        Investigation(
            id=f"legacy-{gap_id}",
            gap_id=gap_id,
            description=question,
            target_entities=["entity-1"],
            estimated_cost=cost,
            metadata={"estimated_duration_seconds": 30, "evidence_availability": 0.7},
            created_at="2026-01-01T00:00:00Z",
        )
    ]
    return item


def candidate(item: KnowledgeGap | None = None) -> CandidateInvestigation:
    return InvestigationGenerator().generate(objective(), [item or gap()])[0]


class TestResearchObjective:
    def test_round_trip_is_json_serializable(self) -> None:
        original = objective()
        payload = original.to_dict()

        assert ResearchObjective.from_dict(json.loads(json.dumps(payload))) == original
        assert payload["created_at"] == "2026-01-01T00:00:00Z"

    @pytest.mark.parametrize(
        ("question", "criteria"),
        [("", ("enough",)), ("question", ())],
    )
    def test_required_fields_are_validated(self, question: str, criteria: tuple[str, ...]) -> None:
        with pytest.raises(DomainError):
            ResearchObjective(question, criteria, created_at=NOW)

    def test_metadata_must_be_json_serializable(self) -> None:
        with pytest.raises(DomainError, match="JSON serializable"):
            ResearchObjective("question", ("criterion",), metadata={"bad": object()})


class TestInvestigationSession:
    def test_explicit_lifecycle_and_iteration(self) -> None:
        session = InvestigationSession("objective-1", budget(), created_at=NOW, updated_at=NOW)

        session.transition(SessionState.EXECUTING, at=NOW + timedelta(seconds=1))
        session.transition(SessionState.EVALUATING, at=NOW + timedelta(seconds=2))
        session.transition(SessionState.UPDATING, at=NOW + timedelta(seconds=3))
        session.complete_iteration(at=NOW + timedelta(seconds=4))

        assert session.state == SessionState.PLANNING
        assert session.iteration == 1

    def test_invalid_transition_is_rejected(self) -> None:
        session = InvestigationSession("objective-1", budget(), created_at=NOW, updated_at=NOW)
        with pytest.raises(InvalidTransition):
            session.transition(SessionState.EVALUATING, at=NOW)

    def test_pause_resumes_exact_prior_state(self) -> None:
        session = InvestigationSession("objective-1", budget(), created_at=NOW, updated_at=NOW)
        session.transition(SessionState.EXECUTING, at=NOW)
        session.pause(at=NOW)

        assert session.paused_from == SessionState.EXECUTING
        assert session.resume(at=NOW).state == SessionState.EXECUTING

    def test_terminal_transition_requires_matching_reason(self) -> None:
        session = InvestigationSession("objective-1", budget(), created_at=NOW, updated_at=NOW)
        with pytest.raises(DomainError, match="terminal transitions"):
            session.transition(SessionState.COMPLETED, at=NOW)
        session.transition(
            SessionState.COMPLETED,
            at=NOW,
            reason=TerminationReason.OBJECTIVE_SATISFIED,
        )
        assert session.termination_reason == TerminationReason.OBJECTIVE_SATISFIED

    def test_usage_and_session_round_trip(self) -> None:
        session = InvestigationSession("objective-1", budget(), created_at=NOW, updated_at=NOW)
        session.record_usage(
            investigations=2,
            agent_runs=3,
            cost=4.5,
            execution_time=timedelta(seconds=20),
        )

        restored = InvestigationSession.from_dict(json.loads(json.dumps(session.to_dict())))

        assert restored.to_dict() == session.to_dict()
        assert restored.remaining_budget()["investigations"] == 4

    @pytest.mark.parametrize(
        "bad_budget",
        [
            {"max_iterations": 0},
            {"max_investigations": 0},
            {"max_agent_runs": 0},
            {"max_cost": 0},
            {"max_execution_time": timedelta(0)},
        ],
    )
    def test_budget_is_bounded(self, bad_budget: dict[str, object]) -> None:
        with pytest.raises(DomainError):
            budget(**bad_budget)


class TestInvestigationGeneration:
    def test_existing_gap_candidates_are_enriched_deterministically(self) -> None:
        item = gap()
        first = InvestigationGenerator().generate(objective(), [item])
        second = InvestigationGenerator().generate(objective(), [item])

        assert [value.to_dict() for value in first] == [value.to_dict() for value in second]
        assert first[0].gap_id == item.id
        assert first[0].capabilities == ("search", "verification")
        assert first[0].metadata["legacy_investigation_id"] == "legacy-gap-1"

    def test_duplicate_candidates_are_collapsed(self) -> None:
        item = gap()
        duplicate = replace(item.candidate_investigations[0], id="legacy-duplicate")
        item.candidate_investigations.append(duplicate)

        assert len(InvestigationGenerator().generate(objective(), [item])) == 1

    def test_candidate_round_trip(self) -> None:
        original = candidate()
        restored = CandidateInvestigation.from_dict(json.loads(json.dumps(original.to_dict())))
        assert restored == original

    def test_invalid_candidate_is_rejected(self) -> None:
        with pytest.raises(DomainError, match="expected_information_gain"):
            replace(candidate(), expected_information_gain=1.1)


class TestInvestigationScoring:
    def test_cost_and_risk_temper_information_gain(self) -> None:
        item = gap()
        base = candidate(item)
        expensive = replace(
            base,
            investigation_id="expensive",
            estimated_cost=100.0,
            risk=0.9,
        )
        model = InvestigationScoringModel()

        assert model.score(base, item).score > model.score(expensive, item).score

    def test_components_explain_selection_value(self) -> None:
        item = gap()
        result = InvestigationScoringModel().score(candidate(item), item)

        assert set(result.components) == {
            "information_gain",
            "gap_importance",
            "uncertainty_reduction",
            "evidence_availability",
            "priority",
            "knowledge_score",
            "cost_penalty",
            "time_penalty",
            "risk_penalty",
            "redundancy_penalty",
        }
        assert "cost_penalty=" in result.rationale

    def test_unknown_gap_is_rejected(self) -> None:
        with pytest.raises(DomainError, match="unknown knowledge gap"):
            InvestigationScoringModel().score_all([candidate()], [])

    def test_information_gain_forecast_is_cost_normalized(self) -> None:
        item = gap()
        score = InvestigationScoringModel().score(candidate(item), item)
        forecast = InvestigationScoringModel().forecast([score])

        assert forecast.expected_information_gain > 0
        assert forecast.information_gain_per_cost == pytest.approx(
            forecast.expected_information_gain / forecast.estimated_cost
        )


class TestInvestigationSelection:
    def test_top_k_capacity_and_cost_are_enforced(self) -> None:
        gaps = [gap(f"gap-{index}", cost=2.0) for index in range(3)]
        candidates = [candidate(item) for item in gaps]
        scored = InvestigationScoringModel().score_all(candidates, gaps)

        result = InvestigationSelector().select(
            scored,
            budget=budget(max_cost=3.0),
            usage=InvestigationUsage(),
            worker_capacity=3,
            top_k=3,
        )

        assert len(result.selected) == 1
        assert set(result.rejected.values()) == {"cost_budget"}

    def test_redundant_evidence_need_is_not_selected_twice(self) -> None:
        first = candidate()
        second = replace(
            first,
            investigation_id="alternative",
            question="independently verify claim?",
        )
        scores = (
            InvestigationScore(first, 0.8, {}, "first"),
            InvestigationScore(second, 0.7, {}, "second"),
        )

        result = InvestigationSelector().select(
            scores,
            budget=budget(),
            usage=InvestigationUsage(),
            worker_capacity=2,
        )

        assert [item.candidate for item in result.selected] == [first]
        assert result.rejected[second.investigation_id] == "redundant_evidence_need"

    def test_remaining_usage_limits_selection(self) -> None:
        item = candidate()
        result = InvestigationSelector().select(
            [InvestigationScore(item, 0.9, {}, "valuable")],
            budget=budget(max_investigations=1),
            usage=InvestigationUsage(investigations=1),
            worker_capacity=1,
        )
        assert result.selected == ()
        assert result.rejected[item.investigation_id] == "selection_limit"


class TestKnowledgeSnapshot:
    def test_snapshot_consumes_existing_gaps_and_round_trips(self) -> None:
        original = KnowledgeSnapshot.capture(
            objective(),
            [gap()],
            observed_at=NOW,
            retrieval_refs=("chunk-2", "chunk-1"),
            entity_ids=("entity-1",),
            contradiction_ids=("contradiction-1",),
            summary="Current evidence is incomplete.",
        )
        restored = KnowledgeSnapshot.from_dict(json.loads(json.dumps(original.to_dict())))

        assert restored.to_dict() == original.to_dict()
        assert restored.gaps[0].id == "gap-1"
        assert restored.mean_uncertainty == pytest.approx(0.8)
        assert restored.retrieval_refs == ("chunk-1", "chunk-2")

    def test_no_gaps_has_zero_uncertainty(self) -> None:
        snapshot = KnowledgeSnapshot.capture(objective(), [], observed_at=NOW)
        assert snapshot.gaps == ()
        assert snapshot.mean_uncertainty == 0.0


def selection_for(*candidates: CandidateInvestigation) -> SelectionResult:
    scores = tuple(
        InvestigationScore(item, 0.8 - index * 0.1, {}, "selected")
        for index, item in enumerate(candidates)
    )
    return SelectionResult(
        selected=scores,
        rejected={},
        total_cost=sum(item.estimated_cost for item in candidates),
        total_execution_seconds=sum(item.estimated_duration_seconds for item in candidates),
    )


class TestInvestigationPlanning:
    def test_independent_investigations_are_parallelizable(self) -> None:
        first = candidate(gap("gap-a"))
        second = candidate(gap("gap-b"))
        session = InvestigationSession(
            "objective-1", budget(), session_id="session-1", created_at=NOW, updated_at=NOW
        )

        plan = InvestigationPlanner().build(session, selection_for(first, second))
        dag = plan.to_task_dag()

        assert len(dag.ready()) == 2
        assert all(not task.dependencies for task in dag.tasks.values())

    def test_dependencies_compile_to_existing_task_dag(self) -> None:
        first = candidate(gap("gap-a"))
        second = candidate(gap("gap-b"))
        plan = InvestigationPlan(
            session_id="session-1",
            investigations=(first, second),
            dependencies={second.investigation_id: (first.investigation_id,)},
            budget=budget(),
            created_at=NOW,
        )

        dag = plan.to_task_dag()
        second_task = dag.tasks[plan.task_id_for(second.investigation_id)]

        assert second_task.dependencies == {plan.task_id_for(first.investigation_id)}
        assert [task.task_id for task in dag.ready()] == [plan.task_id_for(first.investigation_id)]

    def test_cycles_are_rejected(self) -> None:
        first = candidate(gap("gap-a"))
        second = candidate(gap("gap-b"))
        with pytest.raises(DomainError, match="cycle"):
            InvestigationPlan(
                session_id="session-1",
                investigations=(first, second),
                dependencies={
                    first.investigation_id: (second.investigation_id,),
                    second.investigation_id: (first.investigation_id,),
                },
                budget=budget(),
                created_at=NOW,
            )

    def test_plan_round_trip_and_distributed_contract(self) -> None:
        first = candidate(gap("gap-a"))
        second = candidate(gap("gap-b"))
        plan = InvestigationPlan(
            session_id="session-1",
            investigations=(first, second),
            dependencies={second.investigation_id: (first.investigation_id,)},
            budget=budget(),
            created_at=NOW,
        )
        restored = InvestigationPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        run_ids = {
            first.investigation_id: "run-a",
            second.investigation_id: "run-b",
        }

        tasks = restored.to_distributed_tasks(run_ids)

        assert restored.to_dict() == plan.to_dict()
        assert {task.run_id for task in tasks} == {"run-a", "run-b"}
        assert all(task.correlation_id == "session-1" for task in tasks)
        second_task = next(task for task in tasks if task.run_id == "run-b")
        assert second_task.metadata["dependency_task_ids"] == [
            plan.task_id_for(first.investigation_id)
        ]
        assert second_task.required_capabilities == frozenset(second.capabilities)

    def test_compilation_requires_agent_run_lineage(self) -> None:
        item = candidate()
        plan = InvestigationPlan(
            session_id="session-1",
            investigations=(item,),
            dependencies={},
            budget=budget(),
            created_at=NOW,
        )
        with pytest.raises(DomainError, match="missing AgentRun"):
            plan.to_distributed_tasks({})


class TestTerminationPolicy:
    def session(self, **budget_overrides: object) -> InvestigationSession:
        return InvestigationSession(
            "objective-1",
            budget(**budget_overrides),
            created_at=NOW,
            updated_at=NOW,
        )

    @pytest.mark.parametrize(
        ("context", "reason"),
        [
            (
                TerminationContext(True, 0.7, 0, None),
                TerminationReason.OBJECTIVE_SATISFIED,
            ),
            (
                TerminationContext(False, 0.95, 0, 0.5),
                TerminationReason.CONFIDENCE_THRESHOLD_REACHED,
            ),
            (
                TerminationContext(False, 0.5, 1, None),
                TerminationReason.NO_VALUABLE_INVESTIGATION,
            ),
            (
                TerminationContext(
                    False,
                    0.5,
                    1,
                    0.5,
                    unresolved_contradictions=1,
                    contradictions_resolvable=False,
                ),
                TerminationReason.UNRESOLVABLE_CONTRADICTION,
            ),
            (
                TerminationContext(False, 0.5, 1, 0.5, cancellation_requested=True),
                TerminationReason.USER_CANCELLATION,
            ),
        ],
    )
    def test_explicit_stopping_reasons(
        self, context: TerminationContext, reason: TerminationReason
    ) -> None:
        decision = TerminationPolicy().evaluate(self.session(), context)
        assert decision.terminate
        assert decision.reason == reason

    def test_budget_and_iteration_are_bounded(self) -> None:
        context = TerminationContext(False, 0.4, 2, 0.5)
        session = self.session(max_iterations=1)
        session.iteration = 1
        assert (
            TerminationPolicy().evaluate(session, context).reason
            == TerminationReason.MAXIMUM_ITERATIONS_REACHED
        )

        session = self.session(max_cost=1.0)
        session.record_usage(cost=1.0)
        assert (
            TerminationPolicy().evaluate(session, context).reason
            == TerminationReason.BUDGET_EXHAUSTED
        )

    def test_valuable_work_continues(self) -> None:
        decision = TerminationPolicy().evaluate(
            self.session(), TerminationContext(False, 0.4, 2, 0.5)
        )
        assert not decision.terminate
        assert decision.reason is None

    def test_confidence_does_not_hide_remaining_gaps(self) -> None:
        decision = TerminationPolicy().evaluate(
            self.session(), TerminationContext(False, 0.99, 1, 0.5)
        )
        assert not decision.terminate
