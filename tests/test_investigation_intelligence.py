"""Focused tests for Track A autonomous-investigation intelligence."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from nexus_knowledge.domain.knowledge_gap import GapKind, Investigation, KnowledgeGap
from nexus_runtime.investigation.generator import CandidateInvestigation, InvestigationGenerator
from nexus_runtime.investigation.objective import ResearchObjective
from nexus_runtime.investigation.scoring import InvestigationScore, InvestigationScoringModel
from nexus_runtime.investigation.selector import InvestigationSelector
from nexus_runtime.investigation.session import (
    InvestigationBudget,
    InvestigationSession,
    InvestigationUsage,
    SessionState,
    TerminationReason,
)
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
