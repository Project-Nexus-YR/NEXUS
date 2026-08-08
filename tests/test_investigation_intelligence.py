"""Focused tests for Track A autonomous-investigation intelligence."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from nexus_runtime.investigation.objective import ResearchObjective
from nexus_runtime.investigation.session import (
    InvestigationBudget,
    InvestigationSession,
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
