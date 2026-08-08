"""Bounded, explicit termination decisions for investigation sessions."""

from __future__ import annotations

from dataclasses import dataclass

from nexus_runtime.models import DomainError

from .session import (
    InvestigationSession,
    SessionState,
    TerminationReason,
)


@dataclass(frozen=True, slots=True)
class TerminationContext:
    objective_satisfied: bool
    objective_confidence: float
    remaining_gap_count: int
    best_candidate_score: float | None
    unresolved_contradictions: int = 0
    contradictions_resolvable: bool = True
    cancellation_requested: bool = False
    system_failure: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.objective_confidence <= 1.0:
            raise DomainError("objective_confidence must be between zero and one")
        if self.best_candidate_score is not None and not 0.0 <= self.best_candidate_score <= 1.0:
            raise DomainError("best_candidate_score must be between zero and one")
        if self.remaining_gap_count < 0 or self.unresolved_contradictions < 0:
            raise DomainError("gap and contradiction counts cannot be negative")


@dataclass(frozen=True, slots=True)
class TerminationDecision:
    terminate: bool
    reason: TerminationReason | None
    target_state: SessionState | None
    explanation: str

    def __post_init__(self) -> None:
        if self.terminate != (self.reason is not None and self.target_state is not None):
            raise DomainError("termination decisions require both reason and target state")

    def to_dict(self) -> dict[str, object]:
        return {
            "terminate": self.terminate,
            "reason": None if self.reason is None else self.reason.value,
            "target_state": None if self.target_state is None else self.target_state.value,
            "explanation": self.explanation,
        }


class TerminationPolicy:
    """Evaluate terminal conditions in a deterministic precedence order."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.9,
        minimum_investigation_score: float = 0.05,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise DomainError("confidence_threshold must be between zero and one")
        if not 0.0 <= minimum_investigation_score <= 1.0:
            raise DomainError("minimum_investigation_score must be between zero and one")
        self._confidence_threshold = confidence_threshold
        self._minimum_investigation_score = minimum_investigation_score

    def evaluate(
        self, session: InvestigationSession, context: TerminationContext
    ) -> TerminationDecision:
        if context.cancellation_requested or session.state == SessionState.CANCELLED:
            return self._stop(
                TerminationReason.USER_CANCELLATION,
                SessionState.CANCELLED,
                "the user requested cancellation",
            )
        if context.system_failure or session.state == SessionState.FAILED:
            return self._stop(
                TerminationReason.SYSTEM_FAILURE,
                SessionState.FAILED,
                "a non-recoverable system failure occurred",
            )
        if context.objective_satisfied:
            return self._stop(
                TerminationReason.OBJECTIVE_SATISFIED,
                SessionState.COMPLETED,
                "all objective success criteria are satisfied",
            )
        if (
            context.remaining_gap_count == 0
            and context.objective_confidence >= self._confidence_threshold
        ):
            return self._stop(
                TerminationReason.CONFIDENCE_THRESHOLD_REACHED,
                SessionState.COMPLETED,
                f"objective confidence reached {context.objective_confidence:.3f}",
            )
        if session.iteration >= session.budget.max_iterations:
            return self._stop(
                TerminationReason.MAXIMUM_ITERATIONS_REACHED,
                SessionState.COMPLETED,
                f"maximum of {session.budget.max_iterations} iterations reached",
            )
        if self._budget_exhausted(session):
            return self._stop(
                TerminationReason.BUDGET_EXHAUSTED,
                SessionState.COMPLETED,
                "one or more investigation budget dimensions are exhausted",
            )
        if context.unresolved_contradictions and not context.contradictions_resolvable:
            return self._stop(
                TerminationReason.UNRESOLVABLE_CONTRADICTION,
                SessionState.COMPLETED,
                f"{context.unresolved_contradictions} contradiction(s) remain unresolvable",
            )
        if (
            context.best_candidate_score is None
            or context.best_candidate_score < self._minimum_investigation_score
        ):
            return self._stop(
                TerminationReason.NO_VALUABLE_INVESTIGATION,
                SessionState.COMPLETED,
                "no remaining candidate exceeds the minimum investigation value",
            )
        return TerminationDecision(
            terminate=False,
            reason=None,
            target_state=None,
            explanation=(
                f"continue with {context.remaining_gap_count} gap(s); best candidate score "
                f"is {context.best_candidate_score:.3f}"
            ),
        )

    @staticmethod
    def _budget_exhausted(session: InvestigationSession) -> bool:
        budget = session.budget
        usage = session.usage
        return (
            usage.investigations >= budget.max_investigations
            or usage.agent_runs >= budget.max_agent_runs
            or usage.cost >= budget.max_cost
            or usage.execution_time >= budget.max_execution_time
        )

    @staticmethod
    def _stop(
        reason: TerminationReason,
        target: SessionState,
        explanation: str,
    ) -> TerminationDecision:
        return TerminationDecision(True, reason, target, explanation)
