"""Top-k, capacity-aware and budget-aware investigation selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nexus_runtime.models import DomainError

from .scoring import InvestigationScore
from .session import InvestigationBudget, InvestigationUsage


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected: tuple[InvestigationScore, ...]
    rejected: dict[str, str]
    total_cost: float
    total_execution_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": [item.to_dict() for item in self.selected],
            "rejected": dict(self.rejected),
            "total_cost": self.total_cost,
            "total_execution_seconds": self.total_execution_seconds,
        }


class InvestigationSelector:
    """Select valuable non-redundant work without exceeding session limits."""

    def __init__(self, minimum_score: float = 0.0) -> None:
        if not 0.0 <= minimum_score <= 1.0:
            raise DomainError("minimum_score must be between zero and one")
        self._minimum_score = minimum_score

    def select(
        self,
        scored: Sequence[InvestigationScore],
        *,
        budget: InvestigationBudget,
        usage: InvestigationUsage,
        worker_capacity: int,
        top_k: int | None = None,
    ) -> SelectionResult:
        if worker_capacity < 1:
            raise DomainError("worker_capacity must be at least one")
        if top_k is not None and top_k < 1:
            raise DomainError("top_k must be at least one")
        remaining_investigations = max(0, budget.max_investigations - usage.investigations)
        remaining_runs = max(0, budget.max_agent_runs - usage.agent_runs)
        limit = min(
            len(scored),
            worker_capacity,
            remaining_investigations,
            remaining_runs,
            len(scored) if top_k is None else top_k,
        )
        remaining_cost = max(0.0, budget.max_cost - usage.cost)
        remaining_seconds = max(
            0.0, (budget.max_execution_time - usage.execution_time).total_seconds()
        )
        ordered = sorted(scored, key=lambda item: (-item.score, item.candidate.investigation_id))
        selected: list[InvestigationScore] = []
        rejected: dict[str, str] = {}
        redundancy_keys: set[str] = set()
        total_cost = 0.0
        total_seconds = 0.0

        for item in ordered:
            candidate = item.candidate
            reason: str | None = None
            if len(selected) >= limit:
                reason = "selection_limit"
            elif item.score < self._minimum_score:
                reason = "below_minimum_score"
            elif candidate.redundancy_key in redundancy_keys:
                reason = "redundant_evidence_need"
            elif total_cost + candidate.estimated_cost > remaining_cost:
                reason = "cost_budget"
            elif total_seconds + candidate.estimated_duration_seconds > (
                remaining_seconds * worker_capacity
            ):
                reason = "execution_time_budget"

            if reason is not None:
                rejected[candidate.investigation_id] = reason
                continue
            selected.append(item)
            redundancy_keys.add(candidate.redundancy_key)
            total_cost += candidate.estimated_cost
            total_seconds += candidate.estimated_duration_seconds

        return SelectionResult(tuple(selected), rejected, total_cost, total_seconds)
