"""Validated, serializable investigation plans for distributed wave execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from nexus_runtime.models import DomainError

from .generator import CandidateInvestigation, _stable_id
from .objective import (
    _required_string,
    _timestamp_from_text,
    _timestamp_to_text,
    _validate_timestamp,
)
from .selector import SelectionResult
from .session import InvestigationBudget, InvestigationSession, SessionState


@dataclass(frozen=True, slots=True)
class InvestigationPlan:
    session_id: str
    investigations: tuple[CandidateInvestigation, ...]
    dependencies: dict[str, tuple[str, ...]]
    budget: InvestigationBudget
    created_at: datetime
    plan_id: str = ""

    def __post_init__(self) -> None:
        session_id = self.session_id.strip()
        if not session_id:
            raise DomainError("plan session_id is required")
        if not self.investigations:
            raise DomainError("an investigation plan cannot be empty")
        _validate_timestamp(self.created_at, "created_at")
        investigations = tuple(sorted(self.investigations, key=lambda item: item.investigation_id))
        identifiers = [item.investigation_id for item in investigations]
        if len(identifiers) != len(set(identifiers)):
            raise DomainError("plan investigation identifiers must be unique")
        known = set(identifiers)
        normalized: dict[str, tuple[str, ...]] = {}
        unknown_children = set(self.dependencies) - known
        if unknown_children:
            raise DomainError(
                f"dependencies reference unknown investigations: {sorted(unknown_children)}"
            )
        for investigation_id in identifiers:
            parents = tuple(sorted(set(self.dependencies.get(investigation_id, ()))))
            unknown = set(parents) - known
            if unknown:
                raise DomainError(
                    f"dependencies reference unknown investigations: {sorted(unknown)}"
                )
            if investigation_id in parents:
                raise DomainError("an investigation cannot depend on itself")
            normalized[investigation_id] = parents
        self._validate_acyclic(normalized)
        plan_id = self.plan_id.strip() or _stable_id(
            "plan",
            session_id,
            tuple(identifiers),
            tuple((key, normalized[key]) for key in sorted(normalized)),
            _timestamp_to_text(self.created_at),
        )
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "investigations", investigations)
        object.__setattr__(self, "dependencies", normalized)
        object.__setattr__(self, "plan_id", plan_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "session_id": self.session_id,
            "investigations": [item.to_dict() for item in self.investigations],
            "dependencies": {key: list(value) for key, value in self.dependencies.items()},
            "budget": self.budget.to_dict(),
            "created_at": _timestamp_to_text(self.created_at),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> InvestigationPlan:
        investigations = payload.get("investigations")
        dependencies = payload.get("dependencies")
        budget = payload.get("budget")
        if (
            not isinstance(investigations, list)
            or not isinstance(dependencies, dict)
            or not isinstance(budget, dict)
        ):
            raise DomainError("malformed InvestigationPlan")
        normalized_dependencies: dict[str, tuple[str, ...]] = {}
        for key, value in dependencies.items():
            if (
                not isinstance(key, str)
                or not isinstance(value, list)
                or any(not isinstance(item, str) for item in value)
            ):
                raise DomainError("malformed plan dependencies")
            normalized_dependencies[key] = tuple(value)
        try:
            return cls(
                plan_id=_required_string(payload["plan_id"], "plan_id"),
                session_id=_required_string(payload["session_id"], "session_id"),
                investigations=tuple(
                    CandidateInvestigation.from_dict(item)
                    for item in investigations
                    if isinstance(item, dict)
                ),
                dependencies=normalized_dependencies,
                budget=InvestigationBudget.from_dict(budget),
                created_at=_timestamp_from_text(payload["created_at"], "created_at"),
            )
        except KeyError as exc:
            raise DomainError("malformed InvestigationPlan") from exc

    @staticmethod
    def _validate_acyclic(dependencies: Mapping[str, tuple[str, ...]]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(investigation_id: str) -> None:
            if investigation_id in visiting:
                raise DomainError(f"investigation dependency cycle includes {investigation_id}")
            if investigation_id in visited:
                return
            visiting.add(investigation_id)
            for parent in dependencies[investigation_id]:
                visit(parent)
            visiting.remove(investigation_id)
            visited.add(investigation_id)

        for investigation_id in dependencies:
            visit(investigation_id)


class InvestigationPlanner:
    """Build a serializable plan from an explicit selection decision."""

    def build(
        self,
        session: InvestigationSession,
        selection: SelectionResult,
        *,
        dependencies: Mapping[str, tuple[str, ...]] | None = None,
        created_at: datetime | None = None,
    ) -> InvestigationPlan:
        if session.state != SessionState.PLANNING:
            raise DomainError("investigation plans can be built only while PLANNING")
        investigations = tuple(item.candidate for item in selection.selected)
        if not investigations:
            raise DomainError("cannot build a plan from an empty selection")
        timestamp = session.updated_at if created_at is None else created_at
        return InvestigationPlan(
            session_id=session.session_id,
            investigations=investigations,
            dependencies={} if dependencies is None else dict(dependencies),
            budget=session.budget,
            created_at=timestamp,
        )
