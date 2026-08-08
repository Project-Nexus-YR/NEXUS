"""Stable domain records and explicit lifecycle states for the runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import uuid4


def new_id(prefix: str) -> str:
    """Return an opaque, globally unique, stable runtime identifier."""
    return f"{prefix}_{uuid4().hex}"


def utcnow() -> datetime:
    return datetime.now(UTC)


class DomainError(ValueError):
    """A caller supplied an invalid domain operation."""


class InvalidTransition(DomainError):
    """A lifecycle transition is not part of the allowed state machine."""


class AgentRunState(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class Agent:
    name: str
    role: str
    capabilities: frozenset[str]
    preferred_tools: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    instructions: str = ""
    output_schema: str = ""
    agent_id: str = field(default_factory=lambda: new_id("agent"))


@dataclass(frozen=True, slots=True)
class Observation:
    kind: str
    artifact_ref: str | None
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    observation_id: str = field(default_factory=lambda: new_id("obs"))
    created_at: datetime = field(default_factory=utcnow)


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool_name: str
    input: dict[str, Any]
    result_ref: str | None
    status: str
    tool_call_id: str = field(default_factory=lambda: new_id("tool"))
    started_at: datetime = field(default_factory=utcnow)
    completed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Budget:
    """Resource envelope for one AgentRun; child runs may be bounded by this."""

    max_tokens: int
    max_wall_time: timedelta
    max_tool_calls: int
    max_workers: int
    max_experiment_resources: int

    def __post_init__(self) -> None:
        if (
            min(
                self.max_tokens,
                self.max_tool_calls,
                self.max_workers,
                self.max_experiment_resources,
            )
            < 0
        ):
            raise DomainError("budget values cannot be negative")
        if self.max_wall_time <= timedelta(0):
            raise DomainError("wall-time budget must be positive")


@dataclass(frozen=True, slots=True)
class AgentStep:
    phase: str
    input_refs: tuple[str, ...]
    output_ref: str | None
    decision: str
    agent_step_id: str = field(default_factory=lambda: new_id("step"))
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class AgentRun:
    agent_id: str
    investigation_id: str
    task_id: str | None = None
    state: AgentRunState = AgentRunState.CREATED
    context_refs: list[str] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    budget_used: dict[str, int] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: new_id("run"))
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    parent_run_id: str | None = None
    root_run_id: str | None = None
    delegation_id: str | None = None
    depth: int = 0
    attached_delegations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise DomainError("agent run depth cannot be negative")


def _parse_iso(value: object) -> datetime | None:
    return None if value is None else datetime.fromisoformat(str(value))


def _observation_from_dict(value: dict[str, Any]) -> Observation:
    return Observation(
        kind=str(value["kind"]),
        artifact_ref=value.get("artifact_ref"),
        summary=str(value["summary"]),
        data=dict(value.get("data", {})),
        observation_id=str(value["observation_id"]),
        created_at=_parse_iso(value.get("created_at")) or utcnow(),
    )


def _tool_call_from_dict(value: dict[str, Any]) -> ToolCall:
    return ToolCall(
        tool_name=str(value["tool_name"]),
        input=dict(value.get("input", {})),
        result_ref=value.get("result_ref"),
        status=str(value["status"]),
        tool_call_id=str(value["tool_call_id"]),
        started_at=_parse_iso(value.get("started_at")) or utcnow(),
        completed_at=_parse_iso(value.get("completed_at")),
    )


def _agent_step_from_dict(value: dict[str, Any]) -> AgentStep:
    refs = value.get("input_refs", [])
    return AgentStep(
        phase=str(value["phase"]),
        input_refs=tuple(str(item) for item in refs),
        output_ref=value.get("output_ref"),
        decision=str(value["decision"]),
        agent_step_id=str(value["agent_step_id"]),
        created_at=_parse_iso(value.get("created_at")) or utcnow(),
    )


def agent_from_dict(value: dict[str, Any]) -> Agent:
    return Agent(
        name=str(value["name"]),
        role=str(value["role"]),
        capabilities=frozenset(str(item) for item in value.get("capabilities", [])),
        preferred_tools=tuple(str(item) for item in value.get("preferred_tools", [])),
        constraints=tuple(str(item) for item in value.get("constraints", [])),
        instructions=str(value.get("instructions", "")),
        output_schema=str(value.get("output_schema", "")),
        agent_id=str(value.get("agent_id") or new_id("agent")),
    )


def _seconds(value: object, default: float) -> timedelta:
    if isinstance(value, dict):
        return timedelta(seconds=float(value.get("seconds", default)))
    if isinstance(value, (int, float)):
        return timedelta(seconds=float(value))
    return timedelta(seconds=default)


def budget_from_dict(value: dict[str, Any] | None) -> Budget:
    payload = value or {}
    return Budget(
        max_tokens=int(payload.get("max_tokens", 0)),
        max_wall_time=_seconds(payload.get("max_wall_time"), 0.0),
        max_tool_calls=int(payload.get("max_tool_calls", 0)),
        max_workers=int(payload.get("max_workers", 0)),
        max_experiment_resources=int(payload.get("max_experiment_resources", 0)),
    )


def budget_to_dict(value: Budget) -> dict[str, Any]:
    return {
        "max_tokens": value.max_tokens,
        "max_wall_time": {"seconds": value.max_wall_time.total_seconds()},
        "max_tool_calls": value.max_tool_calls,
        "max_workers": value.max_workers,
        "max_experiment_resources": value.max_experiment_resources,
    }


def run_from_checkpoint(payload: dict[str, Any]) -> AgentRun:
    """Restore an AgentRun persisted by the reference StateStore checkpoint adapter."""
    raw = payload.get("run")
    if not isinstance(raw, dict):
        raise DomainError("agent run checkpoint is malformed")
    observations = [
        _observation_from_dict(item)
        for item in raw.get("observations", [])
        if isinstance(item, dict)
    ]
    tool_calls = [
        _tool_call_from_dict(item)
        for item in raw.get("tool_calls", [])
        if isinstance(item, dict)
    ]
    steps = [
        _agent_step_from_dict(item) for item in raw.get("steps", []) if isinstance(item, dict)
    ]
    return AgentRun(
        agent_id=str(raw["agent_id"]),
        investigation_id=str(raw["investigation_id"]),
        task_id=raw.get("task_id"),
        state=AgentRunState(str(raw.get("state", AgentRunState.CREATED.value))),
        context_refs=[str(item) for item in raw.get("context_refs", [])],
        observations=observations,
        tool_calls=tool_calls,
        steps=steps,
        outputs=dict(raw.get("outputs", {})),
        budget_used={
            str(key): int(value) for key, value in dict(raw.get("budget_used", {})).items()
        },
        run_id=str(raw.get("run_id") or new_id("run")),
        created_at=_parse_iso(raw.get("created_at")) or utcnow(),
        updated_at=_parse_iso(raw.get("updated_at")) or utcnow(),
        parent_run_id=raw.get("parent_run_id"),
        root_run_id=raw.get("root_run_id"),
        delegation_id=raw.get("delegation_id"),
        depth=int(raw.get("depth", 0)),
        attached_delegations=[str(item) for item in raw.get("attached_delegations", [])],
    )
