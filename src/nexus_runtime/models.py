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


class TaskState(StrEnum):
    CREATED = "CREATED"
    READY = "READY"
    LEASED = "LEASED"
    RUNNING = "RUNNING"
    RETRYING = "RETRYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AgentRunState(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    CANCELLED = "CANCELLED"


class HypothesisState(StrEnum):
    PROPOSED = "PROPOSED"
    PLANNED = "PLANNED"
    INVESTIGATING = "INVESTIGATING"
    EVIDENCE_COLLECTED = "EVIDENCE_COLLECTED"
    CRITIQUE = "CRITIQUE"
    SYNTHESIS = "SYNTHESIS"
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class ExperimentState(StrEnum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff: timedelta = timedelta(seconds=1)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise DomainError("max_attempts must be at least one")


@dataclass(frozen=True, slots=True)
class ResourceRequirements:
    cpu_units: int = 1
    memory_mb: int = 128
    labels: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.cpu_units < 1 or self.memory_mb < 1:
            raise DomainError("resource requirements must be positive")


@dataclass(slots=True)
class Task:
    description: str
    capability: str
    inputs: dict[str, Any] = field(default_factory=dict)
    dependencies: set[str] = field(default_factory=set)
    priority: int = 0
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    timeout: timedelta = timedelta(minutes=5)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    idempotency_key: str | None = None
    task_id: str = field(default_factory=lambda: new_id("task"))
    state: TaskState = TaskState.CREATED
    outputs: dict[str, Any] | None = None
    attempt_count: int = 0
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
    next_attempt_at: datetime | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    parent_task_id: str | None = None

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise DomainError("task description cannot be empty")
        if not self.capability.strip():
            raise DomainError("task capability cannot be empty")
        if self.timeout <= timedelta(0):
            raise DomainError("task timeout must be positive")


@dataclass(slots=True)
class TaskAttempt:
    task_id: str
    number: int
    worker_id: str
    state: TaskState = TaskState.LEASED
    attempt_id: str = field(default_factory=lambda: new_id("attempt"))
    started_at: datetime = field(default_factory=utcnow)
    completed_at: datetime | None = None
    error: str | None = None


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


@dataclass(slots=True)
class Investigation:
    goal: str
    budget: dict[str, int]
    state: str = "CREATED"
    investigation_id: str = field(default_factory=lambda: new_id("investigation"))
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class Hypothesis:
    statement: str
    rationale: str
    expected_evidence: tuple[str, ...]
    confidence: float
    required_investigation: str
    state: HypothesisState = HypothesisState.PROPOSED
    hypothesis_id: str = field(default_factory=lambda: new_id("hypothesis"))

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise DomainError("hypothesis confidence must be between zero and one")


@dataclass(slots=True)
class Experiment:
    hypothesis_id: str
    parameters: dict[str, Any]
    inputs: dict[str, Any]
    resource_budget: dict[str, int]
    metrics: tuple[str, ...]
    reproducibility: dict[str, str]
    state: ExperimentState = ExperimentState.CREATED
    outputs: dict[str, Any] = field(default_factory=dict)
    artifact_refs: list[str] = field(default_factory=list)
    experiment_id: str = field(default_factory=lambda: new_id("experiment"))


@dataclass(frozen=True, slots=True)
class HypothesisProposal:
    statement: str
    rationale: str
    expected_evidence: tuple[str, ...]
    confidence: float
    required_investigation: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HypothesisProposal:
        required = {
            "statement",
            "rationale",
            "expected_evidence",
            "confidence",
            "required_investigation",
        }
        if set(value) != required or not isinstance(value["expected_evidence"], list):
            raise DomainError("malformed HypothesisProposal")
        return cls(
            statement=str(value["statement"]),
            rationale=str(value["rationale"]),
            expected_evidence=tuple(str(item) for item in value["expected_evidence"]),
            confidence=float(value["confidence"]),
            required_investigation=str(value["required_investigation"]),
        )


@dataclass(frozen=True, slots=True)
class KnowledgeUpdateProposal:
    claims: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    confidence: float
    contradictions: tuple[str, ...]
    justification: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> KnowledgeUpdateProposal:
        required = {"claims", "evidence_refs", "confidence", "contradictions", "justification"}
        list_fields = ("claims", "evidence_refs", "contradictions")
        if set(value) != required or any(not isinstance(value[key], list) for key in list_fields):
            raise DomainError("malformed KnowledgeUpdateProposal")
        confidence = float(value["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise DomainError("proposal confidence must be between zero and one")
        return cls(
            claims=tuple(str(item) for item in value["claims"]),
            evidence_refs=tuple(str(item) for item in value["evidence_refs"]),
            confidence=confidence,
            contradictions=tuple(str(item) for item in value["contradictions"]),
            justification=str(value["justification"]),
        )


def task_from_snapshot(snapshot: dict[str, Any]) -> Task:
    """Restore a task persisted by the reference StateStore adapter."""
    resources = snapshot.get("resources", {})
    retry = snapshot.get("retry_policy", {})
    date_fields = ("lease_expires_at", "next_attempt_at", "created_at", "updated_at")
    dates = {
        key: datetime.fromisoformat(value) if isinstance(value := snapshot.get(key), str) else value
        for key in date_fields
    }
    return Task(
        description=str(snapshot["description"]),
        capability=str(snapshot["capability"]),
        inputs=dict(snapshot.get("inputs", {})),
        dependencies=set(snapshot.get("dependencies", [])),
        priority=int(snapshot.get("priority", 0)),
        resources=ResourceRequirements(
            int(resources.get("cpu_units", 1)),
            int(resources.get("memory_mb", 128)),
            frozenset(resources.get("labels", [])),
        ),
        timeout=timedelta(seconds=float(snapshot.get("timeout", {}).get("seconds", 300)))
        if isinstance(snapshot.get("timeout"), dict)
        else timedelta(minutes=5),
        retry_policy=RetryPolicy(
            int(retry.get("max_attempts", 3)),
            timedelta(seconds=float(retry.get("backoff", {}).get("seconds", 1)))
            if isinstance(retry.get("backoff"), dict)
            else timedelta(seconds=1),
        ),
        idempotency_key=snapshot.get("idempotency_key"),
        task_id=str(snapshot["task_id"]),
        state=TaskState(str(snapshot.get("state", TaskState.CREATED))),
        outputs=snapshot.get("outputs"),
        attempt_count=int(snapshot.get("attempt_count", 0)),
        worker_id=snapshot.get("worker_id"),
        lease_expires_at=dates["lease_expires_at"],
        next_attempt_at=dates["next_attempt_at"],
        created_at=dates["created_at"] or utcnow(),
        updated_at=dates["updated_at"] or utcnow(),
        parent_task_id=snapshot.get("parent_task_id"),
    )
