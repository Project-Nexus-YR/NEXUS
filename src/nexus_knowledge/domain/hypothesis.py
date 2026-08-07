"""Hypotheses, experiments, results and observations.

These objects model the *investigation* side of the platform: a
hypothesis is proposed, tested by experiments, and results are captured
as observations that feed back into the knowledge graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import now_iso
from .ids import new_id

__all__ = [
    "Hypothesis",
    "HypothesisStatus",
    "Experiment",
    "ExperimentStatus",
    "Result",
    "Observation",
]


class HypothesisStatus:
    PROPOSED = "proposed"
    UNDER_INVESTIGATION = "under_investigation"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    RETIRED = "retired"


@dataclass(slots=True)
class Hypothesis:
    """A falsifiable statement proposed for investigation."""

    statement: str
    claim_ids: list[str] = field(default_factory=list)
    status: str = HypothesisStatus.PROPOSED
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("hyp"))
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)


class ExperimentStatus:
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass(slots=True)
class Experiment:
    """A planned investigation targeting one or more hypotheses."""

    hypothesis_id: str
    design: str
    status: str = ExperimentStatus.PLANNED
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("exp"))
    created_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class Result:
    """The outcome of an experiment."""

    experiment_id: str
    outcome: str
    observation_ids: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("res"))
    created_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class Observation:
    """A single recorded observation anchored to a source."""

    source_id: str
    kind: str
    payload: dict[str, Any]
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("obs"))

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = now_iso()
