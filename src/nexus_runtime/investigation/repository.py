"""Durable, provider-neutral records for reconstructing investigation sessions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from nexus_runtime.models import DomainError, new_id, utcnow

from .objective import (
    ResearchObjective,
    _required_string,
    _timestamp_from_text,
    _timestamp_to_text,
)
from .session import InvestigationSession


@dataclass(frozen=True, slots=True)
class InvestigationArtifact:
    """One immutable, serializable decision or observation in a session."""

    kind: str
    iteration: int
    payload: dict[str, Any]
    artifact_id: str = field(default_factory=lambda: new_id("investigation_artifact"))
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if not self.kind.strip() or not self.artifact_id.strip():
            raise DomainError("artifact kind and identifier are required")
        if self.iteration < 0:
            raise DomainError("artifact iteration cannot be negative")
        try:
            json.dumps(self.payload, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DomainError("artifact payload must be JSON serializable") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "iteration": self.iteration,
            "payload": dict(self.payload),
            "created_at": _timestamp_to_text(self.created_at),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> InvestigationArtifact:
        body = payload.get("payload")
        if not isinstance(body, dict):
            raise DomainError("malformed InvestigationArtifact")
        iteration = payload.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise DomainError("malformed InvestigationArtifact")
        try:
            return cls(
                artifact_id=_required_string(payload["artifact_id"], "artifact_id"),
                kind=_required_string(payload["kind"], "artifact kind"),
                iteration=iteration,
                payload=dict(body),
                created_at=_timestamp_from_text(payload["created_at"], "created_at"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DomainError("malformed InvestigationArtifact") from exc


@dataclass(slots=True)
class InvestigationRecord:
    """Reconstructable objective, session state, and ordered decision history."""

    objective: ResearchObjective
    session: InvestigationSession
    artifacts: list[InvestigationArtifact] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.session.objective_id != self.objective.objective_id:
            raise DomainError("session and objective identifiers do not match")
        artifact_ids = [item.artifact_id for item in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise DomainError("investigation artifact identifiers must be unique")

    def append(self, kind: str, payload: Mapping[str, Any]) -> InvestigationArtifact:
        artifact = InvestigationArtifact(
            kind=kind,
            iteration=self.session.iteration,
            payload=dict(payload),
        )
        self.artifacts.append(artifact)
        return artifact

    def latest(self, kind: str) -> InvestigationArtifact | None:
        return next((item for item in reversed(self.artifacts) if item.kind == kind), None)

    def for_iteration(self, iteration: int) -> tuple[InvestigationArtifact, ...]:
        return tuple(item for item in self.artifacts if item.iteration == iteration)

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective.to_dict(),
            "session": self.session.to_dict(),
            "artifacts": [item.to_dict() for item in self.artifacts],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> InvestigationRecord:
        objective = payload.get("objective")
        session = payload.get("session")
        artifacts = payload.get("artifacts")
        if not isinstance(objective, dict) or not isinstance(session, dict):
            raise DomainError("malformed InvestigationRecord")
        if not isinstance(artifacts, list) or any(not isinstance(item, dict) for item in artifacts):
            raise DomainError("malformed InvestigationRecord artifacts")
        return cls(
            objective=ResearchObjective.from_dict(objective),
            session=InvestigationSession.from_dict(session),
            artifacts=[InvestigationArtifact.from_dict(item) for item in artifacts],
        )


class InvestigationRepository(Protocol):
    def save(self, record: InvestigationRecord) -> None: ...

    def get(self, session_id: str) -> InvestigationRecord | None: ...

    def list(self) -> list[InvestigationRecord]: ...


class InMemoryInvestigationRepository:
    """Deterministic adapter that round-trips JSON to expose persistence bugs."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, object]] = {}

    def save(self, record: InvestigationRecord) -> None:
        self._records[record.session.session_id] = record.to_dict()

    def get(self, session_id: str) -> InvestigationRecord | None:
        payload = self._records.get(session_id)
        return None if payload is None else InvestigationRecord.from_dict(payload)

    def list(self) -> list[InvestigationRecord]:
        return [InvestigationRecord.from_dict(self._records[key]) for key in sorted(self._records)]


class SQLiteInvestigationRepository:
    """Durable reference adapter for session state and replayable artifacts."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS investigation_sessions (
                    session_id TEXT PRIMARY KEY,
                    objective_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def save(self, record: InvestigationRecord) -> None:
        encoded = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO investigation_sessions(
                    session_id, objective_id, state, payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    objective_id = excluded.objective_id,
                    state = excluded.state,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record.session.session_id,
                    record.objective.objective_id,
                    record.session.state.value,
                    encoded,
                    _timestamp_to_text(record.session.updated_at),
                ),
            )

    def get(self, session_id: str) -> InvestigationRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json FROM investigation_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        if not isinstance(payload, dict):
            raise DomainError("persisted investigation record is malformed")
        return InvestigationRecord.from_dict(payload)

    def list(self) -> list[InvestigationRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json FROM investigation_sessions ORDER BY session_id"
            ).fetchall()
        records: list[InvestigationRecord] = []
        for row in rows:
            payload = json.loads(str(row["payload_json"]))
            if not isinstance(payload, dict):
                raise DomainError("persisted investigation record is malformed")
            records.append(InvestigationRecord.from_dict(payload))
        return records

    def close(self) -> None:
        self._connection.close()
