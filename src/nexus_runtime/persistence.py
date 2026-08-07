"""Durable local state adapter for transitions, checkpoints, and event replay."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, cast

from .events import Event
from .models import Task


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return {"seconds": value.total_seconds()}
    if is_dataclass(value):
        return asdict(cast(Any, value))
    if isinstance(value, set | frozenset):
        return sorted(value)
    return str(value)


class StateStore(Protocol):
    def record_task(self, task: Task, reason: str) -> None: ...

    def record_event(self, event: Event) -> None: ...

    def save_checkpoint(self, run_id: str, payload: dict[str, Any]) -> int: ...

    def load_checkpoint(self, run_id: str) -> dict[str, Any] | None: ...

    def latest_task_snapshots(self) -> list[dict[str, Any]]: ...


class SQLiteStateStore:
    """SQLite is a durable reference adapter, not a distributed coordination database."""

    def __init__(self, path: str | Path) -> None:
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS task_transitions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    run_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, version)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    correlation_id TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                """
            )

    def record_task(self, task: Task, reason: str) -> None:
        payload = json.dumps(asdict(task), default=_json_default, sort_keys=True)
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO task_transitions(task_id, state, reason, payload_json, recorded_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (task.task_id, task.state.value, reason, payload, task.updated_at.isoformat()),
            )

    def record_event(self, event: Event) -> None:
        payload = json.dumps(event.payload, default=_json_default, sort_keys=True)
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO events(event_id, event_type, payload_json, trace_id, "
                "correlation_id, recorded_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.event_type,
                    payload,
                    event.trace_id,
                    event.correlation_id,
                    event.timestamp.isoformat(),
                ),
            )

    def save_checkpoint(self, run_id: str, payload: dict[str, Any]) -> int:
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 AS next_version "
                "FROM checkpoints WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            version = int(row["next_version"])
            self._connection.execute(
                "INSERT INTO checkpoints(run_id, version, payload_json, recorded_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                (run_id, version, json.dumps(payload, default=_json_default, sort_keys=True)),
            )
            return version

    def load_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json FROM checkpoints WHERE run_id = ? "
                "ORDER BY version DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return None if row is None else json.loads(str(row["payload_json"]))

    def task_history(self, task_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT state, reason, payload_json, recorded_at FROM task_transitions "
                "WHERE task_id = ? ORDER BY sequence",
                (task_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_task_snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT transitions.payload_json
                FROM task_transitions AS transitions
                INNER JOIN (
                    SELECT task_id, MAX(sequence) AS latest_sequence
                    FROM task_transitions GROUP BY task_id
                ) AS latest ON transitions.task_id = latest.task_id
                    AND transitions.sequence = latest.latest_sequence
                ORDER BY transitions.sequence
                """
            ).fetchall()
        return [json.loads(str(row["payload_json"])) for row in rows]

    def close(self) -> None:
        self._connection.close()
