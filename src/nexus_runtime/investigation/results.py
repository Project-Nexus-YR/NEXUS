"""Resolve distributed result references into structured investigation results."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Protocol

from nexus_runtime.distributed.model import DistributedTaskState
from nexus_runtime.models import DomainError

from .evidence import EvidenceSet, InvestigationResult, InvestigationResultState
from .execution import DistributedRuntimePort, PlanExecution


class InvestigationResultRepository(Protocol):
    def save(self, result: InvestigationResult) -> str: ...

    def get(self, result_ref: str) -> InvestigationResult | None: ...


class InMemoryInvestigationResultRepository:
    """Reference adapter used by harness implementations and deterministic tests."""

    def __init__(self) -> None:
        self._results: dict[str, InvestigationResult] = {}

    def save(self, result: InvestigationResult) -> str:
        result_ref = f"investigation-result://{result.result_id}"
        self._results[result_ref] = result
        return result_ref

    def get(self, result_ref: str) -> InvestigationResult | None:
        return self._results.get(result_ref)


class SQLiteInvestigationResultRepository:
    """Durable result-reference adapter for worker/application process separation."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS investigation_results (
                    result_ref TEXT PRIMARY KEY,
                    result_id TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    completed_at TEXT NOT NULL
                )
                """
            )

    def save(self, result: InvestigationResult) -> str:
        result_ref = f"investigation-result://{result.result_id}"
        payload = json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO investigation_results(
                    result_ref, result_id, payload_json, completed_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(result_ref) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    completed_at = excluded.completed_at
                """,
                (result_ref, result.result_id, payload, result.completed_at.isoformat()),
            )
        return result_ref

    def get(self, result_ref: str) -> InvestigationResult | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json FROM investigation_results WHERE result_ref = ?",
                (result_ref,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        if not isinstance(payload, dict):
            raise DomainError("persisted investigation result is malformed")
        try:
            return InvestigationResult.from_dict(payload)
        except ValueError as exc:
            raise DomainError("persisted investigation result is malformed") from exc

    def close(self) -> None:
        self._connection.close()


class RuntimeResultCollector:
    """Collect terminal task outcomes without reading coordinator internals."""

    def __init__(
        self,
        runtime: DistributedRuntimePort,
        results: InvestigationResultRepository,
    ) -> None:
        self._runtime = runtime
        self._results = results

    def collect(self, execution: PlanExecution) -> tuple[InvestigationResult, ...]:
        collected: list[InvestigationResult] = []
        for investigation_id, task_id in sorted(execution.task_ids.items()):
            task = self._runtime.get_task(task_id)
            if task.state == DistributedTaskState.SUCCEEDED:
                if task.result_ref is None:
                    raise DomainError(f"successful task has no result reference: {task_id}")
                result = self._results.get(task.result_ref)
                if result is None:
                    raise DomainError(f"unknown investigation result reference: {task.result_ref}")
                if (
                    result.session_id != execution.session_id
                    or result.investigation_id != investigation_id
                    or result.task_id != task_id
                    or result.run_id != task.run_id
                ):
                    raise DomainError("distributed result lineage does not match its task")
                if not task.attempts or result.attempt_id != task.attempts[-1].attempt_id:
                    raise DomainError("distributed result attempt does not match successful task")
                collected.append(result)
                continue
            if task.state not in {
                DistributedTaskState.CANCELLED,
                DistributedTaskState.DEAD_LETTERED,
            }:
                continue
            attempt_id = task.attempts[-1].attempt_id if task.attempts else f"unattempted:{task_id}"
            state = (
                InvestigationResultState.CANCELLED
                if task.state == DistributedTaskState.CANCELLED
                else InvestigationResultState.FAILED
            )
            error = task.last_error or task.state.value.lower()
            collected.append(
                InvestigationResult(
                    session_id=execution.session_id,
                    investigation_id=investigation_id,
                    task_id=task_id,
                    attempt_id=attempt_id,
                    run_id=task.run_id,
                    state=state,
                    evidence_set=EvidenceSet(
                        session_id=execution.session_id,
                        evidence_set_id=_stable_id("evidence_set", task_id, state.value),
                        created_at=task.updated_at,
                    ),
                    error=error if state == InvestigationResultState.FAILED else None,
                    result_id=_stable_id("investigation_result", task_id, state.value),
                    completed_at=task.updated_at,
                )
            )
        for investigation_id, reason in sorted(execution.blocked_investigations.items()):
            collected.append(
                InvestigationResult(
                    session_id=execution.session_id,
                    investigation_id=investigation_id,
                    task_id=f"blocked:{execution.plan_id}:{investigation_id}",
                    attempt_id="not-scheduled",
                    run_id=execution.run_ids[investigation_id],
                    state=InvestigationResultState.FAILED,
                    evidence_set=EvidenceSet(
                        session_id=execution.session_id,
                        evidence_set_id=_stable_id(
                            "evidence_set", execution.plan_id, investigation_id
                        ),
                        created_at=execution.updated_at,
                    ),
                    error=reason,
                    metadata={"blocked": True},
                    result_id=_stable_id(
                        "investigation_result", execution.plan_id, investigation_id
                    ),
                    completed_at=execution.updated_at,
                )
            )
        return tuple(collected)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"
