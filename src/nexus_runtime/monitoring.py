"""Read-only runtime and investigation projections for operator interfaces."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from .distributed.service import RuntimeApplication
from .distributed.store import task_to_dict
from .investigation.repository import InvestigationRecord, InvestigationRepository


class RuntimeMonitor:
    """Expose durable investigation and task state without coordinator internals."""

    def __init__(
        self,
        investigations: InvestigationRepository,
        runtime: RuntimeApplication | None = None,
    ) -> None:
        self._investigations = investigations
        self._runtime = runtime

    def list_investigations(self) -> list[dict[str, Any]]:
        records = sorted(
            self._investigations.list(),
            key=lambda item: item.session.updated_at,
            reverse=True,
        )
        tasks = [] if self._runtime is None else self._runtime.list_tasks()
        by_session: dict[str, list[Any]] = {}
        for task in tasks:
            by_session.setdefault(task.correlation_id, []).append(task)
        return [
            self._record_summary(record, by_session.get(record.session.session_id, []))
            for record in records
        ]

    def investigation(self, session_id: str) -> dict[str, Any]:
        record = self._investigations.get(session_id)
        if record is None:
            raise KeyError(session_id)
        tasks = (
            []
            if self._runtime is None
            else [
                task
                for task in self._runtime.list_tasks()
                if task.correlation_id == record.session.session_id
            ]
        )
        summary = self._record_summary(record, tasks)
        return {
            **summary,
            "objective": record.objective.to_dict(),
            "session": record.session.to_dict(),
            "artifacts": [item.to_dict() for item in record.artifacts],
            "tasks": [task_to_dict(item) for item in tasks],
            "timeline": self._timeline(record),
        }

    def runtime(self) -> dict[str, Any]:
        if self._runtime is None:
            return {
                "available": False,
                "queue": {},
                "statistics": {},
                "workers": [],
                "tasks": [],
            }
        workers = [self._worker(item) for item in self._runtime.list_workers()]
        tasks = [task_to_dict(item) for item in self._runtime.list_tasks()]
        return {
            "available": True,
            "queue": self._runtime.get_queue_stats(),
            "statistics": self._runtime.get_runtime_stats(),
            "workers": workers,
            "tasks": tasks,
        }

    def graph_overlay(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for item in self.list_investigations():
            session_id = str(item["session_id"])
            nodes.append(
                {
                    "id": session_id,
                    "kind": "investigation",
                    "label": item["question"],
                    "subtitle": item["phase"],
                    "confidence": None,
                    "uncertainty": None,
                    "importance": None,
                    "verification_state": None,
                    "status": item["state"],
                    "degree": len(item["target_gap_ids"]) + len(item["task_counts"]),
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"],
                    "metadata": {
                        "iteration": item["iteration"],
                        "task_counts": item["task_counts"],
                    },
                }
            )
            for gap_id in item["target_gap_ids"]:
                edges.append(
                    {
                        "id": f"investigation-gap:{session_id}:{gap_id}",
                        "source": session_id,
                        "target": gap_id,
                        "kind": "investigation_target",
                        "label": "investigates",
                        "confidence": None,
                        "verification_state": None,
                        "directed": True,
                        "metadata": {},
                    }
                )
        return nodes, edges

    def version(self) -> str:
        state = {
            "investigations": [
                (item["session_id"], item["updated_at"], item["state"], item["iteration"])
                for item in self.list_investigations()
            ],
            "runtime": self.runtime()["queue"],
        }
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _record_summary(record: InvestigationRecord, tasks: list[Any]) -> dict[str, Any]:
        session = record.session
        task_counts = Counter(task.state.value for task in tasks)
        progress = record.latest("progress_report")
        verification = record.latest("verification")
        evidence = record.latest("evidence_set")
        plan = record.latest("investigation_plan")
        target_gap_ids: set[str] = set()
        explicit_gap = record.objective.metadata.get("target_gap_id")
        if isinstance(explicit_gap, str):
            target_gap_ids.add(explicit_gap)
        if plan is not None:
            investigations = plan.payload.get("investigations", [])
            if isinstance(investigations, list):
                for item in investigations:
                    if isinstance(item, dict) and isinstance(item.get("gap_id"), str):
                        target_gap_ids.add(item["gap_id"])
        return {
            "session_id": session.session_id,
            "objective_id": record.objective.objective_id,
            "question": record.objective.question,
            "state": session.state.value,
            "phase": RuntimeMonitor._phase(session.state.value),
            "iteration": session.iteration,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "termination_reason": (
                None if session.termination_reason is None else session.termination_reason.value
            ),
            "remaining_budget": session.remaining_budget(),
            "task_counts": dict(sorted(task_counts.items())),
            "target_gap_ids": sorted(target_gap_ids),
            "progress": None if progress is None else progress.payload,
            "verification": None if verification is None else verification.payload,
            "evidence": None if evidence is None else evidence.payload,
        }

    @staticmethod
    def _timeline(record: InvestigationRecord) -> list[dict[str, Any]]:
        stages = {
            "knowledge_snapshot": "Planning",
            "investigation_plan": "Task decomposition",
            "plan_execution": "Research / execution",
            "evidence_set": "Evidence collection",
            "verification": "Verification",
            "knowledge_update_result": "Knowledge update",
            "progress_report": "Progress measurement",
            "termination_decision": "Termination decision",
        }
        return [
            {
                "stage": stages[item.kind],
                "kind": item.kind,
                "iteration": item.iteration,
                "at": item.created_at.isoformat(),
                "artifact_id": item.artifact_id,
            }
            for item in record.artifacts
            if item.kind in stages
        ]

    @staticmethod
    def _phase(state: str) -> str:
        return {
            "PLANNING": "Planning",
            "EXECUTING": "Research / execution",
            "EVALUATING": "Evidence / verification",
            "UPDATING": "Knowledge update",
            "COMPLETED": "Complete",
            "PAUSED": "Paused",
            "FAILED": "Failed",
            "CANCELLED": "Cancelled",
        }.get(state, state.replace("_", " ").title())

    @staticmethod
    def _worker(worker: Any) -> dict[str, Any]:
        return {
            "worker_id": worker.identity.worker_id,
            "capabilities": sorted(worker.identity.capabilities),
            "version": worker.version,
            "status": worker.status.value,
            "max_concurrency": worker.max_concurrency,
            "current_tasks": sorted(worker.current_tasks),
            "available_slots": worker.available_slots,
            "started_at": worker.started_at.isoformat(),
            "last_heartbeat": worker.last_heartbeat.isoformat(),
        }
