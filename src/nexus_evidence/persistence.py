"""Workflow checkpoints backed by existing investigation artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from nexus_runtime.investigation.repository import InvestigationRepository
from nexus_runtime.models import DomainError


class EvidenceWorkflowStore(Protocol):
    def save(
        self,
        workflow_id: str,
        session_id: str,
        stage: str,
        payload: Mapping[str, Any],
    ) -> None: ...

    def latest(self, workflow_id: str, stage: str = "completed") -> dict[str, Any] | None: ...


class InMemoryEvidenceWorkflowStore:
    def __init__(self) -> None:
        self._checkpoints: dict[tuple[str, str], dict[str, Any]] = {}

    def save(
        self,
        workflow_id: str,
        session_id: str,
        stage: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._checkpoints[(workflow_id, stage)] = {
            "workflow_id": workflow_id,
            "session_id": session_id,
            "stage": stage,
            "payload": dict(payload),
        }

    def latest(self, workflow_id: str, stage: str = "completed") -> dict[str, Any] | None:
        item = self._checkpoints.get((workflow_id, stage))
        return None if item is None else dict(item["payload"])


class InvestigationArtifactEvidenceStore:
    """Persist evidence state in the same durable record as its investigation."""

    ARTIFACT_KIND = "evidence_workflow"

    def __init__(self, investigations: InvestigationRepository, session_id: str) -> None:
        self._investigations = investigations
        self._session_id = session_id

    def save(
        self,
        workflow_id: str,
        session_id: str,
        stage: str,
        payload: Mapping[str, Any],
    ) -> None:
        if session_id != self._session_id:
            raise DomainError("evidence workflow belongs to another investigation session")
        record = self._investigations.get(session_id)
        if record is None:
            raise DomainError(f"unknown investigation session: {session_id}")
        record.append(
            self.ARTIFACT_KIND,
            {
                "workflow_id": workflow_id,
                "stage": stage,
                "payload": dict(payload),
            },
        )
        self._investigations.save(record)

    def latest(self, workflow_id: str, stage: str = "completed") -> dict[str, Any] | None:
        record = self._investigations.get(self._session_id)
        if record is None:
            return None
        for artifact in reversed(record.artifacts):
            if (
                artifact.kind == self.ARTIFACT_KIND
                and artifact.payload.get("workflow_id") == workflow_id
                and artifact.payload.get("stage") == stage
            ):
                payload = artifact.payload.get("payload")
                if not isinstance(payload, dict):
                    raise DomainError("persisted evidence workflow checkpoint is malformed")
                return dict(payload)
        return None
