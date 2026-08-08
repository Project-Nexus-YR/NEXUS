"""Provider-independent research objectives for autonomous investigation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from nexus_runtime.models import DomainError, new_id, utcnow


def _validate_timestamp(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DomainError(f"{field_name} must be timezone-aware")


def _timestamp_to_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _timestamp_from_text(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise DomainError(f"{field_name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DomainError(f"{field_name} must be an ISO-8601 string") from exc
    _validate_timestamp(parsed, field_name)
    return parsed.astimezone(UTC)


def _validate_json(value: object, field_name: str) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DomainError(f"{field_name} must be JSON serializable") from exc


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise DomainError(f"{field_name} must be a string")
    return value


def _string_tuple(value: object, field_name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise DomainError(f"{field_name} must be a sequence of strings")
    if any(not isinstance(item, str) for item in value):
        raise DomainError(f"{field_name} must be a sequence of strings")
    result = tuple(item.strip() for item in value)
    if any(not item for item in result):
        raise DomainError(f"{field_name} cannot contain empty values")
    if not allow_empty and not result:
        raise DomainError(f"{field_name} cannot be empty")
    return result


@dataclass(frozen=True, slots=True)
class ResearchObjective:
    """A reproducible statement of what an investigation must establish."""

    question: str
    success_criteria: tuple[str, ...]
    scope: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    objective_id: str = field(default_factory=lambda: new_id("objective"))
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        question = self.question.strip()
        if not question:
            raise DomainError("objective question cannot be empty")
        objective_id = self.objective_id.strip()
        if not objective_id:
            raise DomainError("objective_id cannot be empty")
        scope = _string_tuple(self.scope, "scope")
        constraints = _string_tuple(self.constraints, "constraints")
        success_criteria = _string_tuple(
            self.success_criteria, "success_criteria", allow_empty=False
        )
        _validate_timestamp(self.created_at, "created_at")
        _validate_json(self.metadata, "objective metadata")
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "objective_id", objective_id)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "success_criteria", success_criteria)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""
        return {
            "objective_id": self.objective_id,
            "question": self.question,
            "scope": list(self.scope),
            "constraints": list(self.constraints),
            "success_criteria": list(self.success_criteria),
            "created_at": _timestamp_to_text(self.created_at),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ResearchObjective:
        required = {
            "objective_id",
            "question",
            "scope",
            "constraints",
            "success_criteria",
            "created_at",
            "metadata",
        }
        if set(payload) != required:
            raise DomainError("malformed ResearchObjective")
        metadata = payload["metadata"]
        if not isinstance(metadata, dict):
            raise DomainError("objective metadata must be an object")
        return cls(
            objective_id=_required_string(payload["objective_id"], "objective_id"),
            question=_required_string(payload["question"], "question"),
            scope=_string_tuple(payload["scope"], "scope"),
            constraints=_string_tuple(payload["constraints"], "constraints"),
            success_criteria=_string_tuple(
                payload["success_criteria"], "success_criteria", allow_empty=False
            ),
            created_at=_timestamp_from_text(payload["created_at"], "created_at"),
            metadata=dict(metadata),
        )
