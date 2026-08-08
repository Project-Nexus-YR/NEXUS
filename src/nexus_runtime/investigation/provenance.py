"""End-to-end lineage for investigation evidence.

The records in this module deliberately contain identifiers rather than
references to runtime objects.  They therefore remain serializable and can be
reconstructed after the coordinator, worker, or agent process restarts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

_LINEAGE_FIELDS = (
    "session_id",
    "investigation_id",
    "task_id",
    "attempt_id",
    "run_id",
    "tool_call_id",
    "source_id",
    "document_id",
    "chunk_id",
)


@dataclass(frozen=True, slots=True)
class EvidenceProvenance:
    """Complete objective-to-source lineage for one evidence item.

    Every correlation identifier is an explicit string.  Evidence without a
    complete chain is rejected at construction time instead of becoming
    anonymous knowledge later in the update pipeline.
    """

    session_id: str
    investigation_id: str
    task_id: str
    attempt_id: str
    run_id: str
    tool_call_id: str
    source_id: str
    document_id: str
    chunk_id: str
    source_reference: str

    def __post_init__(self) -> None:
        values = self.to_dict()
        missing = [
            name
            for name in (*_LINEAGE_FIELDS, "source_reference")
            if not isinstance(values[name], str) or not values[name].strip()
        ]
        if missing:
            raise ValueError(f"incomplete evidence provenance: {', '.join(missing)}")

    @property
    def is_complete(self) -> bool:
        """Return whether all required lineage fields are present."""
        return all(value.strip() for value in self.to_dict().values())

    @property
    def correlation_ids(self) -> tuple[str, ...]:
        """Return runtime correlation identifiers in execution order."""
        return (
            self.session_id,
            self.investigation_id,
            self.task_id,
            self.attempt_id,
            self.run_id,
            self.tool_call_id,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "investigation_id": self.investigation_id,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "tool_call_id": self.tool_call_id,
            "source_id": self.source_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source_reference": self.source_reference,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EvidenceProvenance:
        fields = (*_LINEAGE_FIELDS, "source_reference")
        values: dict[str, str] = {}
        for name in fields:
            value = payload.get(name)
            if not isinstance(value, str):
                raise ValueError(f"malformed evidence provenance field: {name}")
            values[name] = value
        return cls(**values)
