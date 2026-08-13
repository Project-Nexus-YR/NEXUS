"""Small observability port for evidence workflows and audit quality."""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol


class EvidenceMetrics(Protocol):
    def increment(self, name: str, value: int = 1) -> None: ...

    def observe(self, name: str, value: float) -> None: ...


class InMemoryEvidenceMetrics:
    def __init__(self) -> None:
        self.counters: dict[str, int] = defaultdict(int)
        self.observations: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def observe(self, name: str, value: float) -> None:
        self.observations[name].append(value)

    def snapshot(self) -> dict[str, object]:
        return {
            "counters": dict(sorted(self.counters.items())),
            "observations": {
                name: {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
                for name, values in sorted(self.observations.items())
                if values
            },
        }
