"""Structured in-memory runtime metrics suitable for a future exporter adapter."""

from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Protocol


class MetricsSink(Protocol):
    def increment(self, name: str, value: int = 1) -> None: ...

    def observe(self, name: str, value: float) -> None: ...

    def snapshot(self) -> dict[str, object]: ...


class InMemoryMetrics:
    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._lock = RLock()

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._samples[name].append(value)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            observations = {
                name: {
                    "count": len(values),
                    "sum": sum(values),
                    "average": sum(values) / len(values) if values else 0.0,
                    "max": max(values) if values else 0.0,
                }
                for name, values in self._samples.items()
            }
            return {"counters": dict(self._counters), "observations": observations}
