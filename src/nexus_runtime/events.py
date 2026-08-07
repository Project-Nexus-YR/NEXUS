"""Versioned event contract and a deterministic in-memory EventBus adapter."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from .models import new_id, utcnow

EventHandler = Callable[["Event"], None]


@dataclass(frozen=True, slots=True)
class Event:
    event_type: str
    payload: dict[str, Any]
    producer: str
    trace_id: str
    correlation_id: str
    causation_id: str | None = None
    schema_version: int = 1
    event_id: str = field(default_factory=lambda: new_id("event"))
    timestamp: datetime = field(default_factory=utcnow)

    def topic(self) -> str:
        """Events use their leading namespace as a stable subscription topic."""
        return self.event_type.split(".", maxsplit=1)[0]


class EventBus(Protocol):
    def publish(self, event: Event) -> None: ...

    def subscribe(self, topic: str, handler: EventHandler) -> str: ...

    def acknowledge(self, event_id: str, consumer: str) -> None: ...

    def dead_letter(self, event: Event, reason: str) -> None: ...


class InMemoryEventBus:
    """Synchronous test adapter; production adapters may map this contract to a broker."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.dead_letters: list[tuple[Event, str]] = []
        self.acknowledgements: set[tuple[str, str]] = set()
        self._subscribers: dict[str, dict[str, EventHandler]] = defaultdict(dict)

    def publish(self, event: Event) -> None:
        self.events.append(event)
        for topic in (event.topic(), "*"):
            for handler in tuple(self._subscribers[topic].values()):
                try:
                    handler(event)
                except Exception as exc:  # the event remains inspectable and is DLQ'd
                    self.dead_letter(event, f"subscriber failed: {exc!r}")

    def subscribe(self, topic: str, handler: EventHandler) -> str:
        subscription_id = new_id("subscription")
        self._subscribers[topic][subscription_id] = handler
        return subscription_id

    def acknowledge(self, event_id: str, consumer: str) -> None:
        self.acknowledgements.add((event_id, consumer))

    def dead_letter(self, event: Event, reason: str) -> None:
        self.dead_letters.append((event, reason))
