"""Provider contracts. The runtime never reaches through these interfaces."""

from __future__ import annotations

from typing import Any, Protocol


class ModelProvider(Protocol):
    def complete(self, prompt: str, response_schema: dict[str, Any]) -> dict[str, Any]: ...


class Tool(Protocol):
    name: str
    description: str
    capability: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    permissions: frozenset[str]
    timeout_seconds: float
    side_effect: str
    idempotency: str

    def execute(self, input: dict[str, Any], idempotency_key: str | None) -> dict[str, Any]: ...


class MemoryProvider(Protocol):
    def recall(self, query: str, limit: int = 10) -> list[dict[str, Any]]: ...

    def remember(self, record: dict[str, Any]) -> str: ...
