"""Capability-aware tool registry; tools are adapters, not runtime dependencies."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Any

from .contracts import Tool
from .models import DomainError
from .policy import PolicyDecision, PolicyEngine, PolicyRequest


@dataclass(frozen=True, slots=True)
class ToolResult:
    decision: PolicyDecision
    output: dict[str, Any] | None = None
    reason: str | None = None


class ToolRegistry:
    def __init__(self, policy: PolicyEngine) -> None:
        self._policy = policy
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise DomainError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise DomainError(f"unknown tool: {name}") from exc

    def execute(
        self, principal_id: str, name: str, input: dict[str, Any], idempotency_key: str | None
    ) -> ToolResult:
        tool = self.get(name)
        self._validate_schema(input, tool.input_schema, "input")
        decision = self._policy.decide(
            PolicyRequest(principal_id, tool.capability, tool.side_effect, f"tool:{tool.name}")
        )
        if decision != PolicyDecision.ALLOW:
            return ToolResult(decision, reason=f"policy decision: {decision.value}")
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{tool.name}")
        future = executor.submit(tool.execute, input, idempotency_key)
        try:
            output = future.result(timeout=tool.timeout_seconds)
        except TimeoutError as exc:
            future.cancel()
            raise DomainError(f"tool timed out: {tool.name}") from exc
        except Exception as exc:
            raise DomainError(f"tool execution failed: {tool.name}") from exc
        finally:
            # A timed-out in-process call cannot be force-killed safely. Production
            # adapters should execute in a cancellable process or container.
            executor.shutdown(wait=False, cancel_futures=True)
        self._validate_schema(output, tool.output_schema, "output")
        return ToolResult(PolicyDecision.ALLOW, output=output)

    @staticmethod
    def _validate_schema(value: dict[str, Any], schema: dict[str, Any], label: str) -> None:
        if not isinstance(value, dict):
            raise DomainError(f"tool {label} must be an object")
        required = schema.get("required", [])
        if not isinstance(required, list) or any(key not in value for key in required):
            raise DomainError(f"tool {label} does not match required schema fields")
