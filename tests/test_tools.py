from __future__ import annotations

import time
import unittest

from nexus_runtime.models import DomainError
from nexus_runtime.policy import PolicyEngine
from nexus_runtime.tools import ToolRegistry


class FailingTool:
    name = "failing"
    description = "fails"
    capability = "search.execute"
    input_schema = {"required": ["query"]}
    output_schema = {"required": ["results"]}
    permissions = frozenset({"search.execute"})
    timeout_seconds = 1.0
    side_effect = "none"
    idempotency = "keyed"

    def execute(self, input: dict[str, object], idempotency_key: str | None) -> dict[str, object]:
        raise RuntimeError("remote service unavailable")


class SlowTool(FailingTool):
    name = "slow"
    timeout_seconds = 0.001

    def execute(self, input: dict[str, object], idempotency_key: str | None) -> dict[str, object]:
        time.sleep(0.02)
        return {"results": []}


class ToolTests(unittest.TestCase):
    def _registry(self) -> ToolRegistry:
        return ToolRegistry(PolicyEngine({"agent": frozenset({"search.execute"})}))

    def test_external_tool_failure_is_structured_error(self) -> None:
        registry = self._registry()
        registry.register(FailingTool())
        with self.assertRaisesRegex(DomainError, "tool execution failed"):
            registry.execute("agent", "failing", {"query": "q"}, "tool:1")

    def test_tool_timeout_is_detected(self) -> None:
        registry = self._registry()
        registry.register(SlowTool())
        with self.assertRaisesRegex(DomainError, "tool timed out"):
            registry.execute("agent", "slow", {"query": "q"}, "tool:2")
