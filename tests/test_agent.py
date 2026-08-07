from __future__ import annotations

import unittest
from datetime import timedelta

from nexus_runtime.agent import AgentExecutor, Budget
from nexus_runtime.models import Agent, AgentRunState, DomainError
from nexus_runtime.policy import PolicyEngine
from nexus_runtime.tools import ToolRegistry


class Model:
    def __init__(self, response: object) -> None:
        self.response = response

    def complete(self, prompt: str, response_schema: dict[str, object]) -> object:
        return self.response


class Memory:
    def recall(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        return [{"id": "memory-1", "artifact_ref": "artifact://memory/1"}]

    def remember(self, record: dict[str, object]) -> str:
        return "memory-1"


class SearchTool:
    name = "search"
    description = "mock search"
    capability = "search.execute"
    input_schema = {"required": ["query"]}
    output_schema = {"required": ["results"]}
    permissions = frozenset({"search.execute"})
    timeout_seconds = 1.0
    side_effect = "none"
    idempotency = "keyed"

    def execute(self, input: dict[str, object], idempotency_key: str | None) -> dict[str, object]:
        return {"results": [input["query"]]}


class AgentTests(unittest.TestCase):
    def _executor(self, response: object, tool_calls: int = 2) -> tuple[AgentExecutor, Agent]:
        agent = Agent("Researcher", "Researcher", frozenset({"search.execute"}))
        registry = ToolRegistry(PolicyEngine({agent.agent_id: frozenset({"search.execute"})}))
        registry.register(SearchTool())
        executor = AgentExecutor(Model(response), Memory(), registry)
        budget = Budget(100, timedelta(minutes=1), tool_calls, 1, 1)
        run = executor.create_run(agent, "investigation-1", budget)
        executor.transition(run.run_id, AgentRunState.RUNNING, "test")
        return executor, agent

    def test_explicit_loop_finishes_with_structured_output(self) -> None:
        executor, agent = self._executor({"action": "finish", "output": {"summary": "done"}})
        run = next(iter(executor._runs.values()))

        executor.retrieve_context(run.run_id, "question")
        result = executor.run_step(run.run_id, "answer", {"title": "action"})

        self.assertEqual(result.state, AgentRunState.COMPLETED)
        self.assertEqual(result.outputs, {"summary": "done"})
        self.assertGreaterEqual(len(result.steps), 3)
        self.assertEqual(agent.role, "Researcher")

    def test_malformed_model_output_is_rejected(self) -> None:
        executor, _ = self._executor(["not", "an", "object"])
        run = next(iter(executor._runs.values()))
        with self.assertRaisesRegex(DomainError, "malformed"):
            executor.reason(run.run_id, "answer", {})
        self.assertEqual(run.state, AgentRunState.FAILED)

    def test_budget_exhaustion_pauses_before_tool_execution(self) -> None:
        executor, _ = self._executor(
            {"action": "tool", "tool": "search", "input": {"query": "x"}}, tool_calls=0
        )
        run = next(iter(executor._runs.values()))
        with self.assertRaisesRegex(DomainError, "budget exhausted"):
            executor.run_step(run.run_id, "answer", {})
        self.assertEqual(run.state, AgentRunState.PAUSED)
