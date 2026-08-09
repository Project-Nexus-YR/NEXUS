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

    def test_delegation_builds_hierarchical_child_run(self) -> None:
        executor, parent_agent = self._executor({"action": "finish", "output": {"summary": "p"}})
        delegated = Agent("Analyst", "Analyst", frozenset({"search.execute"}))
        parent = next(iter(executor._runs.values()))
        budget = Budget(50, timedelta(minutes=1), 1, 1, 1)

        child = executor.build_delegation(parent.run_id, delegated, "analyze the evidence", budget)

        self.assertIsNotNone(child.delegation_id)
        self.assertEqual(child.parent_run_id, parent.run_id)
        self.assertEqual(child.root_run_id, parent.run_id)
        self.assertEqual(child.depth, 1)
        self.assertEqual(child.outputs["delegation_task"], "analyze the evidence")
        self.assertIn(child.delegation_id, parent.attached_delegations)
        self.assertEqual(
            executor.get_delegation_agent(child.delegation_id).agent_id, delegated.agent_id
        )

    def test_delegation_depth_guard(self) -> None:
        executor, _ = self._executor({"action": "finish", "output": {}})
        delegated = Agent("Analyst", "Analyst", frozenset({"search.execute"}))
        parent = next(iter(executor._runs.values()))
        budget = Budget(50, timedelta(minutes=1), 1, 1, 1)

        with self.assertRaisesRegex(DomainError, "depth"):
            executor.build_delegation(parent.run_id, delegated, "too deep", budget, max_depth=0)

    def test_checkpoint_resume_restores_child_hierarchy(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from nexus_runtime.persistence import SQLiteStateStore

        parent_agent = Agent("Researcher", "Researcher", frozenset({"search.execute"}))
        delegated = Agent("Analyst", "Analyst", frozenset({"search.execute"}))
        task = "analyze the evidence"

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runtime.sqlite"
            store = SQLiteStateStore(path)
            resumed_store: SQLiteStateStore | None = None
            try:
                registry = ToolRegistry(
                    PolicyEngine(
                        {
                            parent_agent.agent_id: frozenset({"search.execute"}),
                            delegated.agent_id: frozenset({"search.execute"}),
                        }
                    )
                )
                registry.register(SearchTool())
                executor = AgentExecutor(Model({}), Memory(), registry, state_store=store)
                parent = executor.create_run(
                    parent_agent, "investigation-1", Budget(100, timedelta(minutes=1), 2, 1, 1)
                )
                executor.transition(parent.run_id, AgentRunState.RUNNING, "test")
                child = executor.build_delegation(
                    parent.run_id, delegated, task, Budget(50, timedelta(minutes=1), 1, 1, 1)
                )
                child_run_id = child.run_id
                delegation_id = child.delegation_id

                resumed_store = SQLiteStateStore(path)
                restarted = AgentExecutor(Model({}), Memory(), registry, state_store=resumed_store)
                restored = restarted.resume(child_run_id, subagent_resume=True)

                self.assertEqual(restored.agent_id, delegated.agent_id)
                self.assertEqual(restored.delegation_id, delegation_id)
                self.assertEqual(restored.parent_run_id, parent.run_id)
                self.assertEqual(restored.depth, 1)
                self.assertEqual(restored.outputs["delegation_task"], task)
                restored_parent = restarted.get_run(parent.run_id)
                self.assertEqual(restored_parent.agent_id, parent_agent.agent_id)
                self.assertIn(delegation_id, restored_parent.attached_delegations)
                self.assertEqual(
                    restarted.get_delegation_agent(delegation_id).agent_id, delegated.agent_id
                )
            finally:
                store.close()
                if resumed_store is not None:
                    resumed_store.close()

    def test_resumed_child_executes_tool_under_own_agent(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from nexus_runtime.persistence import SQLiteStateStore

        parent_agent = Agent("Researcher", "Researcher", frozenset({"search.execute"}))
        delegated = Agent("Analyst", "Analyst", frozenset({"search.execute"}))

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runtime.sqlite"
            store = SQLiteStateStore(path)
            resumed_store: SQLiteStateStore | None = None
            try:
                registry = ToolRegistry(
                    PolicyEngine(
                        {
                            parent_agent.agent_id: frozenset({"search.execute"}),
                            delegated.agent_id: frozenset({"search.execute"}),
                        }
                    )
                )
                registry.register(SearchTool())
                executor = AgentExecutor(Model({}), Memory(), registry, state_store=store)
                parent = executor.create_run(
                    parent_agent, "investigation-1", Budget(100, timedelta(minutes=1), 2, 1, 1)
                )
                executor.transition(parent.run_id, AgentRunState.RUNNING, "test")
                child = executor.build_delegation(
                    parent.run_id, delegated, "analyze", Budget(50, timedelta(minutes=1), 1, 1, 1)
                )
                child_run_id = child.run_id

                resumed_store = SQLiteStateStore(path)
                restarted = AgentExecutor(Model({}), Memory(), registry, state_store=resumed_store)
                restored = restarted.resume(child_run_id, subagent_resume=True)
                restarted.transition(restored.run_id, AgentRunState.RUNNING, "resume")
                action = {"action": "tool", "tool": "search", "input": {"query": "x"}}

                restarted.execute_action(restored.run_id, action)

                tool_call = restored.tool_calls[-1]
                self.assertEqual(tool_call.tool_name, "search")
                self.assertEqual(tool_call.status, "ALLOW")
                self.assertEqual(restored.agent_id, delegated.agent_id)
            finally:
                store.close()
                if resumed_store is not None:
                    resumed_store.close()
