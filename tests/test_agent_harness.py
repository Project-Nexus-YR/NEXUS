from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from nexus_runtime.agent import AgentExecutor
from nexus_runtime.distributed.model import FailureClass
from nexus_runtime.distributed.worker import (
    HarnessExecutionContext,
    HarnessStatus,
)
from nexus_runtime.investigation.agent_harness import AgentHarness
from nexus_runtime.investigation.evidence import InvestigationResultState
from nexus_runtime.investigation.results import InMemoryInvestigationResultRepository
from nexus_runtime.models import Agent, AgentRunState, Budget
from nexus_runtime.persistence import SQLiteStateStore
from nexus_runtime.policy import PolicyEngine
from nexus_runtime.tools import ToolRegistry


class Model:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)

    def complete(self, prompt: str, response_schema: dict[str, object]) -> object:
        response = (
            self.responses.pop(0)
            if self.responses
            else {"action": "finish", "output": {"summary": "done"}}
        )
        if not isinstance(response, dict) or response.get("action") != "finish":
            return response
        observation_id = ""
        for line in prompt.splitlines():
            if line.startswith("[observation:"):
                observation_id = line.split("]", 1)[0][len("[observation:") :]
                break
        if not observation_id:
            return response
        output = dict(response.get("output") or {})
        output.setdefault(
            "conclusions",
            [
                {
                    "claim": {
                        "text": "search returned results",
                        "subject": "results",
                        "predicate": "available",
                        "object": "confirmed",
                        "claim_id": "claim-search-results",
                    },
                    "supporting_observation_ids": [observation_id],
                    "confidence": 0.9,
                    "conclusion_id": "conclusion-search-results",
                }
            ],
        )
        return {"action": "finish", "output": output}

    def recall(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        return []


class CrashingModel:
    def __init__(self, crash_after: int) -> None:
        self.crash_after = crash_after
        self.calls = 0

    def complete(self, prompt: str, response_schema: dict[str, object]) -> object:
        self.calls += 1
        if self.calls > self.crash_after:
            raise RuntimeError("model worker crashed")
        return {"action": "tool", "tool": "search", "input": {"query": "acme"}}


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


def make_context(**metadata: Any) -> HarnessExecutionContext:
    return HarnessExecutionContext(
        run_id="run-1",
        correlation_id="session-1",
        task_id="task-1",
        attempt_id="attempt-1",
        lease_id="lease-1",
        worker_id="worker-1",
        metadata={
            "investigation_id": "investigation-1",
            "question": "What is the answer?",
            **metadata,
        },
    )


def build(
    responses: list[object] | None = None,
    *,
    model: Any | None = None,
    store: SQLiteStateStore | None = None,
    tool_calls: int = 5,
    policy: PolicyEngine | None = None,
    max_steps: int = 8,
):
    agent = Agent("Researcher", "Researcher", frozenset({"search.execute"}))
    engine = policy or PolicyEngine({agent.agent_id: frozenset({"search.execute"})})
    registry = ToolRegistry(engine)
    registry.register(SearchTool())
    executor = AgentExecutor(
        model if model is not None else Model(responses or []),
        Model([]),
        registry,
        state_store=store,
    )
    results = InMemoryInvestigationResultRepository()
    budget = Budget(100, timedelta(minutes=1), tool_calls, 1, 1)
    harness = AgentHarness(executor, results, agent, budget=budget, max_steps=max_steps)
    return harness, executor, results


def test_harness_runs_to_finish_and_persists_result() -> None:
    context = make_context()
    harness, executor, results = build(
        [
            {"action": "tool", "tool": "search", "input": {"query": "acme"}},
            {"action": "finish", "output": {"summary": "done"}},
        ]
    )

    outcome = harness.execute_or_resume(context, lambda: False)

    assert outcome.status == HarnessStatus.SUCCEEDED
    assert outcome.result_ref is not None
    run = executor.get_run(context.run_id)
    assert run.state == AgentRunState.COMPLETED
    assert run.run_id == context.run_id
    assert run.task_id == context.task_id
    result = results.get(outcome.result_ref)
    assert result is not None
    assert result.state == InvestigationResultState.COMPLETED
    assert result.session_id == context.correlation_id
    assert result.investigation_id == "investigation-1"
    assert result.task_id == context.task_id
    assert result.attempt_id == context.attempt_id
    assert result.run_id == context.run_id
    assert len(result.evidence_set.evidence) == 1
    evidence = result.evidence_set.evidence[0]
    assert evidence.source == "tool://search"
    assert evidence.provenance.tool_call_id == run.tool_calls[0].tool_call_id
    assert evidence.provenance.source_reference == evidence.source
    assert evidence.provenance.session_id == context.correlation_id


def test_harness_resumes_checkpointed_run_after_crash() -> None:
    with TemporaryDirectory() as tmp:
        store = SQLiteStateStore(Path(tmp) / "state.db")
        context = make_context()
        harness, _, _ = build(model=CrashingModel(crash_after=1), store=store)
        with pytest.raises(RuntimeError):
            harness.execute_or_resume(context, lambda: False)

        harness2, executor2, results2 = build(
            [{"action": "finish", "output": {"summary": "done"}}],
            store=store,
        )
        outcome = harness2.execute_or_resume(context, lambda: False)

        assert outcome.status == HarnessStatus.SUCCEEDED
        run = executor2.get_run(context.run_id)
        assert run.state == AgentRunState.COMPLETED
        assert len(run.tool_calls) == 1
        result = results2.get(outcome.result_ref)
        assert result is not None
        assert result.state == InvestigationResultState.COMPLETED
        assert len(result.evidence_set.evidence) == 1
        store.close()


def test_harness_cancellation_returns_cancelled_outcome() -> None:
    context = make_context()
    harness, executor, _ = build([{"action": "finish", "output": {}}])

    outcome = harness.execute_or_resume(context, lambda: True)

    assert outcome.status == HarnessStatus.CANCELLED
    assert outcome.checkpoint_ref == f"checkpoint://run/{context.run_id}"
    assert executor.get_run(context.run_id).state == AgentRunState.CANCELLED


def test_harness_budget_exhaustion_maps_to_budget_failure() -> None:
    context = make_context()
    harness, _, _ = build(
        [{"action": "tool", "tool": "search", "input": {"query": "x"}}],
        tool_calls=0,
    )

    outcome = harness.execute_or_resume(context, lambda: False)

    assert outcome.status == HarnessStatus.FAILED
    assert outcome.failure_class == FailureClass.BUDGET_EXHAUSTED
    assert "budget exhausted" in outcome.error


def test_harness_policy_denial_maps_to_policy_violation() -> None:
    harness, _, _ = build(
        [{"action": "tool", "tool": "search", "input": {"query": "x"}}],
        policy=PolicyEngine({}),
    )
    context = make_context()

    outcome = harness.execute_or_resume(context, lambda: False)

    assert outcome.status == HarnessStatus.FAILED
    assert outcome.failure_class == FailureClass.POLICY_VIOLATION
    assert "denied by policy" in outcome.error
    assert outcome.checkpoint_ref is not None


def test_harness_step_limit_fails_permanently() -> None:
    context = make_context()
    harness, executor, _ = build(
        [{"action": "tool", "tool": "search", "input": {"query": "x"}}] * 4,
        max_steps=2,
    )

    outcome = harness.execute_or_resume(context, lambda: False)

    assert outcome.status == HarnessStatus.FAILED
    assert outcome.failure_class == FailureClass.PERMANENT
    assert "paused" in outcome.error
    assert executor.get_run(context.run_id).state == AgentRunState.PAUSED


def test_harness_malformed_model_output_fails_transiently() -> None:
    context = make_context()
    harness, executor, _ = build(["not", "an", "object"])

    outcome = harness.execute_or_resume(context, lambda: False)

    assert outcome.status == HarnessStatus.FAILED
    assert outcome.failure_class == FailureClass.TRANSIENT
    assert executor.get_run(context.run_id).state == AgentRunState.FAILED


def test_harness_cancel_run_best_effort() -> None:
    context = make_context()
    harness, executor, _ = build([{"action": "finish", "output": {}}])
    executor.create_run(
        Agent("Researcher", "Researcher", frozenset({"search.execute"})),
        "investigation-1",
        Budget(100, timedelta(minutes=1), 1, 1, 1),
        task_id=context.task_id,
        run_id="run-other",
    )

    harness.cancel_run("run-other")

    assert executor.get_run("run-other").state == AgentRunState.CANCELLED
