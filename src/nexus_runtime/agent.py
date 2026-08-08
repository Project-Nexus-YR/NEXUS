"""Explicit, checkpointable agent execution loop built from structured phases."""

from __future__ import annotations

from dataclasses import asdict
from enum import StrEnum
from typing import Any

from .contracts import MemoryProvider, ModelProvider
from .events import Event, EventBus, InMemoryEventBus
from .models import (
    Agent,
    AgentRun,
    AgentRunState,
    AgentStep,
    Budget,
    DomainError,
    Observation,
    ToolCall,
    agent_from_dict,
    budget_from_dict,
    budget_to_dict,
    new_id,
    run_from_checkpoint,
    utcnow,
)
from .persistence import StateStore
from .policy import PolicyDecision
from .tools import ToolRegistry


class AgentRole(StrEnum):
    PLANNER = "Planner"
    RESEARCHER = "Researcher"
    EXPERIMENTER = "Experimenter"
    ANALYST = "Analyst"
    CRITIC = "Critic"
    SYNTHESIZER = "Synthesizer"


Budget.__module__ = __name__

__all__ = [
    "AgentExecutor",
    "AgentRole",
    "Budget",
    "budget_from_dict",
    "run_from_checkpoint",
]


class AgentExecutor:
    _TRANSITIONS: dict[AgentRunState, frozenset[AgentRunState]] = {
        AgentRunState.CREATED: frozenset({AgentRunState.RUNNING, AgentRunState.CANCELLED}),
        AgentRunState.RUNNING: frozenset(
            {
                AgentRunState.WAITING,
                AgentRunState.PAUSED,
                AgentRunState.COMPLETED,
                AgentRunState.FAILED,
                AgentRunState.RETRYING,
                AgentRunState.CANCELLED,
            }
        ),
        AgentRunState.WAITING: frozenset({AgentRunState.RUNNING, AgentRunState.CANCELLED}),
        AgentRunState.PAUSED: frozenset({AgentRunState.RUNNING, AgentRunState.CANCELLED}),
        AgentRunState.RETRYING: frozenset(
            {AgentRunState.RUNNING, AgentRunState.FAILED, AgentRunState.CANCELLED}
        ),
        AgentRunState.COMPLETED: frozenset(),
        AgentRunState.FAILED: frozenset(),
        AgentRunState.CANCELLED: frozenset(),
    }

    def __init__(
        self,
        model: ModelProvider,
        memory: MemoryProvider,
        tools: ToolRegistry,
        *,
        event_bus: EventBus | None = None,
        state_store: StateStore | None = None,
    ) -> None:
        self._model = model
        self._memory = memory
        self._tools = tools
        self._bus = event_bus or InMemoryEventBus()
        self._store = state_store
        self._runs: dict[str, AgentRun] = {}
        self._budgets: dict[str, Budget] = {}
        self._agents: dict[str, Agent] = {}
        self._delegation_agents: dict[str, Agent] = {}

    def create_run(
        self, agent: Agent, investigation_id: str, budget: Budget, task_id: str | None = None
    ) -> AgentRun:
        run = AgentRun(agent_id=agent.agent_id, investigation_id=investigation_id, task_id=task_id)
        self._runs[run.run_id] = run
        self._budgets[run.run_id] = budget
        self._agents[agent.agent_id] = agent
        self._emit("agent.run.created", run, {"agent_id": agent.agent_id})
        self.checkpoint(run.run_id)
        return run

    def build_delegation(
        self,
        parent_run_id: str,
        delegated_to: Agent,
        task: str,
        budget: Budget,
        *,
        max_depth: int = 4,
    ) -> AgentRun:
        """Create a child AgentRun whose context_refs seed it with the delegation task.

        The child is persisted as its own checkpoint so that, together with
        ``resume(run_id, subagent_resume=True)``, an interrupted hierarchy can be
        reconstructed and resumed after a crash or restart.
        """
        if max_depth < 0:
            raise DomainError("max delegation depth cannot be negative")
        parent = self._run(parent_run_id)
        self._require_running(parent)
        delegation_id = new_id("delegation")
        depth = parent.depth + 1
        if depth > max_depth:
            raise DomainError(f"delegation depth {depth} exceeds max depth {max_depth}")
        child = AgentRun(
            agent_id=delegated_to.agent_id,
            investigation_id=parent.investigation_id,
            task_id=parent.task_id,
            outputs={"delegation_task": task},
            context_refs=[f"delegation_task:{delegation_id}"],
            parent_run_id=parent.run_id,
            root_run_id=parent.root_run_id or parent.run_id,
            delegation_id=delegation_id,
            depth=depth,
        )
        self._runs[child.run_id] = child
        self._budgets[child.run_id] = budget
        self._agents[delegated_to.agent_id] = delegated_to
        self._delegation_agents[delegation_id] = delegated_to
        parent.attached_delegations.append(delegation_id)
        parent.updated_at = utcnow()
        self._emit(
            "agent.delegation.created",
            parent,
            {
                "delegation_id": delegation_id,
                "child_run_id": child.run_id,
                "task": task,
            },
        )
        self.checkpoint(parent.run_id)
        self.checkpoint(child.run_id)
        return child

    def get_run(self, run_id: str) -> AgentRun:
        return self._run(run_id)

    def get_agent(self, agent_id: str) -> Agent:
        try:
            return self._agents[agent_id]
        except KeyError as exc:
            raise DomainError(f"unknown agent: {agent_id}") from exc

    def get_delegation_agent(self, delegation_id: str) -> Agent:
        try:
            return self._delegation_agents[delegation_id]
        except KeyError as exc:
            raise DomainError(f"unknown delegation: {delegation_id}") from exc

    def observe(self, run_id: str, observation: Observation) -> AgentRun:
        run = self._run(run_id)
        self._require_running(run)
        run.observations.append(observation)
        self._step(run, "observe", (), observation.observation_id, "record observation")
        self._emit(
            "agent.observation.recorded", run, {"observation_id": observation.observation_id}
        )
        return run

    def retrieve_context(self, run_id: str, query: str) -> list[dict[str, Any]]:
        run = self._run(run_id)
        self._require_running(run)
        results = self._memory.recall(query)
        references = [str(item.get("artifact_ref", item.get("id", "inline"))) for item in results]
        run.context_refs.extend(references)
        self._step(run, "retrieve_context", (), None, f"retrieved {len(results)} memories")
        self._emit("agent.context.retrieved", run, {"count": len(results)})
        return results

    def reason(self, run_id: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        run = self._run(run_id)
        self._require_running(run)
        response = self._model.complete(prompt, schema)
        if not isinstance(response, dict):
            self.transition(run_id, AgentRunState.FAILED, "model returned non-object output")
            raise DomainError("malformed agent output")
        self._consume(run, "tokens", int(response.get("token_usage", 0)))
        self._step(run, "reason", tuple(run.context_refs[-10:]), None, "model response received")
        self._emit("agent.reasoned", run, {"schema": schema.get("title", "anonymous")})
        return response

    def choose_action(self, run_id: str, response: dict[str, Any]) -> dict[str, Any]:
        run = self._run(run_id)
        self._require_running(run)
        action = response.get("action")
        if action not in {"tool", "finish", "delegate", "wait"}:
            self.transition(run_id, AgentRunState.FAILED, "unrecognized structured action")
            raise DomainError("malformed agent action")
        if action == "tool" and not isinstance(response.get("tool"), str):
            self.transition(run_id, AgentRunState.FAILED, "tool action lacked tool name")
            raise DomainError("malformed tool action")
        self._step(run, "choose_action", (), None, str(action))
        self._emit("agent.action.chosen", run, {"action": action})
        return response

    def execute_action(self, run_id: str, action: dict[str, Any]) -> dict[str, Any] | None:
        run = self._run(run_id)
        self._require_running(run)
        kind = str(action["action"])
        if kind == "finish":
            run.outputs = dict(action.get("output", {}))
            self._step(run, "execute_action", (), None, "finished")
            self.transition(run_id, AgentRunState.COMPLETED, "agent selected finish")
            return run.outputs
        if kind == "delegate":
            self._step(run, "execute_action", (), None, "delegation requested")
            self.transition(run_id, AgentRunState.WAITING, "awaiting delegated task")
            return None
        if kind == "wait":
            self._step(run, "execute_action", (), None, "waiting")
            self.transition(run_id, AgentRunState.WAITING, "agent selected wait")
            return None
        self._consume(run, "tool_calls", 1)
        tool_name = str(action["tool"])
        tool_input = action.get("input", {})
        if not isinstance(tool_input, dict):
            self.transition(run_id, AgentRunState.FAILED, "tool input was not object")
            raise DomainError("malformed tool input")
        try:
            result = self._tools.execute(
                run.agent_id, tool_name, tool_input, f"{run.run_id}:{len(run.tool_calls)}"
            )
        except DomainError as exc:
            call = ToolCall(tool_name, tool_input, None, "FAILED", completed_at=utcnow())
            run.tool_calls.append(call)
            self._emit("tool.failed", run, {"tool_call_id": call.tool_call_id, "error": str(exc)})
            self.transition(run_id, AgentRunState.FAILED, "tool execution failed")
            raise
        call = ToolCall(tool_name, tool_input, None, result.decision.value, completed_at=utcnow())
        run.tool_calls.append(call)
        self._step(run, "execute_action", (), None, f"tool {tool_name}: {result.decision.value}")
        self._emit(
            "tool.completed",
            run,
            {"tool_call_id": call.tool_call_id, "decision": result.decision.value},
        )
        if result.decision == PolicyDecision.REQUIRE_APPROVAL:
            self.transition(run_id, AgentRunState.WAITING, "tool requires approval")
            return None
        if result.decision == PolicyDecision.DENY:
            self.transition(run_id, AgentRunState.FAILED, "tool denied by policy")
            raise DomainError("tool denied by policy")
        return result.output

    def update_state(self, run_id: str) -> AgentRun:
        run = self._run(run_id)
        self.checkpoint(run_id)
        self._emit("agent.run.checkpointed", run, {})
        return run

    def run_step(self, run_id: str, prompt: str, schema: dict[str, Any]) -> AgentRun:
        """Convenience composition only; each durable loop phase remains separately callable."""
        run = self._run(run_id)
        if run.state == AgentRunState.CREATED:
            self.transition(run_id, AgentRunState.RUNNING, "begin execution")
        response = self.reason(run_id, prompt, schema)
        action = self.choose_action(run_id, response)
        self.execute_action(run_id, action)
        return self.update_state(run_id)

    def transition(self, run_id: str, target: AgentRunState, reason: str) -> AgentRun:
        run = self._run(run_id)
        if target not in self._TRANSITIONS[run.state]:
            raise DomainError(f"invalid agent run transition: {run.state} -> {target}")
        run.state = target
        run.updated_at = utcnow()
        self._emit("agent.run.transitioned", run, {"reason": reason, "state": target.value})
        self.checkpoint(run_id)
        return run

    def checkpoint(self, run_id: str) -> None:
        run = self._run(run_id)
        if self._store:
            self._store.save_checkpoint(
                run_id,
                {
                    "run": asdict(run),
                    "agent": asdict(self.get_agent(run.agent_id)),
                    "budget": budget_to_dict(self._budgets[run.run_id]),
                    "state": run.state.value,
                    "context_refs": run.context_refs,
                    "budget_used": run.budget_used,
                    "checkpointed_at": utcnow().isoformat(),
                },
            )

    def restore_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        return None if self._store is None else self._store.load_checkpoint(run_id)

    def resume(
        self,
        run_id: str,
        *,
        subagent_resume: bool = False,
        max_depth: int = 4,
    ) -> AgentRun:
        """Restore a persisted run in-place.

        With ``subagent_resume=True`` the run is treated as a delegated child: its agent is
        reconstructed from the checkpoint so its tool calls keep the right agent_id, and any
        delegation it spawned is registered on the delegation agent cache.
        """
        if self._store is None:
            if run_id not in self._runs:
                raise DomainError("resume requires a state store for unknown runs")
            existing = self._runs[run_id]
            if subagent_resume and existing.delegation_id:
                self._delegation_agents[existing.delegation_id] = self._agents[existing.agent_id]
            self._emit("agent.run.resumed", existing, {"subagent_resume": subagent_resume})
            return existing
        payload = self._store.load_checkpoint(run_id)
        if payload is None:
            raise DomainError(f"no checkpoint for run: {run_id}")
        raw_agent = payload.get("agent")
        if not isinstance(raw_agent, dict):
            raise DomainError("agent run checkpoint has no agent metadata")
        restored = run_from_checkpoint(payload)
        if restored.run_id != run_id:
            raise DomainError("checkpoint run id does not match requested run")
        agent = agent_from_dict(raw_agent)
        budget = budget_from_dict(payload.get("budget")) or self._budgets.get(run_id)
        if budget is None:
            raise DomainError("agent run checkpoint has no budget")
        self._runs[run_id] = restored
        self._budgets[run_id] = budget
        self._agents[agent.agent_id] = agent
        if subagent_resume:
            delegation_id = restored.delegation_id
            if delegation_id:
                self._delegation_agents[delegation_id] = agent
            self._ensure_tree(restored, max_depth=max_depth)
        self._emit("agent.run.resumed", restored, {"subagent_resume": subagent_resume})
        return restored

    def _ensure_tree(self, child: AgentRun, *, max_depth: int) -> None:
        """Rebuild in-memory state for every ancestor of a restored delegation.

        Walks ``parent_run_id`` links upward, restoring each missing ancestor from its
        checkpoint so the orchestration layer can attach the child's result and resume.
        """
        if max_depth < 0:
            raise DomainError("max delegation depth cannot be negative")
        pending: list[AgentRun] = []
        cursor = child
        while cursor.parent_run_id:
            parent_id = cursor.parent_run_id
            if parent_id in self._runs:
                cursor = self._runs[parent_id]
                continue
            payload = self._store.load_checkpoint(parent_id) if self._store else None
            if payload is None:
                raise DomainError(f"no checkpoint for parent run: {parent_id}")
            raw_agent = payload.get("agent")
            if not isinstance(raw_agent, dict):
                raise DomainError("agent run checkpoint has no agent metadata")
            parent = run_from_checkpoint(payload)
            if parent.run_id != parent_id:
                raise DomainError("checkpoint run id does not match requested run")
            self._runs[parent_id] = parent
            self._budgets[parent_id] = budget_from_dict(payload.get("budget"))
            self._agents[parent.agent_id] = agent_from_dict(raw_agent)
            if parent.delegation_id:
                self._delegation_agents[parent.delegation_id] = self._agents[parent.agent_id]
            pending.append(parent)
            cursor = parent
        for parent in pending:
            self._emit(
                "agent.run.resumed", parent, {"subagent_resume": True, "restored_parent": True}
            )

    def _consume(self, run: AgentRun, key: str, amount: int) -> None:
        if amount < 0:
            raise DomainError("resource usage cannot be negative")
        limits = {"tokens": "max_tokens", "tool_calls": "max_tool_calls"}
        limit_name = limits[key]
        limit = getattr(self._budgets[run.run_id], limit_name)
        used = run.budget_used.get(key, 0) + amount
        run.budget_used[key] = used
        if used > limit:
            self.transition(run.run_id, AgentRunState.PAUSED, f"{key} budget exhausted")
            raise DomainError(f"{key} budget exhausted")

    def _require_running(self, run: AgentRun) -> None:
        if run.state != AgentRunState.RUNNING:
            raise DomainError(f"agent run is not running: {run.state}")
        if utcnow() - run.created_at > self._budgets[run.run_id].max_wall_time:
            self.transition(run.run_id, AgentRunState.PAUSED, "wall-time budget exhausted")
            raise DomainError("wall-time budget exhausted")

    def _step(
        self,
        run: AgentRun,
        phase: str,
        input_refs: tuple[str, ...],
        output_ref: str | None,
        decision: str,
    ) -> None:
        run.steps.append(AgentStep(phase, input_refs, output_ref, decision))
        run.updated_at = utcnow()

    def _emit(self, event_type: str, run: AgentRun, payload: dict[str, Any]) -> None:
        event = Event(
            event_type=event_type,
            payload={"run_id": run.run_id, "investigation_id": run.investigation_id, **payload},
            producer="agent-executor",
            trace_id=run.investigation_id,
            correlation_id=run.run_id,
        )
        self._bus.publish(event)
        if self._store:
            self._store.record_event(event)

    def _run(self, run_id: str) -> AgentRun:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise DomainError(f"unknown agent run: {run_id}") from exc
