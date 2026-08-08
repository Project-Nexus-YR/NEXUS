"""Transport-independent application service API."""

from __future__ import annotations

from typing import Any

from .agent import AgentExecutor, Budget
from .models import (
    Agent,
    AgentRunState,
    DomainError,
    Investigation,
    Task,
    utcnow,
)
from .research import ResearchCoordinator
from .scheduler import Scheduler


class RuntimeAPI:
    def __init__(
        self, scheduler: Scheduler, agents: AgentExecutor, research: ResearchCoordinator
    ) -> None:
        self._scheduler = scheduler
        self._agents = agents
        self._research = research
        self._investigations: dict[str, Investigation] = {}
        self._agents_by_id: dict[str, Agent] = {}

    def create_run(
        self, agent_id: str, investigation_id: str, budget: Budget, task_id: str | None = None
    ) -> str:
        agent = self._agents_by_id[agent_id]
        return self._agents.create_run(agent, investigation_id, budget, task_id).run_id

    def delegate(
        self,
        parent_run_id: str,
        agent_id: str,
        task: str,
        budget: Budget,
        *,
        max_depth: int = 4,
    ) -> str:
        agent = self._agents_by_id[agent_id]
        child = self._agents.build_delegation(
            parent_run_id, agent, task, budget, max_depth=max_depth
        )
        return child.run_id

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        checkpoint = self._agents.restore_checkpoint(run_id)
        return checkpoint

    def start_run(self, run_id: str) -> None:
        self._agents.transition(run_id, AgentRunState.RUNNING, "started via API")

    def resume_run(
        self,
        run_id: str,
        *,
        subagent_resume: bool = False,
        max_depth: int = 4,
    ) -> dict[str, Any]:
        run = self._agents.resume(
            run_id, subagent_resume=subagent_resume, max_depth=max_depth
        )
        return {
            "run_id": run.run_id,
            "agent_id": run.agent_id,
            "state": run.state.value,
            "parent_run_id": run.parent_run_id,
            "root_run_id": run.root_run_id,
            "delegation_id": run.delegation_id,
            "depth": run.depth,
        }

    def attach_delegation_result(
        self, parent_run_id: str, delegation_id: str, outputs: dict[str, Any]
    ) -> None:
        parent = self._agents.get_run(parent_run_id)
        agent = self._agents.get_delegation_agent(delegation_id)
        parent.outputs.setdefault("delegations", {})[delegation_id] = {
            "agent_id": agent.agent_id,
            "outputs": outputs,
        }
        parent.updated_at = utcnow()
        if parent.state == AgentRunState.RUNNING:
            self._agents.checkpoint(parent_run_id)
            return
        self._agents.transition(parent_run_id, AgentRunState.RUNNING, "delegation completed")

    def cancel_run(self, run_id: str) -> None:
        self._agents.transition(run_id, AgentRunState.CANCELLED, "cancelled via API")

    def get_tasks(self) -> list[Task]:
        return list(self._scheduler.tasks.values())

    def get_events(self) -> list[Any]:
        bus = self._scheduler._bus
        return list(getattr(bus, "events", []))

    def create_agent(self, agent: Agent) -> Agent:
        if agent.agent_id in self._agents_by_id:
            raise DomainError(f"agent already exists: {agent.agent_id}")
        self._agents_by_id[agent.agent_id] = agent
        return agent

    def list_agents(self) -> list[Agent]:
        return list(self._agents_by_id.values())

    def create_investigation(self, goal: str, budget: dict[str, int]) -> Investigation:
        investigation = self._research.create_investigation(goal, budget)
        self._investigations[investigation.investigation_id] = investigation
        return investigation

    def get_investigation(self, investigation_id: str) -> Investigation:
        try:
            return self._investigations[investigation_id]
        except KeyError as exc:
            raise DomainError(f"unknown investigation: {investigation_id}") from exc
