from __future__ import annotations

import unittest
from datetime import timedelta

from nexus_runtime.agent import AgentExecutor, Budget
from nexus_runtime.models import (
    Agent,
    Experiment,
    HypothesisProposal,
    HypothesisState,
    KnowledgeUpdateProposal,
    Task,
)
from nexus_runtime.policy import PolicyEngine
from nexus_runtime.research import ExperimentExecutor, HypothesisLifecycle, ResearchCoordinator
from nexus_runtime.scheduler import Scheduler
from nexus_runtime.tools import ToolRegistry
from nexus_runtime.worker import WorkerProcess


class MockKnowledge:
    def find_knowledge_gaps(self, goal: str) -> list[dict[str, str]]:
        return [{"gap": "evidence for " + goal}]

    def propose_claim(self, proposal: dict[str, object]) -> str:
        return "claim-proposal-1"

    def retrieve(self, query: str) -> list[dict[str, str]]:
        return []

    def query_graph(self, query: dict[str, object]) -> dict[str, object]:
        return {}

    def get_subgraph(self, seed_ids: list[str]) -> dict[str, object]:
        return {}

    def score_investigation(self, investigation: dict[str, object]) -> float:
        return 0.5

    def verify_claim(self, claim_id: str) -> dict[str, object]:
        return {"claim_id": claim_id}

    def commit_knowledge_update(self, verified_proposal: dict[str, object]) -> str:
        return "commit-1"


class MockWorkflow:
    def __init__(self) -> None:
        self.statuses: dict[str, str] = {}

    def submit_workflow(self, specification: dict[str, object]) -> str:
        workflow_id = "workflow-1"
        self.statuses[workflow_id] = "COMPLETED"
        return workflow_id

    def get_status(self, workflow_id: str) -> str:
        return self.statuses[workflow_id]

    def cancel_workflow(self, workflow_id: str) -> None:
        self.statuses[workflow_id] = "CANCELLED"

    def get_outputs(self, workflow_id: str) -> dict[str, object]:
        return {"metric": 1.0, "artifact": "artifact://experiment/1"}


class FinishModel:
    def complete(self, prompt: str, response_schema: dict[str, object]) -> dict[str, object]:
        return {"action": "finish", "output": {"critique": "structured"}, "token_usage": 4}


class EmptyMemory:
    def recall(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        return []

    def remember(self, record: dict[str, object]) -> str:
        return "memory-1"


class EndToEndTests(unittest.TestCase):
    def test_goal_to_verified_update_proposal_with_parallel_tasks_and_experiment(self) -> None:
        research = ResearchCoordinator(MockKnowledge())
        investigation = research.create_investigation(
            "Does method X improve metric Y?", {"tokens": 100}
        )
        gaps = research.discover_gaps(investigation)
        hypothesis = research.propose_hypothesis(
            HypothesisProposal(
                "X improves Y",
                "gap identified",
                ("benchmark",),
                0.6,
                investigation.investigation_id,
            )
        )

        scheduler = Scheduler()
        worker_a = WorkerProcess(
            scheduler,
            "worker-a",
            frozenset({"retrieve"}),
            {"retrieve": lambda task: {"source": task.description}},
        )
        worker_b = WorkerProcess(
            scheduler,
            "worker-b",
            frozenset({"retrieve", "synthesize"}),
            {
                "retrieve": lambda task: {"source": task.description},
                "synthesize": lambda task: {"summary": "combined"},
            },
        )
        evidence_a = scheduler.enqueue(Task("retrieve benchmark evidence", "retrieve"))
        evidence_b = scheduler.enqueue(Task("retrieve counter-evidence", "retrieve"))
        synthesis = scheduler.enqueue(
            Task(
                "synthesize evidence",
                "synthesize",
                dependencies={evidence_a.task_id, evidence_b.task_id},
            )
        )
        worker_a.run_once()
        worker_b.run_once()
        worker_b.run_once()

        lifecycle = HypothesisLifecycle()
        for state in (
            HypothesisState.PLANNED,
            HypothesisState.INVESTIGATING,
            HypothesisState.EVIDENCE_COLLECTED,
            HypothesisState.CRITIQUE,
            HypothesisState.SYNTHESIS,
            HypothesisState.SUPPORTED,
        ):
            lifecycle.transition(hypothesis, state)
        experiments = ExperimentExecutor(MockWorkflow())
        experiment_id = experiments.submit(
            Experiment(
                hypothesis.hypothesis_id,
                {"seed": 7},
                {"evidence": synthesis.outputs},
                {"cpu": 1},
                ("metric",),
                {"seed": "7"},
            )
        )
        self.assertEqual(experiments.status(experiment_id).value, "COMPLETED")

        agent = Agent("Critic", "Critic", frozenset())
        executor = AgentExecutor(
            FinishModel(), EmptyMemory(), ToolRegistry(PolicyEngine({agent.agent_id: frozenset()}))
        )
        run = executor.create_run(
            agent, investigation.investigation_id, Budget(10, timedelta(minutes=1), 0, 1, 1)
        )
        executor.run_step(run.run_id, "critique results", {"title": "Critique"})
        claim_id = research.propose_knowledge_update(
            KnowledgeUpdateProposal(
                ("X improves Y in this test",),
                ("artifact://experiment/1",),
                0.6,
                (),
                "mock verification required",
            )
        )

        self.assertEqual(gaps[0]["gap"], "evidence for Does method X improve metric Y?")
        self.assertEqual(synthesis.outputs, {"summary": "combined"})
        self.assertEqual(hypothesis.state, HypothesisState.SUPPORTED)
        self.assertEqual(run.outputs["critique"], "structured")
        self.assertEqual(claim_id, "claim-proposal-1")

    def test_workflow_failure_is_reflected_without_hiding_it(self) -> None:
        workflow = MockWorkflow()
        experiments = ExperimentExecutor(workflow)
        experiment_id = experiments.submit(
            Experiment("hypothesis-1", {}, {}, {"cpu": 1}, (), {"seed": "1"})
        )
        workflow.statuses["workflow-1"] = "FAILED"

        self.assertEqual(experiments.status(experiment_id).value, "FAILED")
