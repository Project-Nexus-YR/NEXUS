"""Investigation, hypothesis, and experiment lifecycle services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import KnowledgeService, WorkflowExecutor
from .events import Event, EventBus, InMemoryEventBus
from .models import (
    DomainError,
    Experiment,
    ExperimentState,
    Hypothesis,
    HypothesisProposal,
    HypothesisState,
    Investigation,
    KnowledgeUpdateProposal,
)


class HypothesisLifecycle:
    _TRANSITIONS: dict[HypothesisState, frozenset[HypothesisState]] = {
        HypothesisState.PROPOSED: frozenset({HypothesisState.PLANNED}),
        HypothesisState.PLANNED: frozenset({HypothesisState.INVESTIGATING}),
        HypothesisState.INVESTIGATING: frozenset({HypothesisState.EVIDENCE_COLLECTED}),
        HypothesisState.EVIDENCE_COLLECTED: frozenset({HypothesisState.CRITIQUE}),
        HypothesisState.CRITIQUE: frozenset({HypothesisState.SYNTHESIS}),
        HypothesisState.SYNTHESIS: frozenset(
            {HypothesisState.SUPPORTED, HypothesisState.REFUTED, HypothesisState.INCONCLUSIVE}
        ),
        HypothesisState.SUPPORTED: frozenset(),
        HypothesisState.REFUTED: frozenset(),
        HypothesisState.INCONCLUSIVE: frozenset(),
    }

    def transition(self, hypothesis: Hypothesis, target: HypothesisState) -> Hypothesis:
        if target not in self._TRANSITIONS[hypothesis.state]:
            raise DomainError(f"invalid hypothesis transition: {hypothesis.state} -> {target}")
        hypothesis.state = target
        return hypothesis


class ExperimentExecutor:
    """Adapter-backed experiment engine. Workflow semantics stay behind this contract."""

    def __init__(self, workflow: WorkflowExecutor) -> None:
        self._workflow = workflow
        self._experiments: dict[str, Experiment] = {}
        self._workflow_ids: dict[str, str] = {}

    def submit(self, experiment: Experiment) -> str:
        if experiment.state != ExperimentState.CREATED:
            raise DomainError("only created experiments can be submitted")
        workflow_id = self._workflow.submit_workflow(
            {
                "hypothesis_id": experiment.hypothesis_id,
                "parameters": experiment.parameters,
                "inputs": experiment.inputs,
                "resource_budget": experiment.resource_budget,
                "metrics": experiment.metrics,
                "reproducibility": experiment.reproducibility,
            }
        )
        experiment.state = ExperimentState.SUBMITTED
        self._experiments[experiment.experiment_id] = experiment
        self._workflow_ids[experiment.experiment_id] = workflow_id
        return experiment.experiment_id

    def status(self, experiment_id: str) -> ExperimentState:
        experiment = self._get(experiment_id)
        status = self._workflow.get_status(self._workflow_ids[experiment_id])
        mapping = {
            "RUNNING": ExperimentState.RUNNING,
            "COMPLETED": ExperimentState.COMPLETED,
            "FAILED": ExperimentState.FAILED,
            "CANCELLED": ExperimentState.CANCELLED,
        }
        if status not in mapping:
            raise DomainError(f"unrecognized workflow status: {status}")
        experiment.state = mapping[status]
        if experiment.state == ExperimentState.COMPLETED:
            experiment.outputs = self._workflow.get_outputs(self._workflow_ids[experiment_id])
        return experiment.state

    def cancel(self, experiment_id: str) -> None:
        experiment = self._get(experiment_id)
        self._workflow.cancel_workflow(self._workflow_ids[experiment_id])
        experiment.state = ExperimentState.CANCELLED

    def artifacts(self, experiment_id: str) -> list[str]:
        return list(self._get(experiment_id).artifact_refs)

    def _get(self, experiment_id: str) -> Experiment:
        try:
            return self._experiments[experiment_id]
        except KeyError as exc:
            raise DomainError(f"unknown experiment: {experiment_id}") from exc


@dataclass(slots=True)
class ResearchCoordinator:
    """Coordinates research records without owning the knowledge subsystem."""

    knowledge: KnowledgeService
    event_bus: EventBus = field(default_factory=InMemoryEventBus)

    def create_investigation(self, goal: str, budget: dict[str, int]) -> Investigation:
        if not goal.strip():
            raise DomainError("investigation goal cannot be empty")
        investigation = Investigation(goal=goal, budget=budget, state="KNOWLEDGE_GAP_DISCOVERY")
        self._emit("investigation.created", investigation.investigation_id, {"goal": goal})
        return investigation

    def discover_gaps(self, investigation: Investigation) -> list[dict[str, Any]]:
        gaps = self.knowledge.find_knowledge_gaps(investigation.goal)
        investigation.state = "HYPOTHESIS_GENERATION"
        self._emit(
            "investigation.gaps.discovered", investigation.investigation_id, {"count": len(gaps)}
        )
        return gaps

    def propose_hypothesis(self, proposal: HypothesisProposal) -> Hypothesis:
        hypothesis = Hypothesis(
            proposal.statement,
            proposal.rationale,
            proposal.expected_evidence,
            proposal.confidence,
            proposal.required_investigation,
        )
        self._emit(
            "hypothesis.proposed", hypothesis.hypothesis_id, {"confidence": hypothesis.confidence}
        )
        return hypothesis

    def propose_knowledge_update(self, proposal: KnowledgeUpdateProposal) -> str:
        """Propose only; explicit verification and commit remain knowledge-service operations."""
        claim_id = self.knowledge.propose_claim(
            {
                "claims": proposal.claims,
                "evidence_refs": proposal.evidence_refs,
                "confidence": proposal.confidence,
                "contradictions": proposal.contradictions,
                "justification": proposal.justification,
            }
        )
        self._emit("knowledge_update.proposed", claim_id, {"claims": len(proposal.claims)})
        return claim_id

    def _emit(self, event_type: str, correlation_id: str, payload: dict[str, Any]) -> None:
        self.event_bus.publish(
            Event(event_type, payload, "research-coordinator", correlation_id, correlation_id)
        )
