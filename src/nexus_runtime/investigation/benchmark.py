"""Deterministic Phase 4 baseline: 10 gaps, 50 candidates, 10 tasks."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from time import perf_counter

from nexus_knowledge.domain.knowledge_gap import Investigation, KnowledgeGap
from nexus_runtime.distributed.service import RuntimeApplication
from nexus_runtime.distributed.simulator import DeterministicHarness, LocalDistributedSimulator

from .evaluation import EvidenceEvaluator
from .evidence import ClaimStatement, Evidence, EvidenceSet
from .execution import PlanExecutionController
from .generator import InvestigationGenerator
from .objective import ResearchObjective
from .planner import InvestigationPlanner
from .provenance import EvidenceProvenance
from .scoring import InvestigationScoringModel
from .selector import InvestigationSelector
from .session import InvestigationBudget, InvestigationSession


def run_benchmark() -> dict[str, object]:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    objective = ResearchObjective(
        question="Which ten missing facts should NEXUS resolve?",
        success_criteria=("all selected gaps have evidence",),
        objective_id="objective-benchmark",
        created_at=timestamp,
    )
    gaps = [_gap(index) for index in range(10)]
    session = InvestigationSession(
        objective_id=objective.objective_id,
        budget=InvestigationBudget(2, 10, 10, 100.0, timedelta(minutes=10)),
        session_id="session-benchmark",
        created_at=timestamp,
        updated_at=timestamp,
    )

    planning_started = perf_counter()
    candidates = InvestigationGenerator().generate(objective, gaps)
    scores = InvestigationScoringModel().score_all(candidates, gaps)
    selection = InvestigationSelector().select(
        scores,
        budget=session.budget,
        usage=session.usage,
        worker_capacity=10,
        top_k=10,
    )
    plan = InvestigationPlanner().build(session, selection, created_at=timestamp)
    planning_seconds = perf_counter() - planning_started

    simulator = LocalDistributedSimulator(start=timestamp)
    harness = DeterministicHarness()
    capabilities = frozenset({"document_analysis", "search"})
    for index in range(10):
        simulator.add_worker(f"investigation-worker-{index}", capabilities, harness)
    runtime = RuntimeApplication(simulator.coordinator)
    run_ids = {
        investigation.investigation_id: f"benchmark-run-{index}"
        for index, investigation in enumerate(plan.investigations)
    }
    scheduling_started = perf_counter()
    execution = PlanExecutionController(runtime).start(plan, run_ids)
    scheduling_seconds = perf_counter() - scheduling_started
    execution_started = perf_counter()
    runtime_report = simulator.run_until_terminal()
    execution_seconds = perf_counter() - execution_started

    evidence_set = EvidenceSet(
        session_id=session.session_id,
        evidence=tuple(
            _evidence(
                session.session_id,
                investigation.investigation_id,
                task_id,
                run_ids[investigation.investigation_id],
                index,
            )
            for index, investigation in enumerate(plan.investigations)
            for task_id in (execution.execution.task_ids[investigation.investigation_id],)
        ),
    )
    evaluation_started = perf_counter()
    evaluation = EvidenceEvaluator().evaluate(evidence_set)
    evaluation_seconds = perf_counter() - evaluation_started
    total_seconds = planning_seconds + scheduling_seconds + execution_seconds + evaluation_seconds
    return {
        "methodology": {
            "gaps": len(gaps),
            "candidates": len(candidates),
            "selected": len(selection.selected),
            "distributed_tasks": runtime_report.tasks,
            "clock": "fixed UTC inputs with local deterministic runtime",
        },
        "results": {
            "planning_seconds": planning_seconds,
            "scheduling_seconds": scheduling_seconds,
            "execution_seconds": execution_seconds,
            "evidence_evaluation_seconds": evaluation_seconds,
            "total_iteration_seconds": total_seconds,
            "accepted_evidence": evaluation.accepted_evidence_count,
            "succeeded_tasks": runtime_report.succeeded,
        },
    }


def _gap(index: int) -> KnowledgeGap:
    gap = KnowledgeGap(
        id=f"gap-{index}",
        kind="missing_evidence",
        description=f"resolve missing fact {index}",
        reason="benchmark fixture",
        affected_entities=[f"entity-{index}"],
        uncertainty=0.8,
        importance=0.8,
        estimated_cost=1.0,
        created_at="2026-01-01T00:00:00Z",
    )
    gap.candidate_investigations = [
        Investigation(
            id=f"legacy-{index}-{candidate}",
            gap_id=gap.id,
            description=f"investigate fact {index} through channel {candidate}",
            target_entities=[f"entity-{index}"],
            estimated_cost=1.0,
            metadata={"evidence_channel": f"channel-{candidate}"},
            created_at="2026-01-01T00:00:00Z",
        )
        for candidate in range(5)
    ]
    return gap


def _evidence(
    session_id: str,
    investigation_id: str,
    task_id: str,
    run_id: str,
    index: int,
) -> Evidence:
    source_id = f"source-{index}"
    return Evidence(
        investigation_id=investigation_id,
        source=f"benchmark://{source_id}",
        claim=ClaimStatement(
            text=f"fact {index} is supported",
            subject=f"fact-{index}",
            predicate="status",
            object="supported",
            claim_id=f"claim-benchmark-{index}",
        ),
        provenance=EvidenceProvenance(
            session_id=session_id,
            investigation_id=investigation_id,
            task_id=task_id,
            attempt_id=f"attempt-{index}",
            run_id=run_id,
            tool_call_id=f"tool-{index}",
            source_id=source_id,
            document_id=f"document-{index}",
            chunk_id=f"chunk-{index}",
            source_reference=f"benchmark://{source_id}",
        ),
        confidence=0.8,
        source_quality=0.8,
        excerpt=f"supporting excerpt {index}",
        evidence_id=f"evidence-benchmark-{index}",
    )


def main() -> int:
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
