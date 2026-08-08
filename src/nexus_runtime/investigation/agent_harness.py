"""Production Agent Harness bridging AgentExecutor runs into the worker port.

The harness owns the durable loop between the distributed worker and the
explicit agent executor: it creates or resumes an ``AgentRun`` under the
coordinator-assigned ``run_id``, drives the reason/choose/execute phases while
honouring distributed cancellation and step limits, and persists the terminal
outcome as a lineage-complete :class:`InvestigationResult`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from nexus_runtime.agent import AgentExecutor
from nexus_runtime.distributed.model import FailureClass
from nexus_runtime.distributed.worker import (
    HarnessExecutionContext,
    HarnessOutcome,
    HarnessStatus,
)
from nexus_runtime.models import (
    Agent,
    AgentRun,
    AgentRunState,
    Budget,
    DomainError,
    ToolCall,
)

from .evidence import (
    ClaimStatement,
    Evidence,
    EvidenceRole,
    EvidenceSet,
    InvestigationResult,
    InvestigationResultState,
)
from .provenance import EvidenceProvenance
from .results import InvestigationResultRepository

DEFAULT_ACTION_SCHEMA: dict[str, Any] = {
    "title": "investigation_action",
    "type": "object",
    "required": ["action"],
    "properties": {
        "action": {
            "type": "string",
            "enum": ["tool", "finish", "delegate", "wait"],
        },
        "tool": {"type": "string"},
        "input": {"type": "object"},
        "output": {"type": "object"},
    },
}

_TERMINAL_RUN_STATES = frozenset(
    {
        AgentRunState.COMPLETED,
        AgentRunState.FAILED,
        AgentRunState.CANCELLED,
    }
)


class AgentHarness:
    """Execute or resume one AgentExecutor run for a distributed task.

    The ``Agent`` used for fresh runs is fixed at construction time; the
    executor's state store (when configured) is the source of truth for
    resuming a checkpointed run after a worker crash or retry.
    """

    def __init__(
        self,
        executor: AgentExecutor,
        results: InvestigationResultRepository,
        agent: Agent,
        *,
        budget: Budget,
        max_steps: int = 8,
    ) -> None:
        self._executor = executor
        self._results = results
        self._agent = agent
        self._budget = budget
        self._max_steps = max_steps

    def execute_or_resume(
        self,
        context: HarnessExecutionContext,
        cancellation_requested: Callable[[], bool],
    ) -> HarnessOutcome:
        run = self._ensure_run(context)
        try:
            if run.state == AgentRunState.CREATED:
                run = self._executor.transition(
                    run.run_id, AgentRunState.RUNNING, "begin harness execution"
                )
            steps = 0
            while run.state == AgentRunState.RUNNING:
                if cancellation_requested():
                    run = self._executor.transition(
                        run.run_id, AgentRunState.CANCELLED, "task cancellation requested"
                    )
                    break
                if steps >= self._max_steps:
                    run = self._executor.transition(
                        run.run_id, AgentRunState.PAUSED, "harness step limit reached"
                    )
                    break
                steps += 1
                response = self._executor.reason(
                    run.run_id, self._build_prompt(context), DEFAULT_ACTION_SCHEMA
                )
                action = self._executor.choose_action(run.run_id, response)
                self._executor.execute_action(run.run_id, action)
                self._executor.update_state(run.run_id)
                run = self._executor.get_run(run.run_id)
        except DomainError as exc:
            run = self._executor.get_run(run.run_id)
            return self._failure_outcome(context, run, str(exc))
        return self._terminal_outcome(context, run)

    def cancel_run(self, run_id: str) -> None:
        """Best-effort cancellation of an in-memory run owned by this harness."""
        try:
            run = self._executor.get_run(run_id)
        except DomainError:
            return
        if run.state not in _TERMINAL_RUN_STATES:
            self._executor.transition(run_id, AgentRunState.CANCELLED, "harness cancel_run")

    def _ensure_run(self, context: HarnessExecutionContext) -> AgentRun:
        try:
            existing = self._executor.get_run(context.run_id)
        except DomainError:
            existing = None
        if existing is not None and existing.run_id == context.run_id:
            return existing
        if self._executor.restore_checkpoint(context.run_id) is not None:
            return self._executor.resume(context.run_id)
        investigation_id = str(context.metadata.get("investigation_id") or "")
        if not investigation_id.strip():
            raise DomainError("harness task metadata lacks investigation_id")
        return self._executor.create_run(
            self._agent,
            investigation_id,
            self._budget,
            task_id=context.task_id,
            run_id=context.run_id,
        )

    def _failure_outcome(
        self,
        context: HarnessExecutionContext,
        run: AgentRun,
        error: str,
    ) -> HarnessOutcome:
        checkpoint_ref = self._checkpoint_ref(context.run_id)
        if run.state == AgentRunState.CANCELLED:
            return HarnessOutcome(
                HarnessStatus.CANCELLED,
                checkpoint_ref=checkpoint_ref,
                error=error,
            )
        return HarnessOutcome(
            HarnessStatus.FAILED,
            checkpoint_ref=checkpoint_ref,
            failure_class=self._classify(error),
            error=error,
        )

    def _terminal_outcome(
        self,
        context: HarnessExecutionContext,
        run: AgentRun,
    ) -> HarnessOutcome:
        checkpoint_ref = self._checkpoint_ref(context.run_id)
        if run.state == AgentRunState.COMPLETED:
            return HarnessOutcome(
                HarnessStatus.SUCCEEDED,
                self._results.save(self._build_result(context, run)),
            )
        if run.state == AgentRunState.CANCELLED:
            return HarnessOutcome(
                HarnessStatus.CANCELLED,
                checkpoint_ref=checkpoint_ref,
                error="agent run cancelled",
            )
        if run.state == AgentRunState.PAUSED:
            return HarnessOutcome(
                HarnessStatus.FAILED,
                checkpoint_ref=checkpoint_ref,
                failure_class=FailureClass.PERMANENT,
                error="agent run paused",
            )
        if run.state == AgentRunState.WAITING:
            return HarnessOutcome(
                HarnessStatus.FAILED,
                checkpoint_ref=checkpoint_ref,
                failure_class=FailureClass.TRANSIENT,
                error="agent run waiting for external input",
            )
        return HarnessOutcome(
            HarnessStatus.FAILED,
            checkpoint_ref=checkpoint_ref,
            failure_class=FailureClass.TRANSIENT,
            error=f"agent run ended in {run.state.value}",
        )

    def _build_result(
        self,
        context: HarnessExecutionContext,
        run: AgentRun,
    ) -> InvestigationResult:
        investigation_id = str(context.metadata.get("investigation_id") or "")
        evidence = tuple(
            self._build_evidence(context, call)
            for call in run.tool_calls
            if call.status != "FAILED"
        )
        evidence_set = EvidenceSet(session_id=context.correlation_id, evidence=evidence)
        return InvestigationResult(
            session_id=context.correlation_id,
            investigation_id=investigation_id,
            task_id=context.task_id,
            attempt_id=context.attempt_id,
            run_id=context.run_id,
            state=InvestigationResultState.COMPLETED,
            evidence_set=evidence_set,
            metadata={
                "agent_id": run.agent_id,
                "outputs": run.outputs,
                "tool_call_count": len(run.tool_calls),
            },
        )

    @staticmethod
    def _build_evidence(
        context: HarnessExecutionContext,
        call: ToolCall,
    ) -> Evidence:
        tool_name = call.tool_name
        status = call.status
        source = f"tool://{tool_name}"
        claim = ClaimStatement(
            text=f"{tool_name} executed with status {status}",
            subject=tool_name,
            predicate="executed",
            object=status,
        )
        provenance = EvidenceProvenance(
            session_id=context.correlation_id,
            investigation_id=str(context.metadata.get("investigation_id") or ""),
            task_id=context.task_id,
            attempt_id=context.attempt_id,
            run_id=context.run_id,
            tool_call_id=call.tool_call_id,
            source_id=f"tool:{tool_name}",
            document_id=_provenance_value(call.input, "document_id", f"document:{tool_name}"),
            chunk_id=_provenance_value(call.input, "chunk_id", f"chunk:{tool_name}"),
            source_reference=source,
        )
        return Evidence(
            investigation_id=provenance.investigation_id,
            source=source,
            claim=claim,
            provenance=provenance,
            confidence=0.5,
            source_quality=0.5,
            excerpt=f"{tool_name} executed",
            payload=dict(call.input),
            role=EvidenceRole.SUPPORTING,
            metadata={"status": status, "task_id": context.task_id},
        )

    @staticmethod
    def _classify(error: str) -> FailureClass:
        lowered = error.lower()
        if "denied by policy" in lowered:
            return FailureClass.POLICY_VIOLATION
        if "budget exhausted" in lowered:
            return FailureClass.BUDGET_EXHAUSTED
        return FailureClass.TRANSIENT

    @staticmethod
    def _build_prompt(context: HarnessExecutionContext) -> str:
        metadata = context.metadata
        parts = ["Investigate the objective and record structured evidence."]
        question = str(metadata.get("question") or "")
        hypothesis = str(metadata.get("hypothesis") or "")
        if question:
            parts.append(f"Question: {question}")
        if hypothesis:
            parts.append(f"Hypothesis: {hypothesis}")
        return "\n".join(parts)

    @staticmethod
    def _checkpoint_ref(run_id: str) -> str:
        return f"checkpoint://run/{run_id}"


def _provenance_value(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        return default
    return value


__all__ = ["AgentHarness", "DEFAULT_ACTION_SCHEMA"]
