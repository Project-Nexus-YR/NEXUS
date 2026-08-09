"""Production Agent Harness bridging AgentExecutor runs into the worker port.

The harness owns the durable loop between the distributed worker and the
explicit agent executor: it creates or resumes an ``AgentRun`` under the
coordinator-assigned ``run_id``, drives the reason/choose/execute phases while
honouring distributed cancellation and step limits, and persists the terminal
outcome as a lineage-complete :class:`InvestigationResult`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
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
    utcnow,
)

from .candidate_claims import CandidateClaimExtractor
from .evidence import (
    AgentConclusion,
    ClaimStatement,
    EvidenceSet,
    InvestigationResult,
    InvestigationResultState,
    ToolObservation,
)
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
        "output": {
            "type": "object",
            "properties": {
                "final_answer": {"type": "string"},
                "conclusions": {
                    "type": "array",
                    "description": "structured assertions derived from the investigation",
                    "items": {
                        "type": "object",
                        "required": ["claim", "supporting_observation_ids"],
                        "properties": {
                            "conclusion_id": {"type": "string"},
                            "claim": {
                                "type": "object",
                                "required": ["text", "subject", "predicate", "object"],
                                "properties": {
                                    "text": {"type": "string"},
                                    "subject": {"type": "string"},
                                    "predicate": {"type": "string"},
                                    "object": {"type": "string"},
                                },
                            },
                            "supporting_observation_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "confidence": {"type": "number"},
                            "metadata": {"type": "object"},
                        },
                    },
                },
            },
        },
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
                    run.run_id, self._build_prompt(context, run), DEFAULT_ACTION_SCHEMA
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
        observations = tuple(
            self._build_observation(context, call)
            for call in run.tool_calls
            if call.status != "FAILED"
        )
        conclusions, malformed = self._parse_conclusions(run.outputs)
        metadata = {
            "agent_id": run.agent_id,
            "outputs": run.outputs,
            "tool_call_count": len(run.tool_calls),
            "malformed_conclusions": malformed,
        }
        result = InvestigationResult(
            session_id=context.correlation_id,
            investigation_id=investigation_id,
            task_id=context.task_id,
            attempt_id=context.attempt_id,
            run_id=context.run_id,
            state=InvestigationResultState.COMPLETED,
            evidence_set=EvidenceSet(session_id=context.correlation_id, evidence=()),
            final_answer=_final_answer(run.outputs),
            conclusions=conclusions,
            observations=observations,
            metadata=metadata,
        )
        extraction = CandidateClaimExtractor().extract(result)
        result_metadata = dict(metadata)
        result_metadata["claim_extraction"] = {
            "extractor": "candidate_claim_extractor",
            "candidate_count": len(extraction.candidates),
            "conclusion_count": len(conclusions),
            "candidate_ids": [item.candidate_id for item in extraction.candidates],
            "diagnostics": [item.to_dict() for item in extraction.diagnostics],
        }
        return replace(
            result,
            evidence_set=extraction.evidence_set,
            metadata=result_metadata,
        )

    @staticmethod
    def _build_observation(
        context: HarnessExecutionContext,
        call: ToolCall,
    ) -> ToolObservation:
        tool_name = call.tool_name
        output = call.output or {}
        source_id = _observation_value(output, call.input, "source_id", f"tool:{tool_name}")
        document_id = _observation_value(output, call.input, "document_id", f"document:{tool_name}")
        chunk_id = _observation_value(output, call.input, "chunk_id", f"chunk:{tool_name}")
        source_reference = _observation_value(
            output, call.input, "source_reference", f"tool://{tool_name}"
        )
        return ToolObservation(
            observation_id=call.tool_call_id,
            tool_name=tool_name,
            status=call.status,
            input=dict(call.input),
            output=None if call.output is None else dict(call.output),
            source_reference=source_reference,
            timestamp=call.completed_at or utcnow(),
            metadata={
                "status": call.status,
                "task_id": context.task_id,
                "source_id": source_id,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "source_reference": source_reference,
                "source_quality": _observation_quality(output, call.input),
            },
        )

    @staticmethod
    def _parse_conclusions(
        outputs: Mapping[str, Any],
    ) -> tuple[tuple[AgentConclusion, ...], list[dict[str, Any]]]:
        raw = outputs.get("conclusions")
        if not isinstance(raw, list):
            return (), []
        conclusions: list[AgentConclusion] = []
        malformed: list[dict[str, Any]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                malformed.append({"index": index, "reason": "conclusion must be an object"})
                continue
            try:
                claim_payload = item.get("claim")
                if not isinstance(claim_payload, dict):
                    raise ValueError("claim must be an object")
                claim = ClaimStatement(
                    text=_payload_string(claim_payload, "text"),
                    subject=_payload_string(claim_payload, "subject"),
                    predicate=_payload_string(claim_payload, "predicate"),
                    object=_payload_string(claim_payload, "object"),
                    claim_id=_optional_string(claim_payload, "claim_id"),
                )
                observation_ids = item.get("supporting_observation_ids", [])
                if not isinstance(observation_ids, list) or any(
                    not isinstance(observation_id, str) for observation_id in observation_ids
                ):
                    raise ValueError("supporting_observation_ids must be a list of strings")
                raw_metadata = item.get("metadata", {})
                if not isinstance(raw_metadata, dict):
                    raise ValueError("conclusion metadata must be an object")
                confidence = item.get("confidence", 0.5)
                if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                    raise ValueError("conclusion confidence must be numeric")
                conclusions.append(
                    AgentConclusion(
                        claim=claim,
                        supporting_observation_ids=tuple(observation_ids),
                        confidence=float(confidence),
                        conclusion_id=_optional_string(item, "conclusion_id"),
                        metadata=dict(raw_metadata),
                    )
                )
            except ValueError as exc:
                malformed.append({"index": index, "reason": str(exc)})
        return tuple(conclusions), malformed

    @staticmethod
    def _classify(error: str) -> FailureClass:
        lowered = error.lower()
        if "denied by policy" in lowered:
            return FailureClass.POLICY_VIOLATION
        if "budget exhausted" in lowered:
            return FailureClass.BUDGET_EXHAUSTED
        return FailureClass.TRANSIENT

    @staticmethod
    def _build_prompt(context: HarnessExecutionContext, run: AgentRun) -> str:
        metadata = context.metadata
        parts = ["Investigate the objective and record structured evidence."]
        question = str(metadata.get("question") or "")
        hypothesis = str(metadata.get("hypothesis") or "")
        if question:
            parts.append(f"Question: {question}")
        if hypothesis:
            parts.append(f"Hypothesis: {hypothesis}")
        for call in run.tool_calls:
            parts.append(
                f"[observation:{call.tool_call_id}] tool {call.tool_name} "
                f"input={json.dumps(call.input, sort_keys=True, default=str)}"
            )
        return "\n".join(parts)

    @staticmethod
    def _checkpoint_ref(run_id: str) -> str:
        return f"checkpoint://run/{run_id}"


def _provenance_value(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        return default
    return value


def _payload_string(payload: Mapping[str, Any], key: str, *, default: str = "") -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value or default


def _optional_string(payload: Mapping[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value or default


def _observation_value(
    output: Mapping[str, Any],
    tool_input: Mapping[str, Any],
    key: str,
    default: str,
) -> str:
    return _provenance_value(output, key, _provenance_value(tool_input, key, default))


def _observation_quality(output: Mapping[str, Any], tool_input: Mapping[str, Any]) -> float:
    for source in (output, tool_input):
        value = source.get("source_quality")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return min(1.0, max(0.0, float(value)))
    return 0.5


def _final_answer(outputs: Mapping[str, Any]) -> str:
    value = outputs.get("final_answer")
    if not isinstance(value, str):
        return ""
    return value


__all__ = ["AgentHarness", "DEFAULT_ACTION_SCHEMA"]
