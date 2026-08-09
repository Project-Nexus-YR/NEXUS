"""Section 14: no synthetic claims may originate from tool status or telemetry.

The only production paths that construct ``ClaimStatement`` are the agent
harness's structured-conclusion parser and the benchmark fixtures; the only
path that constructs ``CandidateClaim`` is the extractor.  A tool result's
``status`` field can never become a claim.
"""

from __future__ import annotations

import ast
from pathlib import Path

from nexus_runtime.investigation.candidate_claims import CandidateClaimExtractor
from nexus_runtime.investigation.evidence import (
    EvidenceSet,
    InvestigationResult,
    InvestigationResultState,
    ToolObservation,
)

_RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "src" / "nexus_runtime"

_CLAIM_STATEMENT_ALLOWED = {
    "agent_harness.py",
    "benchmark.py",
}

_CANDIDATE_CLAIM_ALLOWED = {
    "candidate_claims.py",
}


def _direct_calls(root: Path, name: str) -> list[tuple[str, int]]:
    offenders: list[tuple[str, int]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Name) and function.id == name:
                    offenders.append((path.name, node.lineno))
    return offenders


def test_claim_statement_is_only_constructed_by_the_structured_parser() -> None:
    offenders = _direct_calls(_RUNTIME_ROOT, "ClaimStatement")

    assert {filename for filename, _ in offenders} <= _CLAIM_STATEMENT_ALLOWED


def test_candidate_claim_is_only_constructed_by_the_extractor() -> None:
    offenders = _direct_calls(_RUNTIME_ROOT, "CandidateClaim")

    assert {filename for filename, _ in offenders} <= _CANDIDATE_CLAIM_ALLOWED
    assert {filename for filename, _ in offenders} == {"candidate_claims.py"}


def test_tool_status_alone_can_never_produce_a_claim() -> None:
    observations = (
        ToolObservation(
            observation_id="observation-succeeded",
            tool_name="search",
            status="SUCCEEDED",
            input={},
            output={"text": "the company is active"},
            source_reference="source://succeeded",
        ),
        ToolObservation(
            observation_id="observation-failed",
            tool_name="search",
            status="FAILED",
            input={},
            output=None,
            source_reference="source://failed",
        ),
    )
    investigation_result = InvestigationResult(
        session_id="session-s14",
        investigation_id="investigation-s14",
        task_id="task-s14",
        attempt_id="attempt-s14",
        run_id="run-s14",
        state=InvestigationResultState.COMPLETED,
        evidence_set=EvidenceSet(session_id="session-s14", evidence=()),
        observations=observations,
        conclusions=(),
    )

    extraction = CandidateClaimExtractor().extract(investigation_result)

    assert not extraction.candidates
    assert not extraction.evidence_set.evidence
