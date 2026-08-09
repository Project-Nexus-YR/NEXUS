"""Deterministic extraction of candidate claims from structured agent conclusions.

The extractor is the clean application boundary between agent output and the
existing evidence pipeline:

    structured conclusions
        -> CandidateClaimExtractor
        -> CandidateClaim[] -> EvidenceSet -> EvidenceEvaluator

It never verifies, persists knowledge, mutates a graph, or calls an LLM.
Malformed conclusions and dangling observation references are rejected with
explicit diagnostics instead of becoming knowledge.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from nexus_runtime.models import utcnow

from .evidence import (
    AgentConclusion,
    ClaimStatement,
    Evidence,
    EvidenceRole,
    EvidenceSet,
    InvestigationResult,
    ToolObservation,
    _persisted_float,
    _persisted_string,
    _stable_id,
)
from .provenance import EvidenceProvenance


class CandidateStatus(StrEnum):
    """Explicit acquisition lifecycle layered over verification states.

    This is an acquisition state, not a rename of the knowledge service's
    ``VerificationState``.
    """

    CANDIDATE = "candidate"
    EVALUATED = "evaluated"
    VERIFIED = "verified"
    REJECTED = "rejected"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class ExtractionDiagnostic:
    """Why a conclusion was not acquired as a candidate claim."""

    conclusion_id: str
    code: str
    message: str
    recovered: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.conclusion_id, "conclusion_id"),
            (self.code, "code"),
            (self.message, "message"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "conclusion_id": self.conclusion_id,
            "code": self.code,
            "message": self.message,
            "recovered": self.recovered,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExtractionDiagnostic:
        recovered = payload.get("recovered", False)
        if not isinstance(recovered, bool):
            raise ValueError("malformed extraction diagnostic")
        try:
            return cls(
                conclusion_id=_persisted_string(payload, "conclusion_id"),
                code=_persisted_string(payload, "code"),
                message=_persisted_string(payload, "message"),
                recovered=recovered,
            )
        except KeyError as exc:
            raise ValueError("malformed extraction diagnostic") from exc


@dataclass(frozen=True, slots=True)
class CandidateClaim:
    """An agent-proposed assertion with its supporting evidence attached."""

    claim: ClaimStatement
    evidence: tuple[Evidence, ...]
    conclusion_id: str
    candidate_id: str
    confidence: float
    status: CandidateStatus = CandidateStatus.CANDIDATE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in (
            (self.conclusion_id, "conclusion_id"),
            (self.candidate_id, "candidate_id"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError("candidate confidence must be numeric")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("candidate confidence must be between zero and one")
        if not self.evidence:
            raise ValueError("a candidate claim requires supporting evidence")
        if not isinstance(self.metadata, dict):
            raise ValueError("candidate metadata must be an object")

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "conclusion_id": self.conclusion_id,
            "candidate_id": self.candidate_id,
            "confidence": self.confidence,
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CandidateClaim:
        claim = payload.get("claim")
        evidence = payload.get("evidence")
        metadata = payload.get("metadata", {})
        if (
            not isinstance(claim, dict)
            or not isinstance(evidence, list)
            or any(not isinstance(item, dict) for item in evidence)
            or not isinstance(metadata, dict)
        ):
            raise ValueError("malformed candidate claim")
        try:
            return cls(
                claim=ClaimStatement.from_dict(claim),
                evidence=tuple(Evidence.from_dict(item) for item in evidence),
                conclusion_id=_persisted_string(payload, "conclusion_id"),
                candidate_id=_persisted_string(payload, "candidate_id"),
                confidence=_persisted_float(payload, "confidence"),
                status=CandidateStatus(_persisted_string(payload, "status")),
                metadata=dict(metadata),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed candidate claim") from exc


@dataclass(frozen=True, slots=True)
class CandidateExtractionResult:
    """Accepted candidates plus diagnostics and the evidence set fed to evaluation."""

    session_id: str
    candidates: tuple[CandidateClaim, ...]
    diagnostics: tuple[ExtractionDiagnostic, ...]
    evidence_set: EvidenceSet

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "candidates": [item.to_dict() for item in self.candidates],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "evidence_set": self.evidence_set.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CandidateExtractionResult:
        candidates = payload.get("candidates")
        diagnostics = payload.get("diagnostics")
        evidence_set = payload.get("evidence_set")
        if (
            not isinstance(candidates, list)
            or any(not isinstance(item, dict) for item in candidates)
            or not isinstance(diagnostics, list)
            or any(not isinstance(item, dict) for item in diagnostics)
            or not isinstance(evidence_set, dict)
        ):
            raise ValueError("malformed candidate extraction result")
        try:
            return cls(
                session_id=_persisted_string(payload, "session_id"),
                candidates=tuple(CandidateClaim.from_dict(item) for item in candidates),
                diagnostics=tuple(ExtractionDiagnostic.from_dict(item) for item in diagnostics),
                evidence_set=EvidenceSet.from_dict(evidence_set),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed candidate extraction result") from exc


class CandidateClaimExtractor:
    """Deterministic boundary translating structured conclusions into candidates."""

    def extract(self, result: InvestigationResult) -> CandidateExtractionResult:
        observations = {item.observation_id: item for item in result.observations}
        candidates: dict[str, CandidateClaim] = {}
        diagnostics: list[ExtractionDiagnostic] = []
        seen_conclusions: set[str] = set()
        for conclusion in sorted(result.conclusions, key=lambda item: item.conclusion_id):
            if conclusion.conclusion_id in seen_conclusions:
                diagnostics.append(
                    ExtractionDiagnostic(
                        conclusion.conclusion_id,
                        "duplicate_conclusion",
                        "duplicate conclusion discarded",
                        recovered=True,
                    )
                )
                continue
            seen_conclusions.add(conclusion.conclusion_id)
            candidate, conclusion_diagnostics = self._extract_conclusion(
                result, conclusion, observations
            )
            diagnostics.extend(conclusion_diagnostics)
            if candidate is not None:
                if candidate.candidate_id in candidates:
                    diagnostics.append(
                        ExtractionDiagnostic(
                            conclusion.conclusion_id,
                            "duplicate_candidate",
                            "conclusion duplicates an already extracted candidate claim",
                            recovered=True,
                        )
                    )
                    continue
                candidates[candidate.candidate_id] = candidate
        diagnostics.extend(_malformed_conclusion_diagnostics(result))
        evidence = _deduplicated_evidence(tuple(candidates.values()))
        evidence_set = EvidenceSet(session_id=result.session_id, evidence=evidence)
        return CandidateExtractionResult(
            session_id=result.session_id,
            candidates=tuple(candidates[key] for key in sorted(candidates)),
            diagnostics=tuple(
                sorted(diagnostics, key=lambda item: (item.conclusion_id, item.code))
            ),
            evidence_set=evidence_set,
        )

    def _extract_conclusion(
        self,
        result: InvestigationResult,
        conclusion: AgentConclusion,
        observations: Mapping[str, ToolObservation],
    ) -> tuple[CandidateClaim | None, tuple[ExtractionDiagnostic, ...]]:
        if not conclusion.supporting_observation_ids:
            return None, (
                ExtractionDiagnostic(
                    conclusion.conclusion_id,
                    "no_supporting_observations",
                    "conclusion does not reference any observation",
                ),
            )
        unknown = [
            observation_id
            for observation_id in conclusion.supporting_observation_ids
            if observation_id not in observations
        ]
        if unknown:
            return None, (
                ExtractionDiagnostic(
                    conclusion.conclusion_id,
                    "unknown_observation_reference",
                    "unknown observation reference: " + ", ".join(sorted(unknown)),
                ),
            )
        evidence = tuple(
            self._evidence_for(result, conclusion, observations[observation_id])
            for observation_id in conclusion.supporting_observation_ids
        )
        candidate_id = _stable_id(
            "candidate",
            conclusion.claim.claim_id,
            *conclusion.supporting_observation_ids,
        )
        candidate = CandidateClaim(
            claim=conclusion.claim,
            evidence=evidence,
            conclusion_id=conclusion.conclusion_id,
            candidate_id=candidate_id,
            confidence=conclusion.confidence,
            metadata={
                "conclusion_id": conclusion.conclusion_id,
                "extracted_at": utcnow().isoformat(),
            },
        )
        return candidate, ()

    @staticmethod
    def _evidence_for(
        result: InvestigationResult,
        conclusion: AgentConclusion,
        observation: ToolObservation,
    ) -> Evidence:
        hints = observation.metadata
        tool_name = observation.tool_name
        source_id = _hint(hints, "source_id", f"tool:{tool_name}")
        document_id = _hint(hints, "document_id", f"document:{tool_name}")
        chunk_id = _hint(hints, "chunk_id", f"chunk:{tool_name}")
        source_reference = _hint(
            hints,
            "source_reference",
            observation.source_reference or f"tool://{tool_name}",
        )
        provenance = EvidenceProvenance(
            session_id=result.session_id,
            investigation_id=result.investigation_id,
            task_id=result.task_id,
            attempt_id=result.attempt_id,
            run_id=result.run_id,
            tool_call_id=observation.observation_id,
            source_id=source_id,
            document_id=document_id,
            chunk_id=chunk_id,
            source_reference=source_reference,
        )
        return Evidence(
            investigation_id=result.investigation_id,
            source=source_reference,
            claim=conclusion.claim,
            provenance=provenance,
            confidence=conclusion.confidence,
            excerpt=_observation_excerpt(observation),
            payload=dict(observation.input),
            role=EvidenceRole.SUPPORTING,
            source_quality=_hint_float(hints, "source_quality", 0.5),
            evidence_id=_stable_id(
                "evidence", conclusion.claim.claim_id, observation.observation_id
            ),
            timestamp=observation.timestamp,
            metadata={
                "conclusion_id": conclusion.conclusion_id,
                "observation_id": observation.observation_id,
            },
        )


def _malformed_conclusion_diagnostics(result: InvestigationResult) -> list[ExtractionDiagnostic]:
    raw = result.metadata.get("malformed_conclusions")
    if not isinstance(raw, list):
        return []
    diagnostics: list[ExtractionDiagnostic] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        reference = item.get("conclusion_id")
        if not isinstance(reference, str) or not reference.strip():
            reference = f"index:{item.get('index', '?')}"
        message = item.get("reason")
        if not isinstance(message, str) or not message.strip():
            message = "malformed conclusion"
        diagnostics.append(ExtractionDiagnostic(str(reference), "malformed_conclusion", message))
    return diagnostics


def _deduplicated_evidence(candidates: tuple[CandidateClaim, ...]) -> tuple[Evidence, ...]:
    unique: dict[str, Evidence] = {}
    for candidate in candidates:
        for item in candidate.evidence:
            unique.setdefault(item.evidence_id, item)
    return tuple(unique[key] for key in sorted(unique))


def _observation_excerpt(observation: ToolObservation) -> str:
    output = observation.output or {}
    for key in ("excerpt", "text"):
        value = output.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for source in (output, observation.input):
        if source:
            return json.dumps(source, sort_keys=True, separators=(",", ":"), default=str)
    return f"{observation.tool_name} executed"


def _hint(hints: Mapping[str, Any], key: str, default: str) -> str:
    value = hints.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _hint_float(hints: Mapping[str, Any], key: str, default: float) -> float:
    value = hints.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return min(1.0, max(0.0, float(value)))


__all__ = [
    "CandidateClaim",
    "CandidateClaimExtractor",
    "CandidateExtractionResult",
    "CandidateStatus",
    "ExtractionDiagnostic",
]
