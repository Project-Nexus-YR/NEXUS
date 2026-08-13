"""Immutable domain records for claim-level evidence and citations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def stable_id(prefix: str, *parts: object) -> str:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:24]}"


def utcnow() -> datetime:
    return datetime.now(UTC)


def _required(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _unit_interval(value: float, name: str) -> None:
    if isinstance(value, bool) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between zero and one")


def _jsonable(value: Mapping[str, Any], name: str) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc


class ClaimType(StrEnum):
    FACTUAL = "factual"
    NUMERICAL = "numerical"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    ATTRIBUTIONAL = "attributional"
    QUOTATION = "quotation"
    OPINION = "opinion"
    DERIVED = "derived"


class EvidenceRelation(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"
    UNVERIFIABLE = "unverifiable"


class ClaimVerdict(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    INSUFFICIENT = "insufficient"
    UNVERIFIABLE = "unverifiable"


class CitationIssueKind(StrEnum):
    UNSUPPORTED_CLAIM = "unsupported_claim"
    MISSING_CITATION = "missing_citation"
    WEAK_CITATION = "weak_citation"
    PARTIAL_SUPPORT = "partial_support"
    CONTRADICTION = "contradiction"
    CITATION_MISMATCH = "citation_mismatch"
    CITATION_OVERREACH = "citation_overreach"
    DUPLICATE_CITATION = "duplicate_citation"
    STALE_SOURCE = "stale_source"


@dataclass(frozen=True, slots=True)
class AtomicClaim:
    text: str
    claim_type: ClaimType = ClaimType.FACTUAL
    importance: float = 0.5
    subject: str = ""
    predicate: str = ""
    object: str = ""
    qualifiers: tuple[str, ...] = ()
    source_start: int = 0
    source_end: int = 0
    claim_id: str = ""

    def __post_init__(self) -> None:
        _required(self.text, "claim text")
        _unit_interval(self.importance, "claim importance")
        if self.source_start < 0 or self.source_end < self.source_start:
            raise ValueError("claim source span is invalid")
        if not self.claim_id:
            object.__setattr__(
                self,
                "claim_id",
                stable_id(
                    "atomic_claim",
                    " ".join(self.text.casefold().split()),
                    self.source_start,
                    self.source_end,
                ),
            )

    @property
    def identity(self) -> tuple[str, str, str]:
        return (
            " ".join(self.subject.casefold().split()),
            " ".join(self.predicate.casefold().split()),
            " ".join(self.object.casefold().split()),
        )

    @property
    def spo(self) -> tuple[str, str, str]:
        return self.subject, self.predicate, self.object

    @property
    def is_verifiable(self) -> bool:
        return self.claim_type != ClaimType.OPINION

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "claim_type": self.claim_type.value,
            "importance": self.importance,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "qualifiers": list(self.qualifiers),
            "source_start": self.source_start,
            "source_end": self.source_end,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AtomicClaim:
        return cls(
            claim_id=str(payload["claim_id"]),
            text=str(payload["text"]),
            claim_type=ClaimType(str(payload["claim_type"])),
            importance=float(payload["importance"]),
            subject=str(payload.get("subject", "")),
            predicate=str(payload.get("predicate", "")),
            object=str(payload.get("object", "")),
            qualifiers=tuple(str(item) for item in payload.get("qualifiers", [])),
            source_start=int(payload.get("source_start", 0)),
            source_end=int(payload.get("source_end", 0)),
        )


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    title: str
    reference: str
    source_type: str
    publisher: str = ""
    published_at: str = ""
    retrieved_at: datetime = field(default_factory=utcnow)
    content_hash: str = ""
    quality: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    source_id: str = ""

    def __post_init__(self) -> None:
        _required(self.title, "source title")
        _required(self.reference, "source reference")
        _required(self.source_type, "source type")
        _unit_interval(self.quality, "source quality")
        _jsonable(self.metadata, "source metadata")
        if not self.source_id:
            object.__setattr__(self, "source_id", stable_id("citation_source", self.reference))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "reference": self.reference,
            "source_type": self.source_type,
            "publisher": self.publisher,
            "published_at": self.published_at,
            "retrieved_at": self.retrieved_at.isoformat(),
            "content_hash": self.content_hash,
            "quality": self.quality,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvidenceSource:
        return cls(
            source_id=str(payload["source_id"]),
            title=str(payload["title"]),
            reference=str(payload["reference"]),
            source_type=str(payload["source_type"]),
            publisher=str(payload.get("publisher", "")),
            published_at=str(payload.get("published_at", "")),
            retrieved_at=datetime.fromisoformat(str(payload["retrieved_at"])),
            content_hash=str(payload.get("content_hash", "")),
            quality=float(payload["quality"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    source_id: str
    text: str
    document_id: str
    chunk_id: str
    search_query: str
    retrieval_strategy: str
    rank: int
    page: int | None = None
    section: str = ""
    paragraph: int | None = None
    char_start: int = 0
    char_end: int = 0
    extractor_version: str = "nexus-evidence/1"
    verifier_version: str = "nexus-evidence/1"
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_id: str = ""

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_id, "source_id"),
            (self.text, "evidence text"),
            (self.document_id, "document_id"),
            (self.chunk_id, "chunk_id"),
            (self.search_query, "search_query"),
            (self.retrieval_strategy, "retrieval_strategy"),
        ):
            _required(value, name)
        if self.rank < 1 or self.char_start < 0 or self.char_end < self.char_start:
            raise ValueError("evidence rank or character span is invalid")
        _jsonable(self.metadata, "evidence metadata")
        if not self.evidence_id:
            object.__setattr__(
                self,
                "evidence_id",
                stable_id(
                    "evidence_span",
                    self.source_id,
                    self.document_id,
                    self.chunk_id,
                    self.char_start,
                    self.char_end,
                    hashlib.sha256(self.text.encode()).hexdigest(),
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "text": self.text,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "search_query": self.search_query,
            "retrieval_strategy": self.retrieval_strategy,
            "rank": self.rank,
            "page": self.page,
            "section": self.section,
            "paragraph": self.paragraph,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "extractor_version": self.extractor_version,
            "verifier_version": self.verifier_version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvidenceSpan:
        return cls(
            evidence_id=str(payload["evidence_id"]),
            source_id=str(payload["source_id"]),
            text=str(payload["text"]),
            document_id=str(payload["document_id"]),
            chunk_id=str(payload["chunk_id"]),
            search_query=str(payload["search_query"]),
            retrieval_strategy=str(payload["retrieval_strategy"]),
            rank=int(payload["rank"]),
            page=None if payload.get("page") is None else int(payload["page"]),
            section=str(payload.get("section", "")),
            paragraph=(None if payload.get("paragraph") is None else int(payload["paragraph"])),
            char_start=int(payload.get("char_start", 0)),
            char_end=int(payload.get("char_end", 0)),
            extractor_version=str(payload.get("extractor_version", "nexus-evidence/1")),
            verifier_version=str(payload.get("verifier_version", "nexus-evidence/1")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    claim_id: str
    evidence_id: str
    relation: EvidenceRelation
    confidence: float
    reasons: tuple[str, ...]
    numerical_details: dict[str, Any] = field(default_factory=dict)
    assessment_id: str = ""

    def __post_init__(self) -> None:
        _required(self.claim_id, "claim_id")
        _required(self.evidence_id, "evidence_id")
        _unit_interval(self.confidence, "assessment confidence")
        _jsonable(self.numerical_details, "numerical details")
        if not self.assessment_id:
            object.__setattr__(
                self,
                "assessment_id",
                stable_id("evidence_assessment", self.claim_id, self.evidence_id),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "claim_id": self.claim_id,
            "evidence_id": self.evidence_id,
            "relation": self.relation.value,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "numerical_details": dict(self.numerical_details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvidenceAssessment:
        return cls(
            assessment_id=str(payload["assessment_id"]),
            claim_id=str(payload["claim_id"]),
            evidence_id=str(payload["evidence_id"]),
            relation=EvidenceRelation(str(payload["relation"])),
            confidence=float(payload["confidence"]),
            reasons=tuple(str(item) for item in payload.get("reasons", [])),
            numerical_details=dict(payload.get("numerical_details", {})),
        )


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    claim_id: str
    verdict: ClaimVerdict
    confidence: float
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required(self.claim_id, "claim_id")
        _unit_interval(self.confidence, "claim confidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClaimAssessment:
        return cls(
            claim_id=str(payload["claim_id"]),
            verdict=ClaimVerdict(str(payload["verdict"])),
            confidence=float(payload["confidence"]),
            supporting_evidence_ids=tuple(
                str(item) for item in payload.get("supporting_evidence_ids", [])
            ),
            contradicting_evidence_ids=tuple(
                str(item) for item in payload.get("contradicting_evidence_ids", [])
            ),
            reasons=tuple(str(item) for item in payload.get("reasons", [])),
        )


@dataclass(frozen=True, slots=True)
class Citation:
    claim_id: str
    evidence_id: str
    source_id: str
    locator: str
    citation_id: str = ""

    def __post_init__(self) -> None:
        for value, name in (
            (self.claim_id, "claim_id"),
            (self.evidence_id, "evidence_id"),
            (self.source_id, "source_id"),
            (self.locator, "locator"),
        ):
            _required(value, name)
        if not self.citation_id:
            object.__setattr__(
                self,
                "citation_id",
                stable_id("citation", self.claim_id, self.evidence_id, self.locator),
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "citation_id": self.citation_id,
            "claim_id": self.claim_id,
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "locator": self.locator,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Citation:
        return cls(**{name: str(payload[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class CitationIssue:
    kind: CitationIssueKind
    message: str
    severity: float
    claim_id: str = ""
    citation_id: str = ""
    repairable: bool = True
    issue_id: str = ""

    def __post_init__(self) -> None:
        _required(self.message, "issue message")
        _unit_interval(self.severity, "issue severity")
        if not self.issue_id:
            object.__setattr__(
                self,
                "issue_id",
                stable_id("citation_issue", self.kind.value, self.claim_id, self.citation_id),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "kind": self.kind.value,
            "message": self.message,
            "severity": self.severity,
            "claim_id": self.claim_id,
            "citation_id": self.citation_id,
            "repairable": self.repairable,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CitationIssue:
        return cls(
            issue_id=str(payload["issue_id"]),
            kind=CitationIssueKind(str(payload["kind"])),
            message=str(payload["message"]),
            severity=float(payload["severity"]),
            claim_id=str(payload.get("claim_id", "")),
            citation_id=str(payload.get("citation_id", "")),
            repairable=bool(payload.get("repairable", True)),
        )


@dataclass(frozen=True, slots=True)
class CitationMetrics:
    coverage: float
    weighted_coverage: float
    citation_precision: float
    evidence_support_rate: float
    unsupported_rate: float
    contradiction_rate: float
    source_quality: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _unit_interval(float(getattr(self, name)), name)

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CitationMetrics:
        return cls(**{name: float(payload[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class CitationAudit:
    claims: tuple[AtomicClaim, ...]
    citations: tuple[Citation, ...]
    assessments: tuple[ClaimAssessment, ...]
    issues: tuple[CitationIssue, ...]
    metrics: CitationMetrics
    repair_round: int = 0
    audit_id: str = ""
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if self.repair_round < 0:
            raise ValueError("repair_round cannot be negative")
        if not self.audit_id:
            object.__setattr__(
                self,
                "audit_id",
                stable_id(
                    "citation_audit",
                    *(claim.claim_id for claim in self.claims),
                    *(citation.citation_id for citation in self.citations),
                    self.repair_round,
                ),
            )

    @property
    def passes(self) -> bool:
        return not any(issue.severity >= 0.5 for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "claims": [item.to_dict() for item in self.claims],
            "citations": [item.to_dict() for item in self.citations],
            "assessments": [item.to_dict() for item in self.assessments],
            "issues": [item.to_dict() for item in self.issues],
            "metrics": self.metrics.to_dict(),
            "repair_round": self.repair_round,
            "passes": self.passes,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CitationAudit:
        return cls(
            audit_id=str(payload["audit_id"]),
            claims=tuple(AtomicClaim.from_dict(item) for item in payload.get("claims", [])),
            citations=tuple(Citation.from_dict(item) for item in payload.get("citations", [])),
            assessments=tuple(
                ClaimAssessment.from_dict(item) for item in payload.get("assessments", [])
            ),
            issues=tuple(CitationIssue.from_dict(item) for item in payload.get("issues", [])),
            metrics=CitationMetrics.from_dict(payload["metrics"]),
            repair_round=int(payload.get("repair_round", 0)),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
        )
