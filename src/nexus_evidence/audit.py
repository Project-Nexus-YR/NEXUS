"""Citation synthesis, answer auditing, metrics, and repair selection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from .graph import CitationGraph
from .model import (
    AtomicClaim,
    Citation,
    CitationAudit,
    CitationIssue,
    CitationIssueKind,
    CitationMetrics,
    ClaimAssessment,
    ClaimVerdict,
    EvidenceRelation,
)


class CitationSynthesizer:
    def synthesize(
        self,
        graph: CitationGraph,
        assessments: tuple[ClaimAssessment, ...],
        *,
        citations_per_claim: int = 2,
    ) -> tuple[Citation, ...]:
        citations: list[Citation] = []
        for claim_assessment in assessments:
            accepted_ids = set(claim_assessment.supporting_evidence_ids)
            if not accepted_ids:
                continue
            seen_sources: set[str] = set()
            candidates = sorted(
                (
                    item
                    for item in graph.assessments_for_claim(claim_assessment.claim_id)
                    if item.evidence_id in accepted_ids
                    and item.relation == EvidenceRelation.SUPPORTED
                ),
                key=lambda item: (
                    -item.confidence,
                    -graph.sources[graph.evidence[item.evidence_id].source_id].quality,
                    item.evidence_id,
                ),
            )
            for assessment in candidates:
                evidence = graph.evidence[assessment.evidence_id]
                if evidence.source_id in seen_sources:
                    continue
                citations.append(
                    Citation(
                        claim_id=claim_assessment.claim_id,
                        evidence_id=evidence.evidence_id,
                        source_id=evidence.source_id,
                        locator=self._locator(graph, evidence.evidence_id),
                    )
                )
                seen_sources.add(evidence.source_id)
                if len(seen_sources) >= citations_per_claim:
                    break
        return tuple(citations)

    @staticmethod
    def _locator(graph: CitationGraph, evidence_id: str) -> str:
        evidence = graph.evidence[evidence_id]
        source = graph.sources[evidence.source_id]
        details = []
        if evidence.page is not None:
            details.append(f"page {evidence.page}")
        if evidence.section:
            details.append(f"section {evidence.section}")
        if evidence.paragraph is not None:
            details.append(f"paragraph {evidence.paragraph}")
        details.append(f"characters {evidence.char_start}-{evidence.char_end}")
        return f"{source.reference}#" + ";".join(details)


class CitationAuditor:
    def __init__(self, *, stale_after: timedelta = timedelta(days=730)) -> None:
        self._stale_after = stale_after

    def audit(
        self,
        graph: CitationGraph,
        assessments: tuple[ClaimAssessment, ...],
        *,
        repair_round: int = 0,
        now: datetime | None = None,
    ) -> CitationAudit:
        issues: list[CitationIssue] = []
        assessment_by_claim = {item.claim_id: item for item in assessments}
        for claim in graph.claims.values():
            assessment = assessment_by_claim.get(claim.claim_id)
            citations = graph.citations_for_claim(claim.claim_id)
            if assessment is None or assessment.verdict == ClaimVerdict.INSUFFICIENT:
                issues.append(self._issue(CitationIssueKind.UNSUPPORTED_CLAIM, claim))
            elif assessment.verdict == ClaimVerdict.CONTRADICTED:
                issues.append(self._issue(CitationIssueKind.CONTRADICTION, claim, severity=1.0))
            elif assessment.verdict == ClaimVerdict.CONFLICTING_EVIDENCE:
                issues.append(self._issue(CitationIssueKind.CONTRADICTION, claim, severity=0.9))
            elif assessment.verdict == ClaimVerdict.PARTIALLY_SUPPORTED:
                issues.append(self._issue(CitationIssueKind.PARTIAL_SUPPORT, claim, severity=0.7))
            if claim.is_verifiable and not citations:
                issues.append(self._issue(CitationIssueKind.MISSING_CITATION, claim))
            for citation in citations:
                issues.extend(
                    self._citation_issues(graph, claim, citation, now or datetime.now(UTC))
                )
        issues.extend(self._duplicates(graph))
        metrics = self._metrics(graph, assessments)
        return CitationAudit(
            claims=tuple(graph.claims[key] for key in sorted(graph.claims)),
            citations=tuple(graph.citations[key] for key in sorted(graph.citations)),
            assessments=tuple(sorted(assessments, key=lambda item: item.claim_id)),
            issues=tuple(sorted(issues, key=lambda item: item.issue_id)),
            metrics=metrics,
            repair_round=repair_round,
        )

    def repair_claims(self, audit: CitationAudit) -> tuple[str, ...]:
        return tuple(
            sorted(
                {issue.claim_id for issue in audit.issues if issue.repairable and issue.claim_id}
            )
        )

    def _citation_issues(
        self, graph: CitationGraph, claim: AtomicClaim, citation: Citation, now: datetime
    ) -> list[CitationIssue]:
        issues: list[CitationIssue] = []
        evidence = graph.evidence.get(citation.evidence_id)
        source = graph.sources.get(citation.source_id)
        assessment = next(
            (
                item
                for item in graph.assessments_for_claim(claim.claim_id)
                if item.evidence_id == citation.evidence_id
            ),
            None,
        )
        if evidence is None or source is None or assessment is None:
            issues.append(
                CitationIssue(
                    CitationIssueKind.CITATION_MISMATCH,
                    "citation does not resolve to a verified claim-evidence-source path",
                    1.0,
                    claim.claim_id,
                    citation.citation_id,
                    False,
                )
            )
            return issues
        if assessment.relation != EvidenceRelation.SUPPORTED:
            issues.append(
                CitationIssue(
                    CitationIssueKind.CITATION_OVERREACH,
                    "citation is attached beyond the support established by its evidence",
                    0.8,
                    claim.claim_id,
                    citation.citation_id,
                )
            )
        if source.quality < 0.5:
            issues.append(
                CitationIssue(
                    CitationIssueKind.WEAK_CITATION,
                    "citation relies on a low-quality source",
                    0.6,
                    claim.claim_id,
                    citation.citation_id,
                )
            )
        published = self._date(source.published_at)
        if published is not None and now - published > self._stale_after:
            issues.append(
                CitationIssue(
                    CitationIssueKind.STALE_SOURCE,
                    "citation source is older than the configured freshness horizon",
                    0.5,
                    claim.claim_id,
                    citation.citation_id,
                )
            )
        return issues

    @staticmethod
    def _duplicates(graph: CitationGraph) -> list[CitationIssue]:
        issues: list[CitationIssue] = []
        seen: dict[tuple[str, str], Citation] = {}
        for citation in graph.citations.values():
            key = citation.claim_id, citation.source_id
            original = seen.get(key)
            if original is not None:
                issues.append(
                    CitationIssue(
                        CitationIssueKind.DUPLICATE_CITATION,
                        "multiple citations for this claim resolve to the same source",
                        0.3,
                        citation.claim_id,
                        citation.citation_id,
                    )
                )
            else:
                seen[key] = citation
        return issues

    @staticmethod
    def _metrics(graph: CitationGraph, assessments: tuple[ClaimAssessment, ...]) -> CitationMetrics:
        claims = tuple(graph.claims.values())
        verifiable = tuple(item for item in claims if item.is_verifiable)
        verifiable_ids = {item.claim_id for item in verifiable}
        cited_ids = {item.claim_id for item in graph.citations.values()}
        supported = tuple(
            item
            for item in assessments
            if item.claim_id in verifiable_ids and item.verdict == ClaimVerdict.SUPPORTED
        )
        contradicted = tuple(
            item
            for item in assessments
            if item.claim_id in verifiable_ids
            and item.verdict in {ClaimVerdict.CONTRADICTED, ClaimVerdict.CONFLICTING_EVIDENCE}
        )
        unsupported = tuple(
            item
            for item in assessments
            if item.claim_id in verifiable_ids
            and item.verdict
            in {
                ClaimVerdict.INSUFFICIENT,
                ClaimVerdict.UNVERIFIABLE,
                ClaimVerdict.PARTIALLY_SUPPORTED,
            }
        )
        total_importance = sum(item.importance for item in verifiable)
        cited_importance = sum(item.importance for item in verifiable if item.claim_id in cited_ids)
        valid_citations = 0
        qualities: list[float] = []
        for citation in graph.citations.values():
            relation = next(
                (
                    item.relation
                    for item in graph.assessments_for_claim(citation.claim_id)
                    if item.evidence_id == citation.evidence_id
                ),
                None,
            )
            if relation == EvidenceRelation.SUPPORTED:
                valid_citations += 1
            source = graph.sources.get(citation.source_id)
            if source is not None:
                qualities.append(source.quality)
        claim_count = max(1, len(verifiable))
        citation_count = max(1, len(graph.citations))
        return CitationMetrics(
            coverage=len(cited_ids & {item.claim_id for item in verifiable}) / claim_count,
            weighted_coverage=0.0 if total_importance == 0 else cited_importance / total_importance,
            citation_precision=valid_citations / citation_count,
            evidence_support_rate=len(supported) / claim_count,
            unsupported_rate=len(unsupported) / claim_count,
            contradiction_rate=len(contradicted) / claim_count,
            source_quality=0.0 if not qualities else sum(qualities) / len(qualities),
        )

    @staticmethod
    def _issue(
        kind: CitationIssueKind, claim: AtomicClaim, *, severity: float = 0.8
    ) -> CitationIssue:
        return CitationIssue(
            kind,
            f"{kind.value.replace('_', ' ')}: {claim.text}",
            severity,
            claim.claim_id,
        )

    @staticmethod
    def _date(value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=parsed.tzinfo or UTC).astimezone(UTC)
