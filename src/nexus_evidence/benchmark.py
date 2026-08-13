"""Deterministic benchmark for extraction, verification, contradiction, and citations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .extraction import ClaimExtractor
from .model import AtomicClaim, ClaimType, EvidenceRelation, EvidenceSource, EvidenceSpan
from .planning import SearchQuery
from .retrieval import RetrievedEvidence
from .verification import EvidenceVerifier
from .workflow import EvidenceEngine, EvidenceWorkflowPolicy


@dataclass(frozen=True, slots=True)
class EvidenceBenchmarkReport:
    claim_extraction_precision: float
    claim_extraction_recall: float
    evidence_retrieval_recall: float
    support_classification_accuracy: float
    contradiction_detection_accuracy: float
    citation_coverage: float
    citation_precision: float
    end_to_end_answer_support_rate: float
    cases: int

    def to_dict(self) -> dict[str, float | int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True, slots=True)
class VerificationCase:
    claim: AtomicClaim
    evidence: str
    relation: EvidenceRelation


class _FixtureRetriever:
    def __init__(self, records: tuple[RetrievedEvidence, ...]) -> None:
        self._records = records

    def search(self, query: SearchQuery, *, limit: int = 5) -> tuple[RetrievedEvidence, ...]:
        material = {token.casefold().strip('".,') for token in query.text.split()}
        ranked = [
            item
            for item in self._records
            if len(material & {token.casefold().strip(".,") for token in item.span.text.split()})
            >= 2
        ]
        return tuple(ranked[:limit])


def run_benchmark() -> EvidenceBenchmarkReport:
    draft = (
        "Acme revenue was 12% in 2025. "
        "The Borealis route is 2 km long. "
        'Acme CEO stated "Demand remains strong."'
    )
    expected_claims = {
        "Acme revenue was 12% in 2025.",
        "The Borealis route is 2 km long.",
        'Acme CEO stated "Demand remains strong."',
    }
    extracted = ClaimExtractor().extract(draft)
    extracted_text = {item.text for item in extracted}
    true_positive = len(expected_claims & extracted_text)
    extraction_precision = true_positive / max(1, len(extracted_text))
    extraction_recall = true_positive / len(expected_claims)

    cases = (
        VerificationCase(
            AtomicClaim("Acme revenue was 12% in 2025", ClaimType.NUMERICAL),
            "Acme revenue was 12 percent in 2025.",
            EvidenceRelation.SUPPORTED,
        ),
        VerificationCase(
            AtomicClaim("The Borealis route is 2 km long", ClaimType.NUMERICAL),
            "The Borealis route is 2000 meters long.",
            EvidenceRelation.SUPPORTED,
        ),
        VerificationCase(
            AtomicClaim("Acme is profitable", ClaimType.FACTUAL),
            "The audited accounts show Acme is not profitable.",
            EvidenceRelation.CONTRADICTED,
        ),
        VerificationCase(
            AtomicClaim("Acme revenue was 12% in 2025", ClaimType.NUMERICAL),
            "Acme revenue was 9 percent in 2025.",
            EvidenceRelation.CONTRADICTED,
        ),
    )
    verifier = EvidenceVerifier()
    predicted: list[EvidenceRelation] = []
    for index, case in enumerate(cases):
        predicted.append(verifier.assess(case.claim, _span(case.evidence, index)).relation)
    support_cases = [
        index for index, case in enumerate(cases) if case.relation == EvidenceRelation.SUPPORTED
    ]
    contradiction_cases = [
        index for index, case in enumerate(cases) if case.relation == EvidenceRelation.CONTRADICTED
    ]
    support_accuracy = sum(
        predicted[index] == cases[index].relation for index in support_cases
    ) / len(support_cases)
    contradiction_accuracy = sum(
        predicted[index] == cases[index].relation for index in contradiction_cases
    ) / len(contradiction_cases)

    source = EvidenceSource(
        "Primary benchmark record",
        "https://benchmark.example/primary",
        "primary",
        quality=0.95,
        source_id="benchmark-source",
    )
    records = tuple(
        RetrievedEvidence(source, _span(text, index, source.source_id), 1.0)
        for index, text in enumerate(
            (
                "Acme revenue was 12 percent in 2025.",
                "The Borealis route is 2000 meters long.",
                'Acme CEO stated "Demand remains strong."',
            )
        )
    )
    report = EvidenceEngine(
        _FixtureRetriever(records),
        policy=EvidenceWorkflowPolicy(max_repair_rounds=0),
    ).verify_answer(draft, session_id="benchmark")
    supported_claims = sum(item.verdict.value == "supported" for item in report.audit.assessments)
    retrieval_recall = supported_claims / max(1, len(report.audit.assessments))
    return EvidenceBenchmarkReport(
        claim_extraction_precision=extraction_precision,
        claim_extraction_recall=extraction_recall,
        evidence_retrieval_recall=retrieval_recall,
        support_classification_accuracy=support_accuracy,
        contradiction_detection_accuracy=contradiction_accuracy,
        citation_coverage=report.audit.metrics.coverage,
        citation_precision=report.audit.metrics.citation_precision,
        end_to_end_answer_support_rate=report.audit.metrics.evidence_support_rate,
        cases=len(cases),
    )


def _span(text: str, index: int, source_id: str = "benchmark-source") -> EvidenceSpan:
    return EvidenceSpan(
        source_id,
        text,
        f"benchmark-document-{index}",
        f"benchmark-chunk-{index}",
        "benchmark query",
        "benchmark",
        index + 1,
        char_end=len(text),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nexus-evidence-bench", description=__doc__)
    parser.add_argument("--output", help="write the JSON report to this path")
    args = parser.parse_args(argv)
    report = run_benchmark()
    rendered = report.to_json()
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
