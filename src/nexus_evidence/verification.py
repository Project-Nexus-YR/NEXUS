"""Deterministic lexical, contradiction, quotation, and numerical verification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .model import (
    AtomicClaim,
    ClaimAssessment,
    ClaimType,
    ClaimVerdict,
    EvidenceAssessment,
    EvidenceRelation,
    EvidenceSpan,
)

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_NEGATION = re.compile(
    r"\b(?:not|never|no|neither|false|incorrect|didn't|doesn't|isn't|wasn't)\b", re.I
)
_QUANTITY = re.compile(
    r"(?<![\w.])(?P<value>[+-]?(?:\d{1,3}(?:[ ,]\d{3})+|\d+)(?:\.\d+)?)"
    r"\s*(?P<unit>%|percent|percentage points?|km|kilometers?|m|meters?|kg|kilograms?|g|grams?|"
    r"million|billion|trillion|usd|eur|gbp|dollars?|euros?|years?)?"
    r"(?=\s|[.,;:!?)]|$)",
    re.I,
)
_STOP = frozenset(
    (
        "a an and are as at be because by for from has have in is it of on or that the "
        "this to was were with"
    ).split()
)
_UPWARD = frozenset({"more", "higher", "larger", "better", "increased", "increase"})
_DOWNWARD = frozenset({"less", "fewer", "lower", "smaller", "worse", "decreased", "decrease"})
_UNIT = {
    "percent": "%",
    "percentage point": "percentage_point",
    "percentage points": "percentage_point",
    "kilometer": "m",
    "kilometers": "m",
    "km": "m",
    "meter": "m",
    "meters": "m",
    "m": "m",
    "kilogram": "g",
    "kilograms": "g",
    "kg": "g",
    "gram": "g",
    "grams": "g",
    "g": "g",
    "dollar": "usd",
    "dollars": "usd",
    "euro": "eur",
    "euros": "eur",
    "year": "year",
    "years": "year",
}
_MULTIPLIER = {
    "km": Decimal(1000),
    "kg": Decimal(1000),
    "million": Decimal(1_000_000),
    "billion": Decimal(1_000_000_000),
    "trillion": Decimal(1_000_000_000_000),
}


@dataclass(frozen=True, slots=True)
class Quantity:
    value: Decimal
    unit: str
    raw: str

    @property
    def normalized(self) -> Decimal:
        return self.value * _MULTIPLIER.get(self.unit, Decimal(1))

    @property
    def dimension(self) -> str:
        return _UNIT.get(self.unit, self.unit)


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    support_overlap: float = 0.62
    partial_overlap: float = 0.3
    numerical_relative_tolerance: float = 0.001
    minimum_source_count: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.partial_overlap <= self.support_overlap <= 1:
            raise ValueError("lexical overlap thresholds are invalid")
        if self.numerical_relative_tolerance < 0 or self.minimum_source_count < 1:
            raise ValueError("verification policy bounds are invalid")


class EvidenceVerifier:
    def __init__(self, policy: VerificationPolicy | None = None) -> None:
        self.policy = policy or VerificationPolicy()

    def assess(self, claim: AtomicClaim, evidence: EvidenceSpan) -> EvidenceAssessment:
        if not claim.is_verifiable:
            return self._result(
                claim,
                evidence,
                EvidenceRelation.UNVERIFIABLE,
                1.0,
                "unattributed opinion has no objective verification target",
            )
        overlap = self._overlap(claim.text, evidence.text)
        negated = bool(_NEGATION.search(claim.text)) != bool(_NEGATION.search(evidence.text))
        if claim.claim_type == ClaimType.QUOTATION:
            return self._quotation(claim, evidence, overlap)
        if claim.claim_type == ClaimType.NUMERICAL:
            return self._numerical(claim, evidence, overlap, negated)
        if claim.claim_type == ClaimType.COMPARATIVE:
            comparison = self._comparative(claim.text, evidence.text)
            if comparison == "opposite" and overlap >= self.policy.partial_overlap:
                return self._result(
                    claim,
                    evidence,
                    EvidenceRelation.CONTRADICTED,
                    0.95,
                    "evidence states the opposite comparative direction",
                )
            if comparison == "missing" and overlap >= self.policy.partial_overlap:
                return self._result(
                    claim,
                    evidence,
                    EvidenceRelation.PARTIALLY_SUPPORTED,
                    0.7,
                    "evidence addresses the entities but not the claimed comparison",
                )
        if negated and overlap >= self.policy.partial_overlap:
            return self._result(
                claim,
                evidence,
                EvidenceRelation.CONTRADICTED,
                min(0.98, 0.55 + overlap / 2),
                "matching assertion has opposite negation polarity",
            )
        if overlap >= self.policy.support_overlap:
            return self._result(
                claim,
                evidence,
                EvidenceRelation.SUPPORTED,
                min(0.98, 0.5 + overlap / 2),
                f"evidence covers {overlap:.0%} of material claim terms",
            )
        if overlap >= self.policy.partial_overlap:
            return self._result(
                claim,
                evidence,
                EvidenceRelation.PARTIALLY_SUPPORTED,
                min(0.8, 0.35 + overlap / 2),
                f"evidence covers only {overlap:.0%} of material claim terms",
            )
        return self._result(
            claim,
            evidence,
            EvidenceRelation.INSUFFICIENT,
            max(0.1, 1 - overlap),
            "evidence does not address enough material claim terms",
        )

    def aggregate(
        self,
        claim: AtomicClaim,
        assessments: tuple[EvidenceAssessment, ...],
        source_by_evidence: dict[str, str] | None = None,
    ) -> ClaimAssessment:
        source_by_evidence = source_by_evidence or {}
        supports = tuple(
            item for item in assessments if item.relation == EvidenceRelation.SUPPORTED
        )
        partials = tuple(
            item for item in assessments if item.relation == EvidenceRelation.PARTIALLY_SUPPORTED
        )
        contradictions = tuple(
            item for item in assessments if item.relation == EvidenceRelation.CONTRADICTED
        )
        independent = {
            source_by_evidence.get(item.evidence_id, item.evidence_id) for item in supports
        }
        if supports and contradictions:
            verdict = ClaimVerdict.CONFLICTING_EVIDENCE
            confidence = max(item.confidence for item in supports + contradictions)
            reasons = ("credible evidence supports both sides of the claim",)
        elif contradictions:
            verdict = ClaimVerdict.CONTRADICTED
            confidence = max(item.confidence for item in contradictions)
            reasons = ("contradicting evidence was found",)
        elif supports and len(independent) >= self.policy.minimum_source_count:
            verdict = ClaimVerdict.SUPPORTED
            confidence = 1.0
            for item in supports:
                confidence *= 1.0 - item.confidence
            confidence = 1.0 - confidence
            reasons = (f"supported by {len(independent)} independent source(s)",)
        elif partials:
            verdict = ClaimVerdict.PARTIALLY_SUPPORTED
            confidence = max(item.confidence for item in partials)
            reasons = ("available evidence supports only part of the claim",)
        elif not claim.is_verifiable:
            verdict = ClaimVerdict.UNVERIFIABLE
            confidence = 1.0
            reasons = ("claim does not identify a verifiable attribution",)
        else:
            verdict = ClaimVerdict.INSUFFICIENT
            confidence = max((item.confidence for item in assessments), default=1.0)
            reasons = ("no sufficient evidence was found",)
        return ClaimAssessment(
            claim_id=claim.claim_id,
            verdict=verdict,
            confidence=confidence,
            supporting_evidence_ids=tuple(item.evidence_id for item in supports),
            contradicting_evidence_ids=tuple(item.evidence_id for item in contradictions),
            reasons=reasons,
        )

    def _quotation(
        self, claim: AtomicClaim, evidence: EvidenceSpan, overlap: float
    ) -> EvidenceAssessment:
        match = re.search(r"[\"“]([^\"”]+)[\"”]", claim.text)
        quote = "" if match is None else " ".join(match.group(1).casefold().split())
        evidence_normalized = " ".join(evidence.text.casefold().split())
        if quote and quote in evidence_normalized:
            return self._result(
                claim,
                evidence,
                EvidenceRelation.SUPPORTED,
                0.99,
                "exact quoted words occur in evidence",
            )
        relation = (
            EvidenceRelation.PARTIALLY_SUPPORTED
            if overlap >= self.policy.partial_overlap
            else EvidenceRelation.INSUFFICIENT
        )
        return self._result(
            claim, evidence, relation, 0.8, "evidence does not contain the exact attributed quote"
        )

    def _numerical(
        self, claim: AtomicClaim, evidence: EvidenceSpan, overlap: float, negated: bool
    ) -> EvidenceAssessment:
        expected = self.quantities(claim.text)
        observed = self.quantities(evidence.text)
        if not expected:
            return self._result(
                claim,
                evidence,
                EvidenceRelation.UNVERIFIABLE,
                1.0,
                "claim has no parseable quantity",
            )
        matches = [self._matches(quantity, observed) for quantity in expected]
        details = {
            "expected": [self._quantity_dict(item) for item in expected],
            "observed": [self._quantity_dict(item) for item in observed],
            "matched": matches,
            "relative_tolerance": self.policy.numerical_relative_tolerance,
        }
        same_dimensions = {item.dimension for item in expected if item.dimension} & {
            item.dimension for item in observed if item.dimension
        }
        if all(matches) and overlap >= self.policy.partial_overlap and not negated:
            return self._result(
                claim,
                evidence,
                EvidenceRelation.SUPPORTED,
                0.98,
                "all claim quantities, units, and material terms match",
                details,
            )
        conflicting_quantity = any(
            not matched
            and expected_item.dimension
            and any(actual.dimension == expected_item.dimension for actual in observed)
            for expected_item, matched in zip(expected, matches, strict=True)
        )
        if same_dimensions and conflicting_quantity and overlap >= self.policy.partial_overlap:
            return self._result(
                claim,
                evidence,
                EvidenceRelation.CONTRADICTED,
                0.95,
                "evidence states conflicting values for the same numerical dimension",
                details,
            )
        if any(matches):
            return self._result(
                claim,
                evidence,
                EvidenceRelation.PARTIALLY_SUPPORTED,
                0.8,
                "only some claim quantities match the evidence",
                details,
            )
        return self._result(
            claim,
            evidence,
            EvidenceRelation.INSUFFICIENT,
            0.75,
            "evidence lacks comparable quantities or material context",
            details,
        )

    @staticmethod
    def quantities(text: str) -> tuple[Quantity, ...]:
        quantities: list[Quantity] = []
        for match in _QUANTITY.finditer(text):
            raw_value = match.group("value").replace(" ", "").replace(",", "")
            try:
                value = Decimal(raw_value)
            except InvalidOperation:
                continue
            unit = (match.group("unit") or "").casefold()
            quantities.append(Quantity(value, unit, match.group(0)))
        return tuple(quantities)

    def _matches(self, expected: Quantity, observed: tuple[Quantity, ...]) -> bool:
        for actual in observed:
            if expected.dimension != actual.dimension:
                continue
            tolerance = max(
                Decimal("0.000000001"),
                abs(expected.normalized) * Decimal(str(self.policy.numerical_relative_tolerance)),
            )
            if abs(expected.normalized - actual.normalized) <= tolerance:
                return True
        return False

    @staticmethod
    def _overlap(claim: str, evidence: str) -> float:
        claim_tokens = {item.casefold() for item in _TOKEN.findall(claim)} - _STOP
        evidence_tokens = {item.casefold() for item in _TOKEN.findall(evidence)} - _STOP
        return 0.0 if not claim_tokens else len(claim_tokens & evidence_tokens) / len(claim_tokens)

    @staticmethod
    def _comparative(claim: str, evidence: str) -> str:
        claim_tokens = {item.casefold() for item in _TOKEN.findall(claim)}
        evidence_tokens = {item.casefold() for item in _TOKEN.findall(evidence)}
        claim_up = bool(claim_tokens & _UPWARD)
        claim_down = bool(claim_tokens & _DOWNWARD)
        evidence_up = bool(evidence_tokens & _UPWARD)
        evidence_down = bool(evidence_tokens & _DOWNWARD)
        if (claim_up and evidence_down) or (claim_down and evidence_up):
            return "opposite"
        if (claim_up and not evidence_up) or (claim_down and not evidence_down):
            return "missing"
        return "same"

    @staticmethod
    def _quantity_dict(quantity: Quantity) -> dict[str, str]:
        return {
            "raw": quantity.raw,
            "value": str(quantity.value),
            "normalized": str(quantity.normalized),
            "unit": quantity.unit,
            "dimension": quantity.dimension,
        }

    @staticmethod
    def _result(
        claim: AtomicClaim,
        evidence: EvidenceSpan,
        relation: EvidenceRelation,
        confidence: float,
        reason: str,
        numerical_details: dict[str, object] | None = None,
    ) -> EvidenceAssessment:
        return EvidenceAssessment(
            claim_id=claim.claim_id,
            evidence_id=evidence.evidence_id,
            relation=relation,
            confidence=confidence,
            reasons=(reason,),
            numerical_details=dict(numerical_details or {}),
        )
