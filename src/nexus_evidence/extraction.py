"""Deterministic atomic-claim extraction with an optional model-provider port."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Protocol

from .model import AtomicClaim, ClaimType

VERSION = "nexus-claim-extractor/1"
_SENTENCE = re.compile(r"(?m)(?:^|(?<=[.!?]))\s+|(?<=[\"”])\s+(?=[A-Z0-9])|\n+(?=[-*\d])")
_NUMBER = re.compile(r"(?<!\w)[+-]?(?:\d{1,3}(?:[ ,]\d{3})+|\d+)(?:\.\d+)?\s*(?:%|[A-Za-z²³]+)?")
_YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
_QUOTE = re.compile(r"[\"“][^\"”]{2,}[\"”]")
_COMPARATIVE = re.compile(
    r"\b(?:more|less|fewer|higher|lower|larger|smaller|better|worse|than|increase[ds]?|decrease[ds]?)\b",
    re.I,
)
_CAUSAL = re.compile(r"\b(?:causes?|caused|because|due to|results? in|leads? to|drives?)\b", re.I)
_ATTRIBUTION = re.compile(r"\b(?:said|says|stated|reported|according to|claimed|argued)\b", re.I)
_OPINION = re.compile(r"\b(?:I think|we believe|in my view|arguably|should|ought to)\b", re.I)
_DERIVED = re.compile(r"\b(?:therefore|thus|hence|implies?|we calculate|derived from)\b", re.I)
_VERB = re.compile(
    r"\b(is|are|was|were|has|have|had|became|becomes|causes?|supports?|reported|stated|states?|increased|decreased)\b",
    re.I,
)


class ClaimExtractionProvider(Protocol):
    """Optional provider. Output is validated into the deterministic domain schema."""

    def extract_claims(self, text: str) -> list[Mapping[str, Any]]: ...


class ClaimExtractor:
    def __init__(self, provider: ClaimExtractionProvider | None = None) -> None:
        self._provider = provider

    def extract(self, text: str) -> tuple[AtomicClaim, ...]:
        if not isinstance(text, str) or not text.strip():
            return ()
        if self._provider is not None:
            return self._from_provider(text, self._provider.extract_claims(text))
        claims: list[AtomicClaim] = []
        for sentence, start, end in self._sentences(text):
            for atom, atom_start, atom_end in self._split_compound(sentence, start):
                normalized = " ".join(atom.strip(" -*\t").split())
                if not self._assertive(normalized):
                    continue
                subject, predicate, object_value = self._spo(normalized)
                claim_type = self.classify(normalized)
                claims.append(
                    AtomicClaim(
                        text=normalized,
                        claim_type=claim_type,
                        importance=self._importance(claim_type, normalized),
                        subject=subject,
                        predicate=predicate,
                        object=object_value,
                        source_start=atom_start,
                        source_end=min(atom_end, end),
                    )
                )
        unique: dict[str, AtomicClaim] = {}
        for claim in claims:
            unique.setdefault(" ".join(claim.text.casefold().split()), claim)
        return tuple(unique.values())

    @staticmethod
    def classify(text: str) -> ClaimType:
        if _QUOTE.search(text):
            return ClaimType.QUOTATION
        if _OPINION.search(text):
            return ClaimType.OPINION
        if _DERIVED.search(text):
            return ClaimType.DERIVED
        if _CAUSAL.search(text):
            return ClaimType.CAUSAL
        if _COMPARATIVE.search(text):
            return ClaimType.COMPARATIVE
        if _ATTRIBUTION.search(text):
            return ClaimType.ATTRIBUTIONAL
        without_years = _YEAR.sub("", text)
        if _NUMBER.search(without_years):
            return ClaimType.NUMERICAL
        if _YEAR.search(text) or re.search(r"\b(?:before|after|during|since|until)\b", text, re.I):
            return ClaimType.TEMPORAL
        return ClaimType.FACTUAL

    def _from_provider(
        self, text: str, records: list[Mapping[str, Any]]
    ) -> tuple[AtomicClaim, ...]:
        claims: list[AtomicClaim] = []
        for record in records:
            claim_text = str(record.get("text", "")).strip()
            if not claim_text:
                continue
            start = int(record.get("source_start", text.find(claim_text)))
            if start < 0 or text[start : start + len(claim_text)] != claim_text:
                raise ValueError("provider claim must reference an exact span in the input")
            raw_type = str(record.get("claim_type", self.classify(claim_text).value))
            claims.append(
                AtomicClaim(
                    text=claim_text,
                    claim_type=ClaimType(raw_type),
                    importance=float(record.get("importance", 0.5)),
                    subject=str(record.get("subject", "")),
                    predicate=str(record.get("predicate", "")),
                    object=str(record.get("object", "")),
                    qualifiers=tuple(str(item) for item in record.get("qualifiers", [])),
                    source_start=start,
                    source_end=start + len(claim_text),
                )
            )
        return tuple(claims)

    @staticmethod
    def _sentences(text: str) -> list[tuple[str, int, int]]:
        result: list[tuple[str, int, int]] = []
        cursor = 0
        for boundary in _SENTENCE.finditer(text):
            segment = text[cursor : boundary.start()].strip()
            if segment:
                start = text.find(segment, cursor, boundary.start() + 1)
                result.append((segment, start, start + len(segment)))
            cursor = boundary.end()
        segment = text[cursor:].strip()
        if segment:
            start = text.find(segment, cursor)
            result.append((segment, start, start + len(segment)))
        return result

    @staticmethod
    def _split_compound(sentence: str, sentence_start: int) -> list[tuple[str, int, int]]:
        # Split only when both sides contain their own verb phrase; this avoids
        # turning simple noun lists into claims.
        for match in re.finditer(r"\s+(?:and|but|while|whereas)\s+", sentence, re.I):
            left, right = sentence[: match.start()], sentence[match.end() :]
            if _VERB.search(left) and _VERB.search(right):
                right_start = sentence_start + match.end()
                return ClaimExtractor._split_compound(
                    left, sentence_start
                ) + ClaimExtractor._split_compound(right, right_start)
        return [(sentence, sentence_start, sentence_start + len(sentence))]

    @staticmethod
    def _assertive(text: str) -> bool:
        if len(text.split()) < 3 or text.endswith("?"):
            return False
        return bool(_VERB.search(text) or _NUMBER.search(text) or _ATTRIBUTION.search(text))

    @staticmethod
    def _spo(text: str) -> tuple[str, str, str]:
        match = _VERB.search(text)
        if match is None:
            return "", "asserts", text
        return (
            text[: match.start()].strip(" ,"),
            match.group(0).casefold().replace(" ", "_"),
            text[match.end() :].strip(" ,."),
        )

    @staticmethod
    def _importance(claim_type: ClaimType, text: str) -> float:
        base = {
            ClaimType.NUMERICAL: 0.85,
            ClaimType.CAUSAL: 0.85,
            ClaimType.QUOTATION: 0.8,
            ClaimType.COMPARATIVE: 0.75,
            ClaimType.TEMPORAL: 0.7,
            ClaimType.ATTRIBUTIONAL: 0.7,
            ClaimType.DERIVED: 0.7,
            ClaimType.FACTUAL: 0.65,
            ClaimType.OPINION: 0.35,
        }[claim_type]
        return min(1.0, base + (0.05 if len(text.split()) <= 16 else 0.0))
