"""Deterministic extraction adapters.

Pattern/gazetteer-based entity and relation extractors. Deterministic,
dependency-free, and used both as the default local adapters and as
reference implementations for tests. LLM-based adapters are plugged in
through the same ports without leaking into the domain layer.
"""

from __future__ import annotations

import re

from ..domain.common import Confidence
from ..domain.document import Chunk, Document, Span
from ..port.extractors import ExtractedEntity, ExtractedRelation

__all__ = ["GazetteerEntityExtractor", "PatternRelationExtractor"]

_TOKEN_RE = re.compile(r"[^\s,.;:()]+")

_RELATION_PATTERNS: list[tuple[str, str]] = [
    (r"works?\s+at", "works_at"),
    (r"is?\s+the?\s+founder\s+of", "founded"),
    (r"founded", "founded"),
    (r"located\s+in", "located_in"),
    (r"is?\s+part\s+of", "part_of"),
    (r"headquartered\s+in", "headquartered_in"),
    (r"discovered\s+by", "discovered_by"),
    (r"developed\s+by", "developed_by"),
    (r"caused\s+by", "caused_by"),
    (r"invented", "invented_by"),
    (r"studied\s+at", "studied_at"),
    (r"produced\s+by", "produced_by"),
    (r"is?\s+a?\s+type\s+of", "type_of"),
]

_SENTENCE_RE = re.compile(r"[.!?]\s+")


def _sentence_bounds(text: str, position: int) -> tuple[int, int]:
    """Return ``(start, end)`` of the sentence containing ``position``."""
    start = 0
    for match in _SENTENCE_RE.finditer(text):
        if match.end() > position:
            break
        start = match.end()
    return start, len(text)


class GazetteerEntityExtractor:
    """Extracts entities by matching a gazetteer of known names.

    Supports multi-word names. Matching is longest-first and
    case-insensitive; extracted names preserve the casing found in the
    source text.
    """

    def __init__(self, gazetteer: dict[str, list[str]] | None = None) -> None:
        self._terms: dict[str, list[str]] = {}  # lowered phrase -> [entity_type]
        self._max_words = 0
        for entity_type, names in (gazetteer or {}).items():
            for name in names:
                self.add_term(name, entity_type)

    def add_term(self, name: str, entity_type: str) -> None:
        phrase = self._normalize(name)
        if not phrase:
            return
        self._terms.setdefault(phrase, []).append(entity_type)
        self._max_words = max(self._max_words, len(phrase.split()))

    @staticmethod
    def _normalize(name: str) -> str:
        return " ".join(name.lower().split())

    def extract(self, chunk: Chunk, document: Document) -> list[ExtractedEntity]:
        text = chunk.text
        tokens: list[tuple[int, int, str]] = [
            (m.start(), m.end(), m.group(0).lower()) for m in _TOKEN_RE.finditer(text)
        ]
        found: list[tuple[int, int, str, str]] = []  # (start, end, name, type)
        i = 0
        while i < len(tokens):
            matched = self._longest_match(tokens, i)
            if matched is None:
                i += 1
                continue
            start_idx, end_idx, phrase, entity_type = matched
            start = tokens[start_idx][0]
            end = tokens[end_idx][1]
            name = text[start:end].strip()
            found.append((start, end, name, entity_type))
            i = end_idx + 1
        return [
            ExtractedEntity(
                name=name,
                entity_type=entity_type,
                span=Span(start, end),
                confidence=Confidence(1.0),
            )
            for start, end, name, entity_type in found
        ]

    def _longest_match(
        self,
        tokens: list[tuple[int, int, str]],
        start: int,
    ) -> tuple[int, int, str, str] | None:
        for width in range(self._max_words, 0, -1):
            if start + width > len(tokens):
                continue
            phrase = " ".join(tokens[start + k][2] for k in range(width))
            entity_types = self._terms.get(phrase)
            if entity_types:
                return start, start + width - 1, phrase, entity_types[0]
        return None


class PatternRelationExtractor:
    """Extracts subject-predicate-object relations with regex patterns.

    For each pattern occurrence the nearest entity name in the same
    sentence before and after the match is used as subject/object,
    producing a source span that points at the pattern itself.
    """

    def __init__(self, patterns: list[tuple[str, str]] | None = None) -> None:
        self._patterns = list(patterns or _RELATION_PATTERNS)
        self._compiled = [(re.compile(p, re.IGNORECASE), pred) for p, pred in self._patterns]

    def extract(
        self,
        chunk: Chunk,
        document: Document,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelation]:
        text = chunk.text
        results: list[ExtractedRelation] = []
        for pattern, predicate in self._compiled:
            for match in pattern.finditer(text):
                subject = self._nearest_entity(text, entities, match.start(), before=True)
                object_ = self._nearest_entity(text, entities, match.end(), before=False)
                if subject is None or object_ is None:
                    continue
                if subject.name == object_.name:
                    continue
                confidence = float(subject.confidence) * float(object_.confidence)
                results.append(
                    ExtractedRelation(
                        subject=subject.name,
                        predicate=predicate,
                        object=object_.name,
                        span=Span(match.start(), match.end()),
                        confidence=Confidence(confidence),
                    )
                )
        return results

    def _nearest_entity(
        self,
        text: str,
        entities: list[ExtractedEntity],
        position: int,
        before: bool,
    ) -> ExtractedEntity | None:
        sentence_start, sentence_end = _sentence_bounds(text, position)
        if before:
            candidates = [
                e for e in entities if e.span is not None
                and e.span.start >= sentence_start
                and e.span.end <= min(position, sentence_end)
            ]
            if not candidates:
                return None
            return max(candidates, key=lambda e: e.span.end)
        candidates = [
            e for e in entities if e.span is not None
            and e.span.start >= max(position, sentence_start)
            and e.span.end <= sentence_end
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda e: e.span.start)
