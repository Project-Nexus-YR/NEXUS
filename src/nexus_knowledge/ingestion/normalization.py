"""Text normalization and deterministic chunking."""

from __future__ import annotations

import re

from ..domain.document import Chunk, Document, Span

__all__ = ["normalize_text", "RecursiveChunker"]

_SPACES = re.compile(r"[ \t]+")
_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = _NEWLINES.sub("\n\n", text)
    return text.strip()


class RecursiveChunker:
    """Splits a document into overlapping chunks at natural boundaries.

    Splits first on paragraphs, then sentences, then words, so that
    chunks stay within ``max_chars`` while minimizing mid-sentence cuts.
    Character spans are computed against the *document* text.
    """

    def __init__(self, max_chars: int = 800, overlap: int = 100) -> None:
        if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
            raise ValueError("invalid chunker limits")
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        spans = self._split_into_spans(text)
        chunks: list[Chunk] = []
        for index, (start, end) in enumerate(spans):
            chunks.append(
                Chunk(
                    document_id=document.id,
                    index=index,
                    text=text[start:end],
                    span=Span(start, end),
                )
            )
        return chunks

    def _split_into_spans(self, text: str) -> list[tuple[int, int]]:
        if len(text) <= self.max_chars:
            return [(0, len(text))]

        paragraphs = self._paragraphs(text)
        spans: list[tuple[int, int]] = []
        for start, end in paragraphs:
            if end - start <= self.max_chars:
                spans.append((start, end))
                continue
            spans.extend(self._subsplit(text, start, end))
        return self._add_overlap(text, spans)

    def _paragraphs(self, text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        spans: list[tuple[int, int]] = []
        start = 0
        for match in re.finditer(r"\n\n", text):
            spans.append((start, match.start()))
            start = match.end()
        spans.append((start, len(text)))
        return [s for s in spans if s[1] > s[0]]

    def _subsplit(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        pieces: list[tuple[int, int]] = []
        current = start
        while current < end:
            limit = min(current + self.max_chars, end)
            cut = self._last_sentence_boundary(text, current, limit)
            pieces.append((current, cut))
            current = cut
        return pieces

    def _last_sentence_boundary(self, text: str, start: int, limit: int) -> int:
        window = text[start:limit]
        boundary = -1
        for match in re.finditer(r"[.!?](\s+)", window):
            boundary = match.end(1)
        for match in re.finditer(r"\n", window):
            boundary = max(boundary, match.end())
        if boundary <= 0:
            boundary = len(window)
        return start + boundary

    def _add_overlap(self, text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if self.overlap == 0 or len(spans) <= 1:
            return spans
        result: list[tuple[int, int]] = []
        for i, (start, end) in enumerate(spans):
            if i == 0:
                result.append((start, end))
                continue
            prev_end = result[-1][1]
            overlap_start = max(start, prev_end - self.overlap)
            result.append((overlap_start, end))
        return result
