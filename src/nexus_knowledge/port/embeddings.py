"""Embedding provider port.

Independent of any concrete model (sentence-transformers, OpenAI,
local hashing, ...). The local deterministic implementation lives in
:mod:`nexus_knowledge.embedding`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..domain.common import now_iso
from ..domain.ids import stable_id

__all__ = ["Embedding", "EmbeddingProvider"]


@dataclass(frozen=True, slots=True)
class Embedding:
    """A stored vector for a domain object.

    ``object_id`` references the embedded object (chunk, document,
    claim, ...). ``model``/``version`` let retrieval invalidate or
    re-embed when the model changes.
    """

    object_id: str
    model: str
    version: str
    dimensionality: int
    vector: tuple[float, ...]
    created_at: str = field(default_factory=now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def embedding_id(self) -> str:
        return stable_id("emb", self.object_id, self.model, self.version)

    def __hash__(self) -> int:
        return hash(self.embedding_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Embedding) and other.embedding_id == self.embedding_id


class EmbeddingProvider(Protocol):
    """Embeds text into dense vectors."""

    @property
    def dimensionality(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def embed(self, text: str, object_id: str = "") -> Embedding: ...

    def embed_batch(self, texts: Sequence[str], object_ids: Sequence[str]) -> list[Embedding]: ...
