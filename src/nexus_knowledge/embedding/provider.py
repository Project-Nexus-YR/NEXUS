"""Local embedding provider adapter."""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.common import now_iso
from ..port.embeddings import Embedding
from .hashing import FeatureHashEmbedder

__all__ = ["LocalEmbeddingProvider"]


class LocalEmbeddingProvider:
    """Implements :class:`EmbeddingProvider` with deterministic hashing."""

    def __init__(
        self,
        dimensionality: int = 256,
        model_name: str = "nexus-feature-hash",
        version: str = "v1",
    ) -> None:
        self._embedder = FeatureHashEmbedder(dimensionality)
        self._model_name = model_name
        self._version = version

    @property
    def dimensionality(self) -> int:
        return self._embedder.dimensionality

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        return self._version

    def embed(self, text: str, object_id: str = "") -> Embedding:
        vector = self._embedder.embed(text)
        return Embedding(
            object_id=object_id,
            model=self._model_name,
            version=self._version,
            dimensionality=self.dimensionality,
            vector=tuple(float(x) for x in vector),
            created_at=now_iso(),
        )

    def embed_batch(self, texts: Sequence[str], object_ids: Sequence[str]) -> list[Embedding]:
        return [
            self.embed(text, object_id) for text, object_id in zip(texts, object_ids, strict=True)
        ]
