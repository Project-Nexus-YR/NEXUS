"""Embedding provider and vector store tests."""

import numpy as np
import pytest

from nexus_knowledge.embedding.hashing import FeatureHashEmbedder, tokenize
from nexus_knowledge.embedding.local_store import LocalVectorStore
from nexus_knowledge.embedding.provider import LocalEmbeddingProvider
from nexus_knowledge.port.embeddings import Embedding


class TestTokenizer:
    def test_tokenize(self):
        assert tokenize("Hello, world!") == ["hello", "world"]

    def test_empty(self):
        assert tokenize("") == []


class TestFeatureHashEmbedder:
    def test_deterministic(self):
        embedder = FeatureHashEmbedder(dimensionality=32)
        assert np.array_equal(embedder.embed("same text"), embedder.embed("same text"))

    def test_dimension(self):
        embedder = FeatureHashEmbedder(dimensionality=64)
        assert len(embedder.embed("x")) == 64

    def test_different_inputs_differ(self):
        embedder = FeatureHashEmbedder(dimensionality=32)
        assert not np.array_equal(embedder.embed("alpha"), embedder.embed("beta"))


class TestLocalEmbeddingProvider:
    def test_embed_returns_embedding(self):
        provider = LocalEmbeddingProvider(dimensionality=16, model_name="hash")
        embedding = provider.embed("text", object_id="o1")
        assert embedding.object_id == "o1"
        assert embedding.dimensionality == 16
        assert embedding.model == "hash"
        assert len(embedding.vector) == 16

    def test_batch(self):
        provider = LocalEmbeddingProvider(dimensionality=16)
        embeddings = provider.embed_batch(["a", "b"], ["o1", "o2"])
        assert [e.object_id for e in embeddings] == ["o1", "o2"]


class TestLocalVectorStore:
    def _store(self):
        store = LocalVectorStore()
        provider = LocalEmbeddingProvider(dimensionality=16)
        store.upsert(provider.embed("quantum computing", object_id="q"))
        store.upsert(provider.embed("baking recipes", object_id="b"))
        return store

    def test_upsert_overwrites(self):
        store = LocalVectorStore()
        provider = LocalEmbeddingProvider(dimensionality=16)
        store.upsert(provider.embed("a", object_id="o"))
        first = store.get("o")
        store.upsert(provider.embed("b", object_id="o"))
        assert store.get("o") is not first
        assert store.size() == 1

    def test_query_ranking(self):
        store = self._store()
        provider = LocalEmbeddingProvider(dimensionality=16)
        hits = store.query(provider.embed("quantum computing").vector, top_k=5)
        assert hits[0].object_id == "q"
        assert hits[0].score > hits[1].score

    def test_query_filter(self):
        store = LocalVectorStore()
        provider = LocalEmbeddingProvider(dimensionality=16)
        store.upsert(provider.embed("x", object_id="o1"))
        store.upsert(provider.embed("x", object_id="o2"))
        filtered = store.query(provider.embed("x").vector, metadata_filter={"source": "a"})
        assert filtered == []
        assert len(store.query(provider.embed("x").vector, top_k=10)) == 2

    def test_delete(self):
        store = self._store()
        assert store.delete("q") is True
        assert store.delete("q") is False
        assert store.size() == 1

    def test_negative_top_k(self):
        store = LocalVectorStore()
        with pytest.raises(ValueError):
            store.query((0.0,) * 4, top_k=-1)
