"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from nexus_knowledge.domain.source import Source, SourceKind
from nexus_knowledge.eval.fixtures import build_corpus
from nexus_knowledge.service.factory import Adapters, create_engine

GAZETTEER = {
    "Company": ["Acme Corp", "Initech", "Umbrella Corp"],
    "Person": ["Ada Lovelace", "Alan Turing", "Grace Hopper", "Nikola Tesla"],
    "City": ["London", "Menlo Park"],
}


@pytest.fixture
def gazetteer():
    return GAZETTEER


@pytest.fixture
def engine(gazetteer):
    return create_engine(Adapters(gazetteer=gazetteer))


@pytest.fixture
def ingested_engine(engine):
    """Engine with a small deterministic corpus ingested."""
    corpus = build_corpus()
    for reference, text in corpus.documents[:4]:
        engine.ingest(Source(title=reference, kind=SourceKind.TEXT, reference=reference), text)
    return engine
