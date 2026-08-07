"""Deterministic fixture corpus for benchmarks and tests.

Builds a realistic, self-contained corpus of organizations, people,
cities and technologies with a known ground truth (relations, query
relevance, claim labels). Deterministic, so benchmarks are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["QueryJudgement", "Corpus", "build_corpus"]

GAZETTEER: dict[str, list[str]] = {
    "Company": ["Acme Corp", "Globex", "Initech", "Umbrella Corp", "Sterling Labs"],
    "Person": ["Ada Lovelace", "Alan Turing", "Grace Hopper", "Nikola Tesla", "Marie Curie"],
    "City": ["London", "Cambridge", "Menlo Park", "Manhattan", "Vienna"],
    "Technology": ["analytics software", "quantum computing", "database systems", "cryptography"],
    "Field": ["computing", "mathematics", "physics"],
}

DOCUMENTS: list[tuple[str, str]] = [
    ("corpus-01", "Ada Lovelace works at Acme Corp. Acme Corp is located in London. Acme Corp develops analytics software."),
    ("corpus-02", "Alan Turing works at Acme Corp. Acme Corp is headquartered in London. Alan Turing studied at Cambridge."),
    ("corpus-03", "Grace Hopper works at Initech. Initech is located in Menlo Park. Grace Hopper is a type of person."),
    ("corpus-04", "Nikola Tesla works at Umbrella Corp. Umbrella Corp is headquartered in Manhattan. Umbrella Corp develops quantum computing."),
    ("corpus-05", "Ada Lovelace studied at Cambridge. Marie Curie studied at Vienna. Marie Curie is a type of person."),
    ("corpus-06", "Acme Corp is a type of company. Sterling Labs is a type of company. Sterling Labs is located in London."),
    ("corpus-07", "Grace Hopper is a type of person. Alan Turing is a type of person. Initech develops database systems."),
]

# ground-truth relations as (subject, predicate, object)
GROUND_TRUTH_RELATIONS: set[tuple[str, str, str]] = {
    ("ada lovelace", "works_at", "acme corp"),
    ("alan turing", "works_at", "acme corp"),
    ("grace hopper", "works_at", "initech"),
    ("nikola tesla", "works_at", "umbrella corp"),
    ("acme corp", "located_in", "london"),
    ("acme corp", "headquartered_in", "london"),
    ("initech", "located_in", "menlo park"),
    ("umbrella corp", "headquartered_in", "manhattan"),
    ("ada lovelace", "studied_at", "cambridge"),
    ("alan turing", "studied_at", "cambridge"),
    ("marie curie", "studied_at", "vienna"),
    ("sterling labs", "located_in", "london"),
    ("acme corp", "develops", "analytics software"),
    ("umbrella corp", "develops", "quantum computing"),
    ("initech", "develops", "database systems"),
}

# (query, relevant document ids, relevant entities, relevant relation tuples)
RETRIEVAL_QUERIES: list[tuple[str, list[str], list[str], list[tuple[str, str, str]]]] = [
    ("who works at Acme Corp?", ["corpus-01", "corpus-02"], ["acme corp"], [
        ("ada lovelace", "works_at", "acme corp"),
        ("alan turing", "works_at", "acme corp"),
    ]),
    ("where is Initech located?", ["corpus-03"], ["initech"], [
        ("initech", "located_in", "menlo park"),
    ]),
    ("people who studied at Cambridge", ["corpus-02", "corpus-05"], ["cambridge"], [
        ("ada lovelace", "studied_at", "cambridge"),
        ("alan turing", "studied_at", "cambridge"),
    ]),
    ("companies in London", ["corpus-01", "corpus-02", "corpus-06"], ["london"], [
        ("acme corp", "located_in", "london"),
        ("acme corp", "headquartered_in", "london"),
        ("sterling labs", "located_in", "london"),
    ]),
    ("what does Umbrella Corp develop?", ["corpus-04"], ["umbrella corp"], [
        ("umbrella corp", "develops", "quantum computing"),
    ]),
]

# (claim text, subject, predicate, object, is_true)
CLAIM_LABELS: list[tuple[str, str, str, str, bool]] = [
    ("Ada Lovelace works at Acme Corp", "Ada Lovelace", "works_at", "Acme Corp", True),
    ("Alan Turing works at Acme Corp", "Alan Turing", "works_at", "Acme Corp", True),
    ("Nikola Tesla works at Initech", "Nikola Tesla", "works_at", "Initech", False),
    ("Grace Hopper works at Umbrella Corp", "Grace Hopper", "works_at", "Umbrella Corp", False),
    ("Acme Corp is located in London", "Acme Corp", "located_in", "London", True),
    ("Umbrella Corp is located in Cambridge", "Umbrella Corp", "located_in", "Cambridge", False),
]


@dataclass(frozen=True, slots=True)
class QueryJudgement:
    query: str
    relevant_documents: list[str]
    relevant_entities: list[str]
    relevant_relations: list[tuple[str, str, str]]


@dataclass(slots=True)
class Corpus:
    documents: list[tuple[str, str]] = field(default_factory=list)
    queries: list[QueryJudgement] = field(default_factory=list)
    relations: set[tuple[str, str, str]] = field(default_factory=set)
    claims: list[tuple[str, str, str, str, bool]] = field(default_factory=list)

    @property
    def gazetteer(self) -> dict[str, list[str]]:
        return GAZETTEER


def build_corpus() -> Corpus:
    """Build the deterministic evaluation corpus."""
    return Corpus(
        documents=list(DOCUMENTS),
        queries=[
            QueryJudgement(
                query=q,
                relevant_documents=docs,
                relevant_entities=entities,
                relevant_relations=relations,
            )
            for q, docs, entities, relations in RETRIEVAL_QUERIES
        ],
        relations=set(GROUND_TRUTH_RELATIONS),
        claims=list(CLAIM_LABELS),
    )
