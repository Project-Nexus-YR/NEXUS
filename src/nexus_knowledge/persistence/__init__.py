"""Persistence adapters for the knowledge engine."""

from .json_codec import dumps, load_snapshot, loads, save_snapshot, to_plain
from .memory import (
    InMemoryClaimRepository,
    InMemoryDocumentRepository,
    InMemoryEntityRepository,
    InMemoryKnowledgeRepository,
    InMemoryRepository,
)

__all__ = [
    "InMemoryClaimRepository",
    "InMemoryDocumentRepository",
    "InMemoryEntityRepository",
    "InMemoryKnowledgeRepository",
    "InMemoryRepository",
    "dumps",
    "load_snapshot",
    "loads",
    "save_snapshot",
    "to_plain",
]
