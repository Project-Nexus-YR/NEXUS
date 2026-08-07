"""Knowledge engine service facade and composition root."""

from .engine import (
    KnowledgeEngine,
    KnowledgeUpdate,
    KnowledgeUpdateReceipt,
    ProvenanceResponse,
)
from .factory import Adapters, create_engine

__all__ = [
    "Adapters",
    "KnowledgeEngine",
    "KnowledgeUpdate",
    "KnowledgeUpdateReceipt",
    "ProvenanceResponse",
    "create_engine",
]
