"""External origins of information."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import now_iso
from .ids import new_id

__all__ = ["Source", "SourceKind"]


class SourceKind:
    """Well-known source kinds."""

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    REPOSITORY = "repository"
    DATABASE = "database"
    WEB = "web"
    EXPERIMENT = "experiment"
    OTHER = "other"


@dataclass(slots=True)
class Source:
    """A traceable origin of information.

    Every document, observation and (transitively) every claim is rooted
    in a source. Sources are the top of the provenance chain:

    claim -> evidence -> chunk -> document -> source
    """

    title: str
    kind: str
    reference: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("src"))
    ingested_at: str = field(default_factory=now_iso)
