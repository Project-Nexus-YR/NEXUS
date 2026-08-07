"""Stable ID generation for domain objects.

IDs are short prefixed opaque identifiers. Two strategies are provided:

* :func:`new_id` — random (uuid4 based), for records created at runtime.
* :func:`stable_id` — deterministic hash-based, for IDs that must be
  reproducible from content (e.g. deduplicated entities, chunk ids).

All domain objects carry a string ID produced by these helpers.
"""

from __future__ import annotations

import hashlib
import uuid

__all__ = ["new_id", "stable_id"]


def new_id(prefix: str) -> str:
    """Return a random, collision-resistant ID with the given prefix.

    Args:
        prefix: short lowercase namespace, e.g. ``"doc"``, ``"ent"``.

    Returns:
        An opaque string like ``"doc_8f3a..."``.
    """
    return f"{prefix}_{uuid.uuid4().hex}"


def stable_id(prefix: str, *parts: object) -> str:
    """Return a deterministic ID derived from the given content parts.

    The same inputs always produce the same ID, which makes IDs
    stable across runs and across replicated nodes.

    Args:
        prefix: short lowercase namespace.
        *parts: content used to derive the hash.

    Returns:
        An opaque string like ``"ent_9f2c..."``.
    """
    joined = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
