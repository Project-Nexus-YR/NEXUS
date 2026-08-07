"""Common domain primitives shared across the knowledge model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "Confidence",
    "VerificationState",
    "now_iso",
    "ConfidenceValue",
]


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True, slots=True)
class Confidence:
    """A probability-like confidence value on the unit interval.

    ``value`` is clamped to ``[0.0, 1.0]``.
    """

    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.value!r}")

    def __float__(self) -> float:
        return self.value

    def __str__(self) -> str:
        return f"{self.value:.3f}"


ConfidenceValue = Confidence


class VerificationState(str, Enum):
    """Verification state of a claim or relation.

    The lifecycle is intentionally not a simple linear progression:
    a claim can move from ``UNVERIFIED`` to ``VERIFIED`` or ``REFUTED``,
    and later to ``CONTRADICTED`` or ``STALE`` as new evidence arrives.
    """

    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    VERIFIED = "verified"
    REFUTED = "refuted"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"
    STALE = "stale"
