"""Capability-based authorization for every privileged tool call."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PolicyDecision(StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    principal_id: str
    capability: str
    side_effect: str
    resource: str


class PolicyEngine:
    """A deliberately small policy point; replace rules without changing the runtime."""

    def __init__(
        self,
        grants: dict[str, frozenset[str]],
        approval_capabilities: frozenset[str] = frozenset(),
    ) -> None:
        self._grants = grants
        self._approval_capabilities = approval_capabilities

    def decide(self, request: PolicyRequest) -> PolicyDecision:
        if request.capability not in self._grants.get(request.principal_id, frozenset()):
            return PolicyDecision.DENY
        if (
            request.capability in self._approval_capabilities
            or request.side_effect == "destructive"
        ):
            return PolicyDecision.REQUIRE_APPROVAL
        return PolicyDecision.ALLOW
