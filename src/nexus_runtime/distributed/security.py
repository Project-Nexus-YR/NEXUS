"""Identity and authorization ports for the distributed control plane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..models import DomainError


@dataclass(frozen=True, slots=True)
class WorkerIdentity:
    worker_id: str
    capabilities: frozenset[str]
    subject: str

    def __post_init__(self) -> None:
        if not self.worker_id.strip() or not self.subject.strip():
            raise DomainError("worker identity and subject are required")


class WorkerAuthenticator(Protocol):
    def verify(self, identity: WorkerIdentity) -> None: ...


class RuntimeAuthorizer(Protocol):
    def authorize(self, principal: str, action: str, resource_id: str) -> None: ...


class TrustedLocalWorkerAuthenticator:
    """Explicit local-simulator trust boundary; production must replace this port."""

    def verify(self, identity: WorkerIdentity) -> None:
        if not identity.subject.startswith("local:"):
            raise DomainError("local runtime requires a local worker subject")


class ConfiguredWorkerAuthenticator:
    """Verifies capabilities against deployment-owned worker configuration."""

    def __init__(self, grants: dict[str, frozenset[str]]) -> None:
        self._grants = grants

    def verify(self, identity: WorkerIdentity) -> None:
        if self._grants.get(identity.subject) != identity.capabilities:
            raise DomainError("worker identity or capabilities are not authorized")


class AllowAllRuntimeAuthorizer:
    """Local-only authorizer used by the simulator and development CLI."""

    def authorize(self, principal: str, action: str, resource_id: str) -> None:
        if not principal.strip():
            raise DomainError("an authenticated principal is required")


class StaticRuntimeAuthorizer:
    def __init__(self, grants: dict[str, frozenset[str]]) -> None:
        self._grants = grants

    def authorize(self, principal: str, action: str, resource_id: str) -> None:
        if action not in self._grants.get(principal, frozenset()):
            raise DomainError(f"principal {principal!r} cannot perform {action}")
