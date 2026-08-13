"""Search-plan generation and deterministic claim prioritization."""

from __future__ import annotations

from dataclasses import dataclass

from .model import AtomicClaim, ClaimType, stable_id


@dataclass(frozen=True, slots=True)
class SearchQuery:
    claim_id: str
    text: str
    strategy: str
    priority: float
    query_id: str = ""

    def __post_init__(self) -> None:
        if not self.query_id:
            object.__setattr__(
                self,
                "query_id",
                stable_id("evidence_query", self.claim_id, self.strategy, self.text),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "claim_id": self.claim_id,
            "text": self.text,
            "strategy": self.strategy,
            "priority": self.priority,
        }


class SearchPlanner:
    """Generate complementary exact, entity, source, and contradiction searches."""

    def plan(self, claim: AtomicClaim) -> tuple[SearchQuery, ...]:
        if not claim.is_verifiable:
            return ()
        exact = f'"{claim.text}"'
        core = " ".join(part for part in claim.spo if part).strip() or claim.text
        queries = [
            SearchQuery(claim.claim_id, exact, "exact", claim.importance),
            SearchQuery(claim.claim_id, core, "semantic", claim.importance * 0.95),
        ]
        if claim.claim_type == ClaimType.NUMERICAL:
            queries.append(
                SearchQuery(claim.claim_id, f"{core} data methodology", "primary_numeric", 1.0)
            )
        elif claim.claim_type == ClaimType.QUOTATION:
            queries.append(SearchQuery(claim.claim_id, f"{exact} transcript", "primary_quote", 1.0))
        else:
            queries.append(
                SearchQuery(claim.claim_id, f"{core} primary source", "primary_source", 0.9)
            )
        queries.append(
            SearchQuery(claim.claim_id, f"{core} false disputed correction", "contradiction", 0.8)
        )
        return tuple(queries)

    def prioritize(self, claims: tuple[AtomicClaim, ...]) -> tuple[AtomicClaim, ...]:
        return tuple(sorted(claims, key=lambda item: (-item.importance, item.claim_id)))
