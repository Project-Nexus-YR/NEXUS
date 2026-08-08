"""Autonomous-investigation domain, planning, evidence, and application services."""

from .application import InvestigationApplication, IterationOutcome, PlanningOutcome
from .evidence import Evidence, EvidenceSet, InvestigationResult
from .execution import PlanExecution, PlanExecutionController
from .generator import CandidateInvestigation, InvestigationGenerator
from .objective import ResearchObjective
from .planner import InvestigationPlan, InvestigationPlanner
from .repository import (
    InMemoryInvestigationRepository,
    InvestigationRecord,
    SQLiteInvestigationRepository,
)
from .results import (
    InMemoryInvestigationResultRepository,
    RuntimeResultCollector,
    SQLiteInvestigationResultRepository,
)
from .session import InvestigationBudget, InvestigationSession, SessionState
from .verification import EpistemicStatus, VerificationPolicy

__all__ = [
    "CandidateInvestigation",
    "EpistemicStatus",
    "Evidence",
    "EvidenceSet",
    "InMemoryInvestigationRepository",
    "InvestigationApplication",
    "InvestigationBudget",
    "InvestigationGenerator",
    "InvestigationPlan",
    "InvestigationPlanner",
    "InvestigationRecord",
    "InvestigationResult",
    "InvestigationSession",
    "IterationOutcome",
    "InMemoryInvestigationResultRepository",
    "PlanExecution",
    "PlanExecutionController",
    "PlanningOutcome",
    "ResearchObjective",
    "RuntimeResultCollector",
    "SQLiteInvestigationRepository",
    "SQLiteInvestigationResultRepository",
    "SessionState",
    "VerificationPolicy",
]
