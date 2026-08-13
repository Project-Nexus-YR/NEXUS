"""Composition root for the local GUI server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nexus_knowledge.persistence.json_codec import load_snapshot
from nexus_knowledge.service.explorer import KnowledgeExplorer
from nexus_knowledge.service.factory import Adapters, create_engine
from nexus_runtime.distributed.coordinator import Coordinator
from nexus_runtime.distributed.service import RuntimeApplication
from nexus_runtime.distributed.store import SQLiteTaskStore
from nexus_runtime.investigation.application import InvestigationApplication
from nexus_runtime.investigation.repository import SQLiteInvestigationRepository
from nexus_runtime.monitoring import RuntimeMonitor

from .service import NexusGuiService


@dataclass(slots=True)
class GuiResources:
    service: NexusGuiService
    investigations: SQLiteInvestigationRepository
    task_store: SQLiteTaskStore

    def close(self) -> None:
        self.investigations.close()
        self.task_store.close()


def create_gui_resources(
    *,
    snapshot: Path | None = None,
    investigations_db: Path = Path(".nexus/investigations.sqlite"),
    runtime_db: Path = Path(".nexus/runtime.sqlite"),
) -> GuiResources:
    repository = load_snapshot(snapshot) if snapshot is not None else None
    engine = create_engine(Adapters() if repository is None else Adapters(repository=repository))
    investigations = SQLiteInvestigationRepository(investigations_db)
    task_store = SQLiteTaskStore(runtime_db)
    runtime = RuntimeApplication(Coordinator(task_store))
    application = InvestigationApplication(engine, repository=investigations)
    service = NexusGuiService(
        KnowledgeExplorer(engine),
        RuntimeMonitor(investigations, runtime),
        application,
    )
    return GuiResources(service, investigations, task_store)
