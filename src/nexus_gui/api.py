"""FastAPI transport for the NEXUS visual exploration service."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nexus_knowledge.service.explorer import NODE_KINDS, ExplorerFilters

from .service import NexusGuiService


class InvestigationRequest(BaseModel):
    question: str | None = Field(default=None, min_length=3, max_length=2_000)
    success_criteria: list[str] = Field(default_factory=list, max_length=20)


def create_app(
    service: NexusGuiService,
    *,
    frontend_dir: Path | None = None,
    allowed_origins: Sequence[str] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ),
) -> FastAPI:
    app = FastAPI(
        title="NEXUS Visual API",
        version="0.5.0",
        description="Bounded knowledge graph, provenance, and runtime exploration.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.get("/api/graph")
    def graph(
        kinds: Annotated[list[str] | None, Query()] = None,
        relation_types: Annotated[list[str] | None, Query()] = None,
        verification_states: Annotated[list[str] | None, Query()] = None,
        min_confidence: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
        max_nodes: Annotated[int, Query(ge=1, le=2_000)] = 750,
        max_edges: Annotated[int, Query(ge=1, le=10_000)] = 2_500,
    ) -> dict[str, Any]:
        return service.graph(
            _filters(
                kinds,
                relation_types,
                verification_states,
                min_confidence,
                max_nodes,
                max_edges,
            )
        )

    @app.get("/api/graph/neighborhood/{node_id}")
    def neighborhood(
        node_id: str,
        depth: Annotated[int, Query(ge=1, le=3)] = 1,
        kinds: Annotated[list[str] | None, Query()] = None,
        relation_types: Annotated[list[str] | None, Query()] = None,
        verification_states: Annotated[list[str] | None, Query()] = None,
        min_confidence: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
        max_nodes: Annotated[int, Query(ge=1, le=2_000)] = 500,
        max_edges: Annotated[int, Query(ge=1, le=10_000)] = 2_000,
    ) -> dict[str, Any]:
        try:
            return service.neighborhood(
                node_id,
                depth=depth,
                filters=_filters(
                    kinds,
                    relation_types,
                    verification_states,
                    min_confidence,
                    max_nodes,
                    max_edges,
                ),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown graph node: {node_id}") from exc

    @app.get("/api/nodes/{node_id}")
    def node(node_id: str) -> dict[str, Any]:
        try:
            return service.node(node_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown graph node: {node_id}") from exc

    @app.get("/api/search")
    def search(
        q: Annotated[str, Query(min_length=1, max_length=500)],
        limit: Annotated[int, Query(ge=1, le=100)] = 25,
    ) -> list[dict[str, Any]]:
        return service.search(q, limit=limit)

    @app.get("/api/gaps")
    def gaps() -> list[dict[str, Any]]:
        return service.gaps()

    @app.post("/api/gaps/{gap_id}/investigations", status_code=201)
    def create_investigation(gap_id: str, request: InvestigationRequest) -> dict[str, Any]:
        try:
            return service.create_gap_investigation(
                gap_id,
                question=request.question,
                success_criteria=tuple(request.success_criteria),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown knowledge gap: {gap_id}") from exc

    @app.get("/api/contradictions")
    def contradictions() -> list[dict[str, Any]]:
        return service.contradictions()

    @app.get("/api/sources")
    def sources() -> list[dict[str, Any]]:
        return service.sources()

    @app.get("/api/investigations")
    def investigations() -> list[dict[str, Any]]:
        return service.investigations()

    @app.get("/api/investigations/{session_id}")
    def investigation(session_id: str) -> dict[str, Any]:
        try:
            return service.investigation(session_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"unknown investigation session: {session_id}"
            ) from exc

    @app.get("/api/runtime")
    def runtime() -> dict[str, Any]:
        return service.runtime()

    if frontend_dir is not None and (frontend_dir / "index.html").is_file():
        assets = frontend_dir / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str) -> FileResponse:
            requested = (frontend_dir / path).resolve()
            root = frontend_dir.resolve()
            if requested.is_relative_to(root) and requested.is_file():
                return FileResponse(requested)
            return FileResponse(frontend_dir / "index.html")

    return app


def _filters(
    kinds: list[str] | None,
    relation_types: list[str] | None,
    verification_states: list[str] | None,
    min_confidence: float,
    max_nodes: int,
    max_edges: int,
) -> ExplorerFilters:
    try:
        return ExplorerFilters(
            node_kinds=NODE_KINDS if kinds is None else frozenset(kinds),
            relation_types=frozenset(relation_types or ()),
            verification_states=frozenset(verification_states or ()),
            min_confidence=min_confidence,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
