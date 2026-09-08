"""Release contract for the public citation package surface."""

from __future__ import annotations

import tomllib
from pathlib import Path

import nexus_knowledge

ROOT = Path(__file__).parents[1]


def _project_config() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_module_is_the_single_authority_for_the_070_package_version() -> None:
    config = _project_config()
    project = config["project"]
    setuptools = config["tool"]["setuptools"]

    assert "version" not in project
    assert project["dynamic"] == ["version"]
    assert setuptools["dynamic"]["version"] == {"attr": "nexus_knowledge.__version__"}
    assert nexus_knowledge.__version__ == "0.7.0"


def test_citation_extra_keeps_the_supported_parser_bounds() -> None:
    project = _project_config()["project"]

    assert project["optional-dependencies"]["citation"] == [
        "pypdf>=6.16.2,<7",
        "defusedxml>=0.7,<1",
    ]


def test_citation_contract_is_exported_from_the_package_root() -> None:
    from nexus_knowledge import (
        CitationCandidate,
        CitationIngestionPort,
        CitationIngestionService,
        CitationLexicalSearch,
        CitationSearchFilters,
        CitationSearchPort,
    )
    from nexus_knowledge.ingestion import CitationIngestionService as ServiceSource
    from nexus_knowledge.port import (
        CitationCandidate as CandidateSource,
    )
    from nexus_knowledge.port import (
        CitationIngestionPort as IngestionPortSource,
    )
    from nexus_knowledge.port import (
        CitationSearchFilters as SearchFiltersSource,
    )
    from nexus_knowledge.port import CitationSearchPort as SearchPortSource
    from nexus_knowledge.retrieval import CitationLexicalSearch as SearchSource

    assert CitationIngestionPort is IngestionPortSource
    assert CitationSearchPort is SearchPortSource
    assert CitationCandidate is CandidateSource
    assert CitationSearchFilters is SearchFiltersSource
    assert CitationIngestionService is ServiceSource
    assert CitationLexicalSearch is SearchSource

    expected = {
        "CitationIngestionPort",
        "CitationSearchPort",
        "CitationCandidate",
        "CitationSearchFilters",
        "CitationIngestionService",
        "CitationLexicalSearch",
    }
    assert expected <= set(nexus_knowledge.__all__)
