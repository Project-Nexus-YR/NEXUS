from __future__ import annotations

import json

from nexus_runtime.investigation.application import InvestigationApplication
from nexus_runtime.investigation.benchmark import run_benchmark
from nexus_runtime.investigation.cli import main
from nexus_runtime.investigation.repository import InMemoryInvestigationRepository

from .test_investigation_application import FakeKnowledge


def test_research_cli_create_status_pause_resume_and_cancel(capsys) -> None:
    app = InvestigationApplication(FakeKnowledge(), repository=InMemoryInvestigationRepository())
    assert main(["create", "What is Acme?", "--criterion", "verified"], application=app) == 0
    created = json.loads(capsys.readouterr().out)
    session_id = created["session_id"]

    assert main(["status", session_id], application=app) == 0
    assert json.loads(capsys.readouterr().out)["session"]["state"] == "PLANNING"
    assert main(["pause", session_id], application=app) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "PAUSED"
    assert main(["resume", session_id], application=app) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "PLANNING"
    assert main(["cancel", session_id], application=app) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "CANCELLED"


def test_research_cli_explain_gaps_plan_and_iterations(capsys) -> None:
    app = InvestigationApplication(FakeKnowledge())
    assert main(["create", "What is Acme?", "--criterion", "verified"], application=app) == 0
    session_id = json.loads(capsys.readouterr().out)["session_id"]
    app.plan_iteration(session_id, worker_capacity=1)

    assert main(["gaps", session_id], application=app) == 0
    assert json.loads(capsys.readouterr().out)[0]["id"] == "gap-0"
    assert main(["plan", session_id], application=app) == 0
    assert json.loads(capsys.readouterr().out)["session_id"] == session_id
    assert main(["iterations", session_id], application=app) == 0
    assert json.loads(capsys.readouterr().out)[0]["iteration"] == 0
    assert main(["explain", session_id], application=app) == 0
    explanation = json.loads(capsys.readouterr().out)
    assert explanation["why_investigated"]


def test_investigation_benchmark_has_required_workload() -> None:
    report = run_benchmark()
    assert report["methodology"] == {
        "gaps": 10,
        "candidates": 50,
        "selected": 10,
        "distributed_tasks": 10,
        "clock": "fixed UTC inputs with local deterministic runtime",
    }
    assert report["results"]["succeeded_tasks"] == 10
    assert report["results"]["accepted_evidence"] == 10
