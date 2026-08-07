"""CLI entry point tests."""

import json

import pytest

from nexus_knowledge.cli import main


def _ingest(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("Ada Lovelace works at Acme Corp in London.", encoding="utf-8")
    return str(path)


class TestIngest:
    def test_ingest_text(self, tmp_path):
        exit_code = main(["ingest", _ingest(tmp_path), "--title", "probe"])
        assert exit_code == 0

    def test_ingest_markdown(self, tmp_path):
        path = tmp_path / "doc.md"
        path.write_text("# Section\nAcme Corp develops software.", encoding="utf-8")
        assert main(["ingest", str(path), "--kind", "markdown"]) == 0

    def test_ingest_with_gazetteer(self, tmp_path):
        gazetteer = tmp_path / "gazetteer.json"
        gazetteer.write_text(json.dumps({"Company": ["Acme Corp"]}), encoding="utf-8")
        assert main(["ingest", _ingest(tmp_path), "--gazetteer", str(gazetteer)]) == 0

    def test_healthcheck_prints_json(self, tmp_path, capsys):
        main(["ingest", _ingest(tmp_path)])
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"sources", "documents", "chunks", "entities", "relations", "claims", "evidence"}
        assert payload["sources"] == 1


class TestRetrieve:
    def test_retrieve_requires_ingested_data(self, capsys):
        exit_code = main(["retrieve", "Acme Corp"])
        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["query"] == "Acme Corp"


class TestGraphRAG:
    def test_graphrag_returns_evidence(self, capsys):
        main(["graphrag", "Ada Lovelace"])
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, dict)


class TestGaps:
    def test_gaps_returns_list(self, capsys):
        main(["gaps"])
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, list)
        if payload:
            assert {"kind", "description", "priority", "estimated_cost"} <= set(payload[0])


class TestScore:
    def test_score_returns_list(self, capsys):
        main(["score", "--top-k", "5"])
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, list)


class TestStats:
    def test_stats_returns_graph_stats(self, capsys):
        main(["stats"])
        payload = json.loads(capsys.readouterr().out)
        assert "num_entities" in payload


class TestBench:
    def test_bench_prints_report(self, capsys):
        main(["bench"])
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"graph", "knowledge", "retrieval"}

    def test_bench_writes_output_file(self, tmp_path):
        output = tmp_path / "report.json"
        assert main(["bench", "--output", str(output)]) == 0
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert "retrieval" in payload


class TestInvalidCommand:
    def test_unknown_command_errors(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["frobnicate"])
        assert excinfo.value.code != 0
