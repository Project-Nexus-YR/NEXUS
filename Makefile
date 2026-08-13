.PHONY: format lint typecheck test verify benchmark gui-install gui-test gui-build gui

format:
	python3 -m ruff format src/nexus_runtime src/nexus_evidence tests/test_agent.py tests/test_agent_harness.py tests/test_tools.py tests/test_distributed_runtime.py tests/test_investigation_intelligence.py tests/test_evidence_loop.py tests/test_investigation_application.py tests/test_investigation_cli.py tests/test_evidence_engine.py

lint:
	python3 -m ruff check src/nexus_runtime src/nexus_evidence tests/test_agent.py tests/test_agent_harness.py tests/test_tools.py tests/test_distributed_runtime.py tests/test_investigation_intelligence.py tests/test_evidence_loop.py tests/test_investigation_application.py tests/test_investigation_cli.py tests/test_evidence_engine.py

typecheck:
	python3 -m mypy src/nexus_runtime src/nexus_evidence

test:
	python3 -m pytest -ra

verify:
	python3 -m compileall -q src tests
	python3 -m pytest -q

benchmark:
	python3 -m nexus_runtime.distributed.benchmark
	python3 -m nexus_runtime.investigation.benchmark
	python3 -m nexus_evidence.benchmark

gui-install:
	python3 -m pip install -e '.[gui]'
	npm --prefix frontend install

gui-test:
	npm --prefix frontend run lint
	npm --prefix frontend run typecheck
	npm --prefix frontend test

gui-build:
	npm --prefix frontend run build

gui: gui-build
	nexus-gui
