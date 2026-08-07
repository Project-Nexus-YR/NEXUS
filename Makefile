.PHONY: format lint typecheck test verify

format:
	python3 -m ruff format src tests

lint:
	python3 -m ruff check src tests

typecheck:
	python3 -m mypy src

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

verify:
	python3 -m compileall -q src tests
	PYTHONPATH=src python3 -m unittest discover -s tests -v
