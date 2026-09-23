.PHONY: setup lint test

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy --strict packages/py/trw/src
	uv run python scripts/check_licenses.py

test:
	uv run pytest -q
