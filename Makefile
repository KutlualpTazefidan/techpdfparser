# Regenerate lock files from pyproject.toml.
# Run after editing dependency declarations; commit the resulting *.lock files.

.PHONY: lock lock-cpu lock-gpu test lint type check

lock: lock-cpu

lock-cpu:
	uv pip compile pyproject.toml --extra dev -o requirements/cpu.lock

lock-gpu:
	uv pip compile pyproject.toml --extra dev --extra gpu -o requirements/gpu.lock

test:
	pytest -q

lint:
	ruff check extraction

type:
	mypy

check: lint type test
