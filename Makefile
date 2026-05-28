.PHONY: install lock preflight preflight-basic preflight-models preflight-rate preflight-report test lint fmt typecheck clean

install:
	uv sync

lock:
	uv lock

preflight:
	uv run python scripts/api_key_probe.py --suite all

preflight-basic:
	uv run python scripts/api_key_probe.py --suite basic

preflight-models:
	uv run python scripts/api_key_probe.py --suite models

preflight-rate:
	uv run python scripts/api_key_probe.py --suite rate-limit

preflight-report:
	uv run python scripts/api_key_probe.py --suite report

test:
	uv run pytest -q

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

typecheck:
	uv run mypy scripts tests

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build reports/*.json reports/*.md

