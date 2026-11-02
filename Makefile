.PHONY: setup test lint fmt core-check core-manifest smoke

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src tests stages scripts

fmt:
	uv run ruff format src tests stages scripts

core-check:
	uv run python scripts/core_manifest.py

# NUR nach menschlicher Freigabe (siehe AGENTS.md, Regel 1)
core-manifest:
	uv run python scripts/core_manifest.py --write

smoke:
	uv run python stages/s0a_grokking/run.py --arm none --seeds 0 --steps 200 --eval-every 100 --name smoke
