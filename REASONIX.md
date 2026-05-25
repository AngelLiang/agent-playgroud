# agent-playgroud — REASONIX.md

## Stack
- **Language:** Python ≥3.11 (.python-version)
- **Runtime:** managed via `uv` (uv.lock present)
- **Deps (from uv.lock):** `openai>=2.38.0`, `python-dotenv>=1.2.2`
  (pyproject.toml `dependencies = []` is stale — the lock file is authoritative)

## Layout
- `main.py` — entry point (prints "Hello from agent-playgroud!")
- `react.py` — empty file, placeholder / to be implemented
- `pyproject.toml` — project metadata
- `uv.lock` — auto-generated dependency lockfile (do not edit by hand)
- `.python-version` — pinned Python 3.11

## Commands
- No scripts defined in pyproject.toml. Use `uv run <script>.py` to execute.

## Watch out for
- `react.py` is empty. It exists as a filename but has no code.
- `pyproject.toml` lists `dependencies = []` but `uv.lock` shows a dependency on
  `openai` and `python-dotenv`. If adding new deps, run `uv add <pkg>` to keep
  both in sync.
- No test runner, formatter, or linter configured.
