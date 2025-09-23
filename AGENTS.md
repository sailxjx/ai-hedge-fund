# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the Python trading engine: agent prompts in `src/agents/`, orchestration graphs in `src/graph/`, shared tooling in `src/tools/` and `src/utils/`, and the CLI entry point at `src/main.py`.
- Backtesting helpers live in `src/backtesting/` (driven by `src/backtesting/cli.py`) and automated checks live in `tests/` with fixtures in `tests/fixtures/`.
- The web surface is in `app/`: `app/backend/` exposes the FastAPI service and Alembic migrations, while `app/frontend/` houses the Vite + React client; Docker scripts sit in `docker/`.

## Build, Test, and Development Commands
- `poetry install` — resolve Python dependencies from the repo root.
- `poetry run python src/main.py --ticker AAPL,MSFT,NVDA` — execute the multi-agent CLI; add `--ollama` for local LLMs.
- `poetry run backtester run --ticker TSLA` — exercise the backtesting pipeline; inspect flags with `poetry run backtester --help`.
- `poetry run uvicorn app.backend.main:app --reload` (run from `app/backend/`) — start the REST API, and `npm install && npm run dev` (from `app/frontend/`) boots the UI.

## Coding Style & Naming Conventions
- Use 4-space indentation, type hints for new Python surfaces, and snake_case functions with PascalCase classes.
- Run `poetry run black .` (line length 420) and `poetry run isort .` before `poetry run flake8`; keep imports grouped alphabetically.
- Frontend code follows the bundled ESLint rules; name React components in PascalCase and hooks/utilities in camelCase.

## Testing Guidelines
- Run `poetry run pytest` for the full suite; target-specific checks with patterns like `poetry run pytest tests/backtesting -k covariance`.
- Add tests beside the feature under `tests/<domain>/` using `test_<feature>.py`, and lean on fixtures in `tests/fixtures/` to avoid repetitive setup.
- Cover edge cases that influence trading decisions (missing fundamentals, rate limits) and record manual verification in the PR when UI work lacks automated tests.

## Commit & Pull Request Guidelines
- Keep commits compact with lowercase, imperative summaries aligned with the existing `fix:`/`feat:` style; link issues via `#123` where applicable.
- PRs should explain the change, list verification commands (`poetry run pytest`, `npm run lint`, etc.), and attach screenshots or CLI snippets for user-facing updates.
- Never commit secrets or raw market data; document new env vars in `README.md` or the PR body.

## Environment & Configuration Tips
- Duplicate `.env.example` to `.env` and set the required keys (`OPENAI_API_KEY`, `FINANCIAL_DATASETS_API_KEY`) before running the engine or API.
- Ignore local artifacts (datasets, notebooks, IDE files) by updating `.gitignore` as workflows evolve.
