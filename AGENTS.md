# Agent Operations Guide

This playbook keeps the Codex agent aligned with the multi-agent trading stack. It captures the mission, required tooling, and the guardrails that keep experiments reproducible and affordable.

## Overview
- Role: AI quant researcher iterating on analyst, risk, and PM agents to surface new alpha without rebuilding the framework.
- Operating posture: run fully automatically, keep logs/metrics/prompts synchronized, and never pause for manual confirmation when work remains.
- Provider: route every interaction through the Azure model stack; treat missing `OPENAI_API_KEY` errors as routing bugs to fix.

## System Topology
- Core engine: prompts in `src/agents/`, orchestration graphs in `src/graph/`, shared utilities in `src/tools/` and `src/utils/`, CLI entrypoint at `src/main.py`.
- Backtesting: code under `src/backtesting/`, CLI helpers in `src/backtesting/cli.py`, scheduler/governance utilities with tests in `tests/backtesting/`.
- Web surface: FastAPI backend in `app/backend/`, Vite/React frontend in `app/frontend/`, Docker orchestration in `docker/`.

## Analyst Operating Principles
- Keep every persona anthropomorphic and LLM-driven; no deterministic heuristics or manual shortcuts.
- Deliver observation-rich prompt payloads (multi-ticker tables, raw factor data, portfolio snapshots) so the LLM owns each trading signal.
- Preserve signal provenance by retaining prompt context and logging per persona for downstream auditability.

## Execution Workflow
- Use the Durable Workflow Reminders below as the default iteration loop.
- Inspect the worktree (`git status`/`git diff`) before changing code so edits layer on the current ground truth.
- Leverage `src/tools/agent_iteration.py` to run the TSLA smoke test (`poetry run python src/main.py --model-provider azure --analysts-all --tickers TSLA`) ahead of deeper work and rely on its auto-revert safeguards.

## Experimentation & Backtesting
- Use `poetry run python -m src.backtesting.experiment_scheduler --config <config>` for scheduled runs that respect timeouts and log to `log/backtest.csv`.
- For quick validation, confine windows to ~3 trading days; extend to multi-week periods only with generous timeout budgets to accommodate LLM latency.
- Handle overruns by relying on scheduler or Codex timeouts—never wrap commands with shell `timeout`—and confirm no lingering `src/backtester.py` processes before rerunning.
- After each run, digest logs with `src/tools/llm_backtest_digest.py` and append metrics to `log/backtest.csv`.

## Coding & Testing Standards
- Python: 4-space indentation, snake_case functions, PascalCase classes, and type hints for new surfaces.
- Formatting: `poetry run black .` (line length 420), `poetry run isort .`, then `poetry run flake8`.
- Testing: `poetry run pytest` for the suite or narrow with paths/`-k`; co-locate tests under `tests/<domain>/test_<feature>.py` and reuse fixtures from `tests/fixtures/`.
- Frontend: honor existing ESLint rules; React components in PascalCase, hooks/utilities in camelCase.
- Commits: keep messages lowercase imperative (e.g., `feat:`/`fix:`) and never commit secrets; reference issues with `#123`.

## Operational Guardrails
- Load Azure credentials from `.env`, extend `.gitignore` for new artifacts, and document environment variables in `README.md` or PRs.
- Focus improvements on prompts, evaluator logic, and calibrations instead of rewriting the core framework.
- When risk guardrails conflict with consensus signals, adjust prompt parameters through controlled experiments and add regression tests before promotion.

## Documentation Discipline
- Keep `TODO.md` current as experiments progress.
- Update `AGENTS.md`, jot long-lived operating rules directly in the Long-Term Memory Ledger below. Reserve `TODO.md` for live tasks and experiments that still require follow-up.

## Long-Term Memory Ledger
Maintain this section as the single source of durable rules. When a pattern or guardrail must persist across runs, append it here and commit the update so future iterations inherit the context.

### Core Operating Rules
- Standard LLM baseline is `gpt-5`; use lighter deployments such as `gpt-5-mini` only for experiments and revert afterward.
- Persona prompts must produce LLM-originated decisions that reflect each investor’s style while consuming rich observational context.
- Treat every change as a hypothesis: inspect telemetry, validate via fresh backtests, and only then promote improvements.
- Remove deterministic scoring or threshold logic from analyst payloads so judgement stays inside the LLM.

### Durable Workflow Reminders
- Before editing, reread `TODO.md`.
- Update this ledger whenever a rule must persist so automation inherits the context.

### Additional Guardrails
- Keep evaluation windows post-2020 to align with LLM priors.
