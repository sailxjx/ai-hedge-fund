# Agent Operations Guide

This playbook keeps the Codex agent aligned with the multi-agent trading stack. It captures the mission, required tooling, and the guardrails that keep experiments reproducible and affordable.

## Platform Orientation
- Core engine lives in `src/`: prompts in `src/agents/`, orchestration graphs in `src/graph/`, shared utilities in `src/tools/` and `src/utils/`, CLI entrypoint at `src/main.py`.
- Backtesting lives under `src/backtesting/` with helper CLI in `src/backtesting/cli.py` and scheduler/governance utilities alongside tests in `tests/backtesting/`.
- Web surface: FastAPI backend in `app/backend/`, Vite/React frontend in `app/frontend/`, Docker orchestration in `docker/`.

## Mission & Operating Posture
- Role: AI quant researcher iterating on analyst, risk, and PM agents to surface new alpha without rebuilding the framework.
- Style: Operate fully automatically—no human confirmation loops—while keeping logs, metrics, and prompts synchronized.
- Provider: Always run with the Azure model stack; missing `OPENAI_API_KEY` errors are routing bugs that must be fixed, not patched with legacy keys.

## Analyst Baseline Requirements
- Anthropomorphic agents only: every analyst must be an LLM-powered persona with a distinct investment style and decision rubric.
- No fixed-rule shortcuts: never encode manual policies or deterministic signal heuristics inside analyst prompts or code paths; the LLM must originate each trading signal.
- Signal provenance: retain prompt context and logging so downstream consumers can audit that recommendations came from the assigned persona rather than handcrafted logic.

## Development Workflow
1. Start every iteration by reading `TODO.md` to pick up outstanding context and update it as work progresses.
2. Review recent changes via `git status`/`git diff` so new edits build on the current ground truth.
3. Run a smoke test before substantial changes: `poetry run python src/main.py --model-provider azure --analysts-all --tickers TSLA` (set Codex `timeout_ms` to cap runtime).
4. Use `src/tools/agent_iteration.py` and associated tests to loop diagnostics → prompt/code edits → smoke test → pytest, with auto-revert on failures.
5. Record every hypothesis, experiment, and decision in `TODO.md` and the relevant logs under `log/analysis/`.

## Coding & Testing Standards
- Python: 4-space indentation, snake_case functions, PascalCase classes, type hints for new surfaces.
- Formatting: `poetry run black .` (line length 420) then `poetry run isort .`; finish with `poetry run flake8`.
- Testing: `poetry run pytest` for the suite or narrow with paths/`-k`. Co-locate tests under `tests/<domain>/test_<feature>.py` and reuse fixtures from `tests/fixtures/`.
- Frontend: Follow existing ESLint rules; React components in PascalCase, hooks/utilities in camelCase.
- Commits: Keep messages lowercase imperative (e.g., `feat:`/`fix:`); reference issues with `#123` and never commit secrets.

## Backtesting Protocol
- Baseline scheduling: Use `poetry run python -m src.backtesting.experiment_scheduler --config <config>` so runs log to `log/backtest.csv` and respect per-experiment timeouts.
- Sanity checks: For quick validation after a code or prompt change, limit the backtest window to ~3 trading days; the default scheduler timeout easily covers this.
- Full regimes: When running multi-week windows, set the scheduler/backtester timeout to a generous budget (~2 hours) so LLM latency doesn’t abort the run prematurely.
- Timeout handling: Never wrap commands with shell `timeout`. Rely on Codex `timeout_ms` or the scheduler’s `timeout_seconds` parameter to avoid orphaned processes. The scheduler now kills the entire backtester process group when the limit is hit, so expect a surfaced `ExperimentRunError` if a run exceeds the budget and bump the config value only when absolutely necessary. After any timeout, immediately confirm no lingering `src/backtester.py` PIDs remain (e.g. `ps -eo pid,ppid,pgid,cmd | grep src/backtester.py`) and only rerun once the tree is clear.
- Command template: `poetry run python src/backtester.py --model-provider azure --analysts-all --tickers <comma-separated> --start-date YYYY-MM-DD --end-date YYYY-MM-DD --log-file <path>` (only adjust analysts, tickers, dates, log path, and timeout).
- Ledger discipline: After each run, parse logs with `llm_backtest_digest`, append metrics/deltas to `log/backtest.csv`, and review governance guardrails via `src/backtesting/governance_monitor.py`.

## Experiment Iteration
- Keep evaluation windows recent (post-2020) to match LLM priors; sanity-check dates with `date +"%Y-%m-%d"`.
- Mine prior logs for failure signatures (missed reversals, range breaks) and design new analyst prompts that address the gaps.
- Run paired backtests for baseline vs. new prompts and summarize deltas in dedicated CSVs under `log/`.
- Treat each idea as unproven until backtests confirm; document adjustments, wins, and regressions in the decision log.

## Operational Guardrails
- Use Azure credentials from `.env`; update `.gitignore` for new artifacts and document env vars in `README.md` or PRs.
- Avoid manual framework rewrites; invest effort in agent prompts, evaluator logic, and data-driven calibrations.
- When risk guardrails conflict with consensus signals, adjust prompt parameters through controlled experiments and add regression tests before promoting changes.
- Maintain an experiment backlog so the iteration loop never stalls; queue future analyst ideas, time windows, or calibration studies for Codex to execute.
