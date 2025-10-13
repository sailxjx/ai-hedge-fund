# Async Persona Parallelism Guide

This guide captures the async persona execution model, telemetry plumbing, and validation
loops that ship with the async parallelism refactor. Use it as the source of truth when
enabling `ASYNC_PERSONAS`, reviewing timing telemetry, or extending the async suite.

## Enable Async Personas

- Async personas are enabled by default across the CLI, FastAPI backend, and standalone
  backtester. No environment flag is required for the async graph.
- Export `ASYNC_PERSONAS=0` (or `false`, `off`) to force the legacy synchronous graph.
  Re-enable async by removing the override or setting the flag back to `1`.
- Optional knobs:
  - `LLM_ASYNC_MAX_CONCURRENCY` (default `8`) caps the number of concurrent persona LLM
    calls guarded by the semaphore in `src/utils/llm.py`.
  - `LLM_CALL_TIMEOUT_SECONDS` and `LLM_MAX_RETRIES` apply to both sync and async paths.
  - `LLM_ASYNC_MAX_CONCURRENCY` pairs with the async meta that the backtester stores so
    you can compute utilization.

## CLI and Backtester Usage

```bash
# Async CLI smoke (Azure provider shown, async default on)
poetry run python src/main.py --model-provider azure --analysts-all --tickers TSLA

# Async backtester window
poetry run python src/backtester.py --tickers TSLA,NVDA \
  --start-date 2024-10-01 --end-date 2024-10-03 --model-provider azure

# Legacy synchronous fallback (optional)
ASYNC_PERSONAS=0 poetry run python src/main.py --model-provider azure --analysts-all --tickers TSLA
```

The CLI automatically routes through `run_hedge_fund_async` when async personas are
enabled and captures per-agent timings in the payload returned to the caller.

## FastAPI Backend

- The backend compiles the async LangGraph automatically. Export `ASYNC_PERSONAS=0` only
  if you need to run the synchronous graph for bisects, then restore the flag (unset or
  `1`) to re-enable async execution.
- Progress streaming payloads now include UTC timestamps plus persona analyses thanks to
  the upgraded handler signature in `src/utils/progress.py`.
- The backend switches between `graph.invoke` and `graph.ainvoke` transparently and uses
  the same runtime helper that powers the CLI/backtester.

## Telemetry and Timing Summaries

- Async backtests persist timing JSON under `log/backtest_timings/` with aggregate stage
  timings (`agent_invoke`, `prefetch_data`, etc.), per-agent totals, and `async_meta`
  derived metrics (concurrency ratio, semaphore utilization, slowest persona, etc.).
- Pass `--include-async-telemetry` (or configure `include_async_telemetry` in scheduler
  configs) when running `poetry run python -m src.backtesting.experiment_scheduler` to
  append these metadata columns into `log/backtest.csv`.
- Use `src/tools/llm_backtest_digest.py --logs <log path>` and
  `src/tools/llm_combo_diagnostics.py --logs <log path>` to summarize async backtest
  behaviour; both utilities understand the enriched logs that the async stack emits.

## Regression and Diagnostics

- Async regression suite:
  ```bash
  poetry run pytest tests/agents/test_async_persona.py \
    tests/agents/test_async_wrappers.py \
    tests/backtesting/test_async_engine.py \
    tests/backend/test_graph_service.py -q
  ```
- Scheduler / governance additions covering async telemetry:
  ```bash
  poetry run pytest tests/backtesting/test_experiment_scheduler.py \
    tests/backtesting/test_governance_monitor.py -q
  ```
- Logs for the canonical async benchmark windows live under `log/backtest_timings/` and
  `log/analysis/backtest_*_async_*`.

## Troubleshooting

- Confirm that any explicit `ASYNC_PERSONAS` override is exported in the same shell that
  launches the process (default async requires no override).
- If async timing summaries are missing, ensure the backtester metadata `run_label`
  matches the label used in the scheduler lookup helpers.
- When semaphore utilization is persistently `1.0`, consider bumping
  `LLM_ASYNC_MAX_CONCURRENCY` (and rerun the async pytest quartet) before scaling
  hardware.
