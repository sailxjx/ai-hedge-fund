# TODO

## Auto-Improve Policy
1. Recover context before acting: read the latest entries in this file, scan `log/backtest.csv`, and tail the most recent artifacts under `log/analysis/` plus `/tmp/codex_exec.log` to understand where the last iteration ended.
2. Inspect governance guardrails via `poetry run python src/backtesting/governance_monitor.py` (or review its JSON output) to identify any breached Sharpe/return/drawdown thresholds.
3. Mine recent backtest logs with `src/tools/llm_backtest_digest.py` and `src/tools/llm_combo_diagnostics.py` to pinpoint failure motifs (missed reversals, squeeze overrides, regime mismatches).
4. Select the highest-impact gap and design a hypothesis (prompt tweak, analyst addition, calibration, or risk adjustment) that addresses it without rewriting core infrastructure.
5. Iterate using the guarded loop in `src/tools/agent_iteration.py`: apply the change, run the smoke test `poetry run python src/main.py --model-provider azure --analysts-all --tickers TSLA` (bounded by Codex `timeout_ms`), execute focused pytest modules, and revert automatically if checks fail.
6. Validate hypotheses with paired backtests (baseline vs revised) via `poetry run python -m src.backtesting.experiment_scheduler --config <config>`; ensure metrics land in `log/backtest.csv`, summarize deltas, and append insights here.
7. Document outcomes, residual risks, and next questions in this file so the next iteration can resume autonomously.

## Operational Guardrails
- Backtest windows: keep evaluation periods recent (post-2020). For quick sanity checks after edits, cap ranges to ~3 trading days; for full multi-week assessments, allocate generous time (≈2 hours) via scheduler `timeout_seconds` or Codex `timeout_ms`.
- Timeout handling: never wrap commands with shell `timeout`. Trust the scheduler’s built-in timeout enforcement and focus on strategy improvements rather than plumbing.
- Provider routing: all runs must use the Azure model provider. Missing `OPENAI_API_KEY` prompts indicate a bug that needs fixing.
- Logging discipline: every backtest must write a digestible log (`llm_backtest_digest` compatible) and ledger entry. Do not move on until the metrics and notes are captured.
- Framework scope: do not rebuild the architecture. Invest cycles in prompts, calibrations, and diagnostics-driven refinements.

## Active Initiatives
- [x] Observation-only persona interface for every analyst signal.
  - [x] Define a shared observation schema helper in `src/agents/persona_utils.py::persona_from_observations`.
  - [x] Remove auto-signals from analyst prompts so personas synthesize the final recommendation.
- [x] Refactor discretionary/fundamental personas (`aswath_damodaran`, `ben_graham`, `bill_ackman`, `cathie_wood`, `charlie_munger`, `michael_burry`, `mohnish_pabrai`, `peter_lynch`, `phil_fisher`, `rakesh_jhunjhunwala`, `stanley_druckenmiller`, `valuation`, `warren_buffett`) to emit observation payloads only.
- [x] Refactor growth/momentum/mean-reversion/valuation/sentiment analysts (technical, growth momentum, stat mean reversion, fundamentals, sentiment, valuation, trend regime, stop-loss guardian) to stream normalized metrics without deterministic signals.
- [x] Rework sentinel + allocator analysts (`momentum_guardian`, `breakout_cover_sentinel`, `range_recovery_sentinel`, `short_squeeze_guardian`, `macro_volatility_sentinel`, `downside_flow_sentinel`, `crash_short_allocator`, `regime_meta`) so personas own guardrails and constraints derived from observation payloads.
- [x] Update portfolio/risk integrations: portfolio manager and risk override amplifier now ingest persona-owned directives without reinterpretation; risk manager documentation remains accurate.
- [x] Refresh regression coverage and docs: adjusted `tests/agents/test_risk_override_amplifier.py`, recorded persona contract in `README.md`, and added observation metadata across agent payloads.
- [x] Run smoke + paired validation:
  - Smoke run: `poetry run python src/main.py --model-provider azure --analysts-all --tickers TSLA` (2025-10-07) completed successfully and logs stored under `log/`.
  - Backtester sanity: triggered TSLA 2024-09-16→2024-09-18 and 2024-09-16→2024-09-17 runs (logs `log/backtest_tsla_obs_refactor*.log`); orchestration prints portfolio summary although the CLI exits after 240s—metrics captured for inspection.

## Notes & References
- Risk/audit utilities: `src/backtesting/risk_override_audit.py`, `src/backtesting/analyze_risk_conflicts.py`.
- Diagnostics helpers: `src/tools/llm_backtest_digest.py`, `src/tools/llm_combo_diagnostics.py`.
- 2024-10-09: Ran combined TSLA,NVDA backtest (2024-10-01→2024-10-03) via `poetry run python src/backtester.py --model-provider azure --analysts-all --tickers TSLA,NVDA --start-date 2024-10-01 --end-date 2024-10-03 --log log/backtest_tsla_nvda_oct01_03.log`; timing JSON at `log/backtest_timings/backtest_TSLA_NVDA_20241001_to_20241003_20251009T095628Z.json` shows `agent_invoke` ≈97.4 % of total timed runtime, turnover_rate ≈6.66 %.
- Async persona parallelism checklist:
  - [ ] Introduce `run_hedge_fund_async` in `src/main.py`, compile the LangGraph with async support, and shim legacy callers via `asyncio.run`.
  - [ ] Convert persona agents in `src/agents/` to `async def`, offload blocking data fetches via `asyncio.to_thread`, and expose async progress helpers.
  - [ ] Extend `src/agents/persona_utils.py`/`src/utils/llm.py` with `async_call_llm` (using `llm.ainvoke`), retries, and concurrency semaphores plus async persona wrappers.
  - [ ] Guard shared `state["data"]["analyst_signals"]` and risk state writes with an `asyncio.Lock` helper to avoid races.
  - [ ] Update `BacktestEngine` instrumentation to await the async workflow, capture per-agent durations, and persist the expanded timing JSON.
  - [ ] Gate the async path behind `ASYNC_PERSONAS=1` (default off) so CLI/backtester users can toggle the new engine.
  - [ ] Add pytest coverage for async persona execution and an async backtest smoke test with mocked LLM latency.
