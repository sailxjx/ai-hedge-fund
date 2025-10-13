# TODO

## Active Initiatives

### Immediate Priorities
- [ ] Persona observation refactor (phase 2): convert the remaining personas (`momentum_guardian`, `regime_meta`, `stat_mean_reversion`, `cathie_wood`, `peter_lynch`, `rakesh_jhunjhunwala`, `short_cover_classifier`) to observation-only prompts; refresh helper outputs, invalidate cached calibrations, and extend coverage to their async wrappers.
- [ ] Guardrail calibration execution: apply the fixes scoped in `log/analysis/guardrail_calibration_plan_20251009.md` for `range_recovery_sentinel`, `downside_flow_sentinel`, `regime_meta`, and `event_catalyst`; document deltas and rerun focused async backtests before promotion.
- [ ] Observation-rich persona inputs: ensure multi-ticker observation tables, raw factor data, and current portfolio snapshots are bundled into a single prompt payload per analyst so the LLM evaluates cross-ticker context without heuristics.

### Validation & Metrics
- [ ] Backtest consistency study: after completing the phase 2 refactor, rerun paired backtests for `TSLA,NVDA,GOOGL` (2025-10-01→2025-10-07), `AAPL,MSFT,NVDA` (2025-09-22→2025-09-26), and `TSLA,AAPL,GOOGL` (2025-08-04→2025-08-08); capture variance tables in `log/analysis/` and append metrics to `log/backtest.csv`.
- [ ] Investment performance benchmarking: compare observation-only personas against the last heuristics-enabled runs by mining the digests above; summarize insight and next hypotheses in `log/analysis/guardrail_calibration_plan_rolling.md`.
- [ ] Async telemetry cost study: compare async vs. sync persona latency across a 3-day TSLA,NVDA,MSFT window; log Azure token/cost deltas and update `docs/async_parallelism.md`.

### Workflow Enablement
- [ ] Analyst backlog triage: maintain the persona prompt experiment queue (range recovery, downside flow, diversification personas) in `log/analysis/guardrail_calibration_plan_rolling.md` with hypotheses and expected diagnostics.
- [ ] Documentation & memory hygiene: as experiments conclude, refresh `MEMORY.md`, `AGENTS.md`, and `log/backtest.csv` notes so the automation loop inherits the latest rules and learnings without manual intervention.
