# Arena Baseline Notes

This document tracks the canonical baseline context used by the evolutionary arena.
Populate these sections as instrumentation lands and new benchmarks are established.

## Current Baselines
- **Smoke Test:** TSLA single-analyst (Ben Graham) one-day window executed 2025-10-13 10:30Z via `poetry run python src/main.py --model-provider azure --analysts ben_graham --tickers TSLA --start-date 2024-10-01 --end-date 2024-10-02`; stdout saved at `log/smoke_tests/tsla_smoke_20241001_20241002.log`. Outcome: hold posture under risk guardrail, no trades, async engine stable. Associated risk overrides captured in `log/risk_overrides/cli_TSLA_20241001_to_20241002_20251013T103023Z.jsonl`.
- **Azure routing check:** `poetry run python src/tools/agent_iteration.py --label azure-routing-check --generation-id baseline --arena-run-id baseline-smoke --genome-id base-persona --genome-label ben_graham_baseline --smoke-command "poetry run python src/main.py --model-provider azure --analysts ben_graham --tickers TSLA --start-date 2024-10-01 --end-date 2024-10-02" --no-stage --timeout 360` completed successfully, confirming Azure credentials/environment wiring.
- **Arena Windows:** Generated diversified 5-trading-day windows spanning 2022-10-05 through 2025-10-14 via `src.tools.window_catalog`. The catalog (`configs/evolution/window_catalog.json`) samples 22 regimes across {TSLA,NVDA,GOOGL,MSFT,AAPL}, covering broad rallies, selloffs, dispersion shocks, and volatility spikes using rolling compound returns/volatility to avoid bias toward any single month.
- **Generation gen-001 (arena-oct24):** Initial full-roster runs breached async limits (ratios 14–17 vs cap 8) so no promotions. Refined trios solved concurrency while improving performance: value trio returned +0.28%/+0.13% with mixed Sharpe (–2.08 / 7.80), whereas the growth trio booked +0.22% in both weeks with Sharpe 21.63 / 5.71 and no guardrail hits. Regression passes in Nov (+0.13% Sharpe 11.3) and Dec (+0.19% Sharpe 7.88) stayed compelling; Jan 2025 dipped –1.14% (Sharpe –11.95) under choppy tape, queued for diagnosis.

## Logging Schema & Instrumentation
- `log/backtest.csv` now records `generation_id`, `arena_run_id`, genome identifiers/labels, analyst weight JSON, patriarch variant, async mode, model name, Azure token spend, and auxiliary metadata paths in addition to the existing metrics columns.
- `src/tools/agent_iteration.py` accepts generation/arena metadata via CLI flags and merges JSON payloads for reproducible lineage logging.
- Governance monitor dry-run (`poetry run python src/backtesting/governance_monitor.py --dry-run`) highlights legacy guardrail breaches; risk override audit on the smoke test (`poetry run python src/backtesting/risk_override_audit.py --log log/smoke_tests/tsla_smoke_20241001_20241002.log --overrides log/risk_overrides/cli_TSLA_20241001_to_20241002_20251013T103023Z.jsonl --csv log/analysis/risk_override_audit_tsla_smoke_20241001_20241002.csv`) surfaced missing trade records because the run remained flat—origin expected, no action required.
- Evolution schedule template lives at `configs/evolution/generation_template.json`, outlining rotating 7-day windows across {TSLA,NVDA,GOOGL,MSFT,AAPL} with generation metadata, genome registry path, and diversity guard thresholds.
- `src/backtesting/experiment_scheduler` now enforces post-run guardrails: it compares async telemetry against `LLM_ASYNC_MAX_CONCURRENCY`, terminates lingering `src/backtester.py` processes, and logs governance violations to `log/arena/guardrails/events.jsonl` before any ledger entry is written.
- Arena selection decisions flow through `src/tools/arena_selection.py` using rules from `configs/selection_rules.yaml`. Each evaluation emits JSON/Markdown reports under `log/arena/selection/` capturing candidate fitness, gating violations, and promotion calls for audit.

## Outstanding Fill-Ins
- Document baseline performance metrics once the initial arena runs complete.
- Describe governance thresholds (risk override caps, diversity floors) after they are set.
- Link to dashboards or notebooks summarising lineage and telemetry trends when available.
- Investigate the January 2025 regression drawdown (–1.14% return, Sharpe –11.95) to determine whether it was data cadence or prompt posture; capture mitigations before promoting to production rotation.
