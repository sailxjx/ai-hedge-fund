# Automated Backtest Evaluation Plan

## Inputs
- Accept one or more log paths (raw CLI logs stripped of ANSI) or structured JSON exports containing per-day trade tables.
- Expected metrics in text logs: final portfolio value, total return, Sharpe, Sortino, Max Drawdown.
- Risk-manager diagnostics to track: number of force-cover overrides, target_long_shares activations, block_new_shorts toggles.

## Derived Metrics
1. Aggregate Sharpe/Sortino/Return per run; compute deltas relative to baseline run keys.
2. Identify sessions where `prob_up` (from growth momentum) > 0.6 yet realized daily return < 0 to flag miscalibrated signals.
3. Count override compliance events to ensure risk instructions were respected.

## Output Format
- CSV summary table with columns: run_label, window, total_return, sharpe, sortino, max_drawdown, override_force_covers, override_target_longs, negative_return_prob_calls.
- Optional JSON payload for downstream dashboards.

## Next Steps
- Implement CLI `src/backtesting/evaluate_logs.py` exposing `--logs`, `--baseline` flags.
- Add pytest harness with sample log fixture representing baseline + candidate run.
