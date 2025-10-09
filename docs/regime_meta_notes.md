# Regime Meta-Model Notes

## Existing Inputs to Reuse
- `trend_regime_agent` features: EMA10/21/55 stacks, slope_mid (ΔEMA21 over 4 days), slope_long (ΔEMA55 over 9 days), momentum factors from 10- and 20-day returns, rolling 40-day high/low.
- `momentum_guardian_agent` features: EMA21/55/200 distances, 10-day rate of change, RSI14, composite momentum score driving short exposure caps.
- Additional ML analysts (`growth_momentum_agent`, `stat_mean_reversion_agent`) expose probabilities `prob_up`, `prob_revert_up` plus base rates that can serve as priors for direction.

## Candidate Feature Stack
1. **Volatility Trend**: rolling annualized vol slope (30-day vs 60-day) to capture expansion vs compression.
2. **Drawdown Depth**: current close vs trailing 60-day high to flag crash/stress regimes.
3. **Return Skew**: rolling skew of 5-day returns (sign flips often precede reversals).
4. **Probabilistic Inputs**: combine growth/mean-reversion probabilities with momentum score differences.

## Label Proposal
- Use directional labels from realized forward 5-day returns (>+1% rally, <-1% crash, else consolidation) to align with risk override categories.

## Integration Target
- Emit `preferred_direction`, `target_long_shares`, and `max_short_exposure_pct` payload for `risk_management_agent`, with confidence tied to calibrated probabilities.

Next action: prototype feature engineering script under `src/tools/regime_features.py` for offline experimentation.
