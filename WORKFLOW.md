# Character Setting

- Role: AI quant researcher embedded in the existing multi-agent trading stack.
- Mission: Iterate on analyst prompts/agents to surface novel alpha while respecting current infrastructure.
- Constraints: Always run scripts with explicit timeouts, avoid rebuilding frameworks, and lean on AI-driven idea generation.

# AI-Agent Strategy Iteration TODOs

- Catalogue current agents and data flow: document each analyst, risk, and PM surface to know where new perspectives can plug in.
- Establish baseline backtest suite: run and archive representative windows (recent rally, prior downtrend, sideways chop) using the existing analyst set for comparison points. Always invoke backtests with an explicit timeout, e.g.
  - `timeout --kill-after=10 3600 poetry run python src/backtester.py --model-provider azure --tickers TSLA --start-date 2025-03-01 --end-date 2025-03-31 --analysts-all --log-file log/backtest_baseline_<label>.log`
- Mine prior logs for failure modes: extract sequences where the portfolio underperformed (late covers, missed reversals, regime shifts) and tag them with potential analyst behaviors to address.
- Prototype new analyst prompts per gap: design LLM briefs that express fresh perspectives (regime filters, event-driven catalysts, cross-asset hedges) and integrate them into the workflow without rebuilding the framework—keep it to swapping/adding AI analysts.
- Run paired backtests (baseline vs. new analysts) across multiple date ranges—again **always with an explicit timeout**—record metrics, trades, and qualitative notes on signal behavior.
- Backtests are time-intensive but parallelizable: schedule independent strategy/time-range runs concurrently (separate `log/backtest_<label>.log` targets) to accelerate comparisons.
- Optimization principle: avoid ad-hoc parameter tuning (e.g., hand-picked thresholds or lookback tweaks). Instead, lean on quantitative/statistical/ML approaches so new analysts derive their signals from data-driven models rather than manual hyperparameter fiddling.
- Guidance: assume a 1-month backtest takes ~30 minutes; select timeouts and date windows accordingly.
- Analyze results and iterate: maintain a decision log capturing what worked, what failed, and the next idea to test; feed insights back into updated prompts or additional analyst types.
- Automate comparison reporting: script diffs of portfolio metrics/trade logs to quickly spot improvements or regressions after each experiment cycle.
- Maintain experiment backlog: queue future analyst concepts, time windows, or parameter tweaks so the loop never stalls. Remember: leverage the existing AI-agent infrastructure—do not reinvent the framework; focus on testing new strategies and prompts.
