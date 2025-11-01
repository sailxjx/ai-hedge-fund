# Agent Catalog and Data Flow

## Workflow Overview
- `src/main.py` builds a LangGraph `StateGraph` pipeline composed of one `start_node`, the selected analysts, then the `risk_management_agent` and `portfolio_management_agent`.
- Every agent receives the shared `AgentState` with keys `messages`, `data`, and `metadata`. Analysts append their JSON payloads into `state['data']['analyst_signals']`.
- After all analyst nodes run in parallel fan-out from `start_node`, their outputs converge on `risk_management_agent`, which fetches historical prices via `src.tools.api.get_prices`, computes volatility/correlation budgets, and applies guard-rails emitted by analysts (momentum, trend, growth, mean reversion, stop-loss).
- The `portfolio_management_agent` (see `src/agents/portfolio_manager.py`) consumes risk overrides plus analyst consensus to produce final trade instructions returned to the CLI and optionally logged via `print_trading_output`.

## Core Agents

| Key | Name | Type | Order | Investing Style |
| --- | --- | --- | --- | --- |
| aswath_damodaran | Aswath Damodaran | analyst | 0 | Focuses on intrinsic value and financial metrics to assess investment opportunities through rigorous valuation analysis. |
| ben_graham | Ben Graham | analyst | 1 | Emphasizes a margin of safety and invests in undervalued companies with strong fundamentals through systematic value analysis. |
| bill_ackman | Bill Ackman | analyst | 2 | Seeks to influence management and unlock value through strategic activism and contrarian investment positions. |
| cathie_wood | Cathie Wood | analyst | 3 | Focuses on disruptive innovation and growth, investing in companies that are leading technological advancements and market disruption. |
| charlie_munger | Charlie Munger | analyst | 4 | Advocates for value investing with a focus on quality businesses and long-term growth through rational decision-making. |
| michael_burry | Michael Burry | analyst | 5 | Makes contrarian bets, often shorting overvalued markets and investing in undervalued assets through deep fundamental analysis. |
| mohnish_pabrai | Mohnish Pabrai | analyst | 6 | Focuses on value investing and long-term growth through fundamental analysis and a margin of safety. |
| peter_lynch | Peter Lynch | analyst | 6 | Invests in companies with understandable business models and strong growth potential using the 'buy what you know' strategy. |
| phil_fisher | Phil Fisher | analyst | 7 | Emphasizes investing in companies with strong management and innovative products, focusing on long-term growth through scuttlebutt research. |
| rakesh_jhunjhunwala | Rakesh Jhunjhunwala | analyst | 8 | Leverages macroeconomic insights to invest in high-growth sectors, particularly within emerging markets and domestic opportunities. |
| stanley_druckenmiller | Stanley Druckenmiller | analyst | 9 | Focuses on macroeconomic trends, making large bets on currencies, commodities, and interest rates through top-down analysis. |
| warren_buffett | Warren Buffett | analyst | 10 | Seeks companies with strong fundamentals and competitive advantages through value investing and long-term ownership. |
| technical_analyst | Technical Analyst | analyst | 11 | Focuses on chart patterns and market trends to make investment decisions, often using technical indicators and price action analysis. |
| growth_momentum | Growth Momentum Analyst | analyst | 12 | Quantifies rolling momentum and volatility compression to lean long in durable uptrends. |
| stat_mean_reversion | Statistical Mean Reversion | analyst | 13 | Looks for extreme deviations from rolling means and quantifies reversion odds via normal tail probabilities. |
| fundamentals_analyst | Fundamentals Analyst | analyst | 14 | Delves into financial statements and economic indicators to assess the intrinsic value of companies through fundamental analysis. |
| sentiment_analyst | Sentiment Analyst | analyst | 15 | Gauges market sentiment and investor behavior to predict market movements and identify opportunities through behavioral analysis. |
| valuation_analyst | Valuation Analyst | analyst | 16 | Specializes in determining the fair value of companies, using various valuation models and financial metrics for investment decisions. |
| momentum_guardian | Momentum Guardian | analyst | 17 | Screens technical momentum to prevent fighting strong bullish trends when sizing short exposure. |
| event_catalyst | Event Catalyst Analyst | analyst | 18 | Surfaces imminent catalysts from news flow to tilt short-term positioning. |
| trend_regime | Trend Regime Analyst | analyst | 19 | Quantifies medium-term trend strength and breakouts to bias the book toward prevailing momentum. |
| short_squeeze_guardian | Short Squeeze Guardian | analyst | 20 | Detects rapid upside squeezes using deterministic velocity, breadth, and volume metrics to cap short exposure. |
| short_cover_classifier | Short-Cover Classifier | analyst | 21 | Fits a logistic classifier on price/volume features to forecast upside squeezes and unwind stale shorts. |
| breakout_cover_sentinel | Breakout Cover Sentinel | analyst | 22 | Identifies durable upside breakouts and forces persistent shorts to cover while biasing the book long. |
| range_recovery_sentinel | Range Recovery Sentinel | analyst | 23 | Detects upside drift inside compressed ranges to unwind stale shorts and cap fresh short exposure. |
| regime_meta | Regime Meta-Model | analyst | 24 | Blends deterministic regime features with ML analyst probabilities to steer long-vs-short exposure nudges. |
| macro_volatility_sentinel | Macro Volatility Sentinel | analyst | 25 | Monitors volatility acceleration and drawdowns to suppress long exposure overrides ahead of crash regimes. |
| downside_flow_sentinel | Downside Flow Sentinel | analyst | 26 | Tracks persistent downside momentum and drawdowns to enforce defensive short tilts when crash flows emerge. |
| crash_short_allocator | Crash Short Allocator | analyst | 27 | Aggregates crash probability, volatility surges, and downside flow diagnostics into explicit short targets for risk overrides. |
| stop_loss_guardian | Stop-Loss Guardian | analyst | 28 | Monitors live positions and enforces deterministic stop-loss trims once adverse moves breach volatility bands. |

## Risk & Portfolio Surfaces
- `risk_management_agent` consumes historical prices, constructs volatility/correlation matrices, applies analyst-derived constraints (preferred directions, exposure caps), and outputs per-ticker risk budgets.
- `portfolio_management_agent` (plus helpers in `src/utils/portfolio.py`) consolidates analyst signals and risk overrides into executable orders while tracking cash, margin, and realized PnL.

## Data Tooling & Utilities
- Price data routes through `src/tools/api.py` (`get_prices`, `prices_to_df`), requiring `FINANCIAL_DATASETS_API_KEY` loaded from `.env` and extracted via `src/utils/api_key.get_api_key_from_state`.
- Progress updates are pushed to the CLI via `src/utils/progress.progress` to surface long-running fetches or model fits.
- Shared helpers for prompt construction, formatting, and analytics live under `src/utils/` and `src/tools/`.
