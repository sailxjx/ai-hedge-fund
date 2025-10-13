import asyncio
import copy
import inspect
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices_async, prices_to_df
from src.utils.api_key import get_api_key_from_state
from src.utils.async_state import update_analyst_signals_async, update_risk_state_async
from src.utils.progress import progress

# Backwards compatibility hook: tests monkeypatch `get_prices` directly.
get_prices = None

BULLISH_TREND_CONVICTION = 65
BEARISH_TREND_CONVICTION = 70

CONSENSUS_EXCLUDED_AGENTS = {
    "risk_management_agent",
    "risk_override_amplifier",
    "portfolio_manager",
    "portfolio_manager_agent",
    "crash_short_allocator_agent",
    "macro_volatility_sentinel_agent",
    "momentum_guardian_agent",
    "trend_regime_agent",
    "range_recovery_sentinel_agent",
    "short_squeeze_guardian_agent",
    "short_cover_classifier_agent",
    "breakout_cover_sentinel_agent",
    "downside_flow_sentinel_agent",
    "event_catalyst_agent",
    "stop_loss_guardian_agent",
    "regime_meta_agent",
    "growth_momentum_agent",
}

CONSENSUS_RELAX_BLOCKERS = {
    "RangeRecovery",
    "RegimeMeta",
    "Momentum",
    "Trend",
    "GrowthMomentum",
    "MeanReversion",
    "EventCatalyst",
    "ShortSqueeze",
    "ShortCover",
}

CAUTION_REBLOCK_TREND_THRESHOLD = 0.28
CAUTION_REBLOCK_STREAK_THRESHOLD = 2

SHORT_COVER_RELAX_MARGIN = 0.02
SHORT_COVER_IMPROVEMENT_FACTOR = 0.75

CONSENSUS_BEARISH_KEYWORDS = {
    "bear",
    "short",
    "sell",
    "trim",
    "reduce",
    "crash",
    "downside",
}

CONSENSUS_BULLISH_KEYWORDS = {
    "bull",
    "long",
    "buy",
    "accumulate",
    "add",
    "upside",
}

CONSENSUS_NEUTRAL_KEYWORDS = {
    "hold",
    "neutral",
    "flat",
    "wait",
}

CONSENSUS_VOTE_WEIGHT = 0.5
CONSENSUS_VOTE_CAP = 3.0
BEARISH_CONSENSUS_MIN = 3.0
BEARISH_CONSENSUS_DELTA = 1.0


# Directional analysts that may emit zero short targets to enforce a long bias.
# When crash-mode or crash overrides are active we ignore their zero targets so the
# crash allocator can surface its requested short exposure.
DIRECTIONAL_SHORT_SOURCES = {
    "Trend",
    "Momentum",
    "GrowthMomentum",
    "MeanReversion",
    "RegimeMeta",
    "RangeRecovery",
    "DownsideFlow",
}

FUNDAMENTAL_BEAR_AGENTS = {
    "aswath_damodaran_agent",
    "ben_graham_agent",
    "charlie_munger_agent",
    "michael_burry_agent",
    "mohnish_pabrai_agent",
    "warren_buffett_agent",
    "fundamentals_analyst_agent",
    "valuation_analyst_agent",
}

DOWNSIDE_SENTINEL_SIGNALS = {
    "bearish",
    "bear",
    "downside",
    "downside_trend",
    "short",
    "short_bias",
    "selloff",
}


@dataclass(slots=True)
class OverrideLogger:
    path: Path | None
    enabled: bool

    def write(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled or self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(payload, default=_json_default)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.write("\n")
        except OSError:
            # Logging failures should never abort trading logic; ignore silently.
            return


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return list(value)
    return value


def _remove_note(notes: list[str], text: str) -> None:
    while text in notes:
        notes.remove(text)


def _float_or_none(value: Any) -> float | None:
    """Best-effort float conversion that tolerates missing or invalid values."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _short_cover_near_trigger(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False

    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return False

    prob = _float_or_none(metrics.get("probability"))
    soft_threshold = _float_or_none(metrics.get("soft_threshold"))
    improvement = _float_or_none(metrics.get("improvement"))
    improvement_threshold = _float_or_none(metrics.get("improvement_threshold"))
    squeeze_trigger = bool(metrics.get("squeeze_trigger"))
    soft_trigger = bool(metrics.get("soft_trigger"))

    if squeeze_trigger or soft_trigger:
        return True

    near_probability = False
    if prob is not None and soft_threshold is not None:
        near_probability = prob >= soft_threshold - SHORT_COVER_RELAX_MARGIN

    near_improvement = False
    if improvement is not None:
        if improvement_threshold is not None and improvement_threshold > 0:
            near_improvement = improvement >= improvement_threshold * SHORT_COVER_IMPROVEMENT_FACTOR
        else:
            near_improvement = improvement > 0

    return near_probability or near_improvement


def _resolve_override_logger(metadata: Mapping[str, Any]) -> OverrideLogger:
    explicit_path = metadata.get("risk_override_log_path")
    enabled_flag = metadata.get("enable_override_logging")

    if explicit_path:
        return OverrideLogger(path=Path(str(explicit_path)), enabled=True)

    if not enabled_flag:
        return OverrideLogger(path=None, enabled=False)

    log_dir = Path(metadata.get("risk_override_log_dir") or "log/risk_overrides")
    run_label = metadata.get("run_label") or metadata.get("log_file") or metadata.get("log_path")
    if run_label:
        slug = Path(str(run_label)).stem
    else:
        slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = log_dir / f"{slug}.jsonl"
    return OverrideLogger(path=path, enabled=True)


@dataclass(slots=True)
class ConsensusMetrics:
    bullish_strength: float
    bearish_strength: float
    neutral_strength: float
    bullish_agents: list[str]
    bearish_agents: list[str]


def _confidence_to_weight(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(parsed):
        return 0.5
    scaled = max(0.0, min(100.0, parsed)) / 100.0
    return max(0.25, scaled)


def _normalise_signal(signal: Any) -> str:
    if not signal:
        return ""
    return str(signal).strip().lower()


def _classify_signal(signal: Any) -> str:
    normalised = _normalise_signal(signal)
    if not normalised:
        return "unknown"
    if any(keyword in normalised for keyword in CONSENSUS_BEARISH_KEYWORDS):
        return "bearish"
    if any(keyword in normalised for keyword in CONSENSUS_BULLISH_KEYWORDS):
        return "bullish"
    if any(keyword in normalised for keyword in CONSENSUS_NEUTRAL_KEYWORDS):
        return "neutral"
    return "unknown"


def _compute_consensus_metrics(
    analyst_signals: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ticker: str,
) -> ConsensusMetrics:
    bullish_strength = 0.0
    bearish_strength = 0.0
    neutral_strength = 0.0
    bullish_agents: list[str] = []
    bearish_agents: list[str] = []

    for agent_id, payload in analyst_signals.items():
        if agent_id in CONSENSUS_EXCLUDED_AGENTS:
            continue
        if not isinstance(payload, Mapping):
            continue

        view = payload.get(ticker)
        if not isinstance(view, Mapping):
            continue

        polarity = _classify_signal(view.get("signal"))
        if polarity == "unknown":
            continue

        weight = _confidence_to_weight(view.get("confidence"))
        if polarity == "bearish":
            bearish_strength += weight
            bearish_agents.append(agent_id)
        elif polarity == "bullish":
            bullish_strength += weight
            bullish_agents.append(agent_id)
        elif polarity == "neutral":
            neutral_strength += weight

    return ConsensusMetrics(
        bullish_strength=bullish_strength,
        bearish_strength=bearish_strength,
        neutral_strength=neutral_strength,
        bullish_agents=bullish_agents,
        bearish_agents=bearish_agents,
    )


async def _fetch_prices_for_risk_manager(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str | None,
):
    """Resolve price data using a patched synchronous helper when available."""
    override = globals().get("get_prices")
    if callable(override):
        result = override(ticker=ticker, start_date=start_date, end_date=end_date, api_key=api_key)
        if inspect.isawaitable(result):
            return await result
        return result
    return await get_prices_async(ticker=ticker, start_date=start_date, end_date=end_date, api_key=api_key)


##### Risk Management Agent #####
async def _risk_management_agent_impl(state: AgentState, agent_id: str = "risk_management_agent"):
    """Controls position sizing based on volatility-adjusted risk factors for multiple tickers."""
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]
    metadata = state.get("metadata", {})
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analyst_signals = data.get("analyst_signals", {})
    override_logger = _resolve_override_logger(metadata)
    rm_state = data.setdefault("risk_manager_state", {})
    crash_mode_registry: dict[str, dict[str, Any]] = rm_state.setdefault("crash_mode", {})

    # Initialize risk analysis for each ticker
    risk_analysis = {}
    current_prices = {}  # Store prices here to avoid redundant API calls
    volatility_data = {}  # Store volatility metrics
    returns_by_ticker: dict[str, pd.Series] = {}  # For correlation analysis

    # First, fetch prices and calculate volatility for all relevant tickers
    all_tickers = set(tickers) | set(portfolio.get("positions", {}).keys())

    for ticker in all_tickers:
        await progress.aupdate_status(agent_id, ticker, "Fetching price data and calculating volatility")

        prices = await _fetch_prices_for_risk_manager(
            ticker=ticker,
            start_date=data["start_date"],
            end_date=data["end_date"],
            api_key=api_key,
        )

        if not prices:
            await progress.aupdate_status(agent_id, ticker, "Warning: No price data found")
            volatility_data[ticker] = {"daily_volatility": 0.05, "annualized_volatility": 0.05 * np.sqrt(252), "volatility_percentile": 100, "data_points": 0}  # Default fallback volatility (5% daily)  # Assume high risk if no data
            continue

        prices_df = prices_to_df(prices)

        if not prices_df.empty and len(prices_df) > 1:
            current_price = prices_df["close"].iloc[-1]
            current_prices[ticker] = current_price

            # Calculate volatility metrics
            volatility_metrics = calculate_volatility_metrics(prices_df)
            volatility_data[ticker] = volatility_metrics

            # Store returns for correlation analysis (use close-to-close returns)
            daily_returns = prices_df["close"].pct_change().dropna()
            if len(daily_returns) > 0:
                returns_by_ticker[ticker] = daily_returns

            await progress.aupdate_status(agent_id, ticker, f"Price: {current_price:.2f}, Ann. Vol: {volatility_metrics['annualized_volatility']:.1%}")
        else:
            await progress.aupdate_status(agent_id, ticker, "Warning: Insufficient price data")
            current_prices[ticker] = 0
            volatility_data[ticker] = {"daily_volatility": 0.05, "annualized_volatility": 0.05 * np.sqrt(252), "volatility_percentile": 100, "data_points": len(prices_df) if not prices_df.empty else 0}

    # Build returns DataFrame aligned across tickers for correlation analysis
    correlation_matrix = None
    if len(returns_by_ticker) >= 2:
        try:
            returns_df = pd.DataFrame(returns_by_ticker).dropna(how="any")
            if returns_df.shape[1] >= 2 and returns_df.shape[0] >= 5:
                correlation_matrix = returns_df.corr()
        except Exception:
            correlation_matrix = None

    # Determine which tickers currently have exposure (non-zero absolute position)
    active_positions = {t for t, pos in portfolio.get("positions", {}).items() if abs(pos.get("long", 0) - pos.get("short", 0)) > 0}

    # Calculate total portfolio value based on current market prices (Net Liquidation Value)
    total_portfolio_value = portfolio.get("cash", 0.0)

    for ticker, position in portfolio.get("positions", {}).items():
        if ticker in current_prices:
            # Add market value of long positions
            total_portfolio_value += position.get("long", 0) * current_prices[ticker]
            # Subtract market value of short positions
            total_portfolio_value -= position.get("short", 0) * current_prices[ticker]

    await progress.aupdate_status(agent_id, None, f"Total portfolio value: {total_portfolio_value:.2f}")

    # Calculate volatility- and correlation-adjusted risk limits for each ticker
    for ticker in tickers:
        await progress.aupdate_status(agent_id, ticker, "Calculating volatility- and correlation-adjusted limits")

        if ticker not in current_prices or current_prices[ticker] <= 0:
            await progress.aupdate_status(agent_id, ticker, "Failed: No valid price data")
            risk_analysis[ticker] = {"remaining_position_limit": 0.0, "current_price": 0.0, "reasoning": {"error": "Missing price data for risk calculation"}}
            continue

        current_price = current_prices[ticker]
        vol_data = volatility_data.get(ticker, {})

        # Calculate current market value of this position
        position = portfolio.get("positions", {}).get(ticker, {})
        long_value = position.get("long", 0) * current_price
        short_value = position.get("short", 0) * current_price
        current_position_value = abs(long_value - short_value)  # Use absolute exposure

        # Volatility-adjusted limit pct
        vol_adjusted_limit_pct = calculate_volatility_adjusted_limit(vol_data.get("annualized_volatility", 0.25))

        # Correlation adjustment
        corr_metrics = {
            "avg_correlation_with_active": None,
            "max_correlation_with_active": None,
            "top_correlated_tickers": [],
        }
        corr_multiplier = 1.0
        if correlation_matrix is not None and ticker in correlation_matrix.columns:
            # Compute correlations with active positions (exclude self)
            comparable = [t for t in active_positions if t in correlation_matrix.columns and t != ticker]
            if not comparable:
                # If no active positions, compare with all other available tickers
                comparable = [t for t in correlation_matrix.columns if t != ticker]
            if comparable:
                series = correlation_matrix.loc[ticker, comparable]
                # Drop NaNs just in case
                series = series.dropna()
                if len(series) > 0:
                    avg_corr = float(series.mean())
                    max_corr = float(series.max())
                    corr_metrics["avg_correlation_with_active"] = avg_corr
                    corr_metrics["max_correlation_with_active"] = max_corr
                    # Top 3 most correlated tickers
                    top_corr = series.sort_values(ascending=False).head(3)
                    corr_metrics["top_correlated_tickers"] = [{"ticker": idx, "correlation": float(val)} for idx, val in top_corr.items()]
                    corr_multiplier = calculate_correlation_multiplier(avg_corr)

        # Combine volatility and correlation adjustments
        combined_limit_pct = vol_adjusted_limit_pct * corr_multiplier
        position_limit = total_portfolio_value * combined_limit_pct

        if current_price > 0:
            risk_limit_shares = max(0, int(math.floor(position_limit / current_price)))
            residual_limit_value = max(0.0, position_limit - current_position_value)
            risk_remaining_shares = max(0, int(math.floor(residual_limit_value / current_price)))
        else:
            risk_limit_shares = None
            risk_remaining_shares = None

        constraint_notes: list[str] = []
        overrides: dict[str, Any] = {}
        pending_target_longs: list[tuple[int, str, str | None]] = []
        existing_long = int(position.get("long", 0) or 0)
        existing_short = int(position.get("short", 0) or 0)

        consensus_metrics = _compute_consensus_metrics(analyst_signals, ticker)
        consensus_short_vote = min(CONSENSUS_VOTE_CAP, consensus_metrics.bearish_strength * CONSENSUS_VOTE_WEIGHT)
        consensus_long_vote = min(CONSENSUS_VOTE_CAP, consensus_metrics.bullish_strength * CONSENSUS_VOTE_WEIGHT)
        bearish_consensus_active = consensus_metrics.bearish_strength >= BEARISH_CONSENSUS_MIN and (consensus_metrics.bearish_strength - consensus_metrics.bullish_strength) >= BEARISH_CONSENSUS_DELTA
        if bearish_consensus_active:
            constraint_notes.append("Bearish consensus pressure {:.2f} vs bullish {:.2f}".format(consensus_metrics.bearish_strength, consensus_metrics.bullish_strength))
            if consensus_metrics.bearish_agents:
                highlighted = ", ".join(agent.replace("_agent", "") for agent in consensus_metrics.bearish_agents[:3])
                constraint_notes.append(f"Consensus short push led by {highlighted}" + ("…" if len(consensus_metrics.bearish_agents) > 3 else ""))

        fundamental_bearish_count = sum(1 for agent in consensus_metrics.bearish_agents if agent in FUNDAMENTAL_BEAR_AGENTS)
        fundamentals_stack_active = fundamental_bearish_count >= 2

        consensus_relaxed_blocks: list[str] = []

        short_squeeze_payload = (analyst_signals.get("short_squeeze_guardian_agent") or {}).get(ticker, {})
        short_squeeze_constraints = dict(short_squeeze_payload.get("constraints") or {})
        short_cover_payload = (analyst_signals.get("short_cover_classifier_agent") or {}).get(ticker, {})
        short_cover_constraints = dict(short_cover_payload.get("constraints") or {})
        short_cover_metrics_raw = short_cover_payload.get("metrics")
        short_cover_metrics = short_cover_metrics_raw if isinstance(short_cover_metrics_raw, Mapping) else {}
        caution_unblock_reason = str(short_cover_metrics.get("new_short_unblock_reason") or "").strip().lower()
        if caution_unblock_reason:
            precision_trend_strength = _float_or_none(short_cover_metrics.get("precision_trend_strength"))
            precision_trim_streak_val = _float_or_none(short_cover_metrics.get("precision_boost_trim_streak"))
            precision_trim_streak = int(precision_trim_streak_val) if precision_trim_streak_val is not None else 0
            heavy_trend = precision_trend_strength is not None and precision_trend_strength >= CAUTION_REBLOCK_TREND_THRESHOLD
            streak_exceeds = precision_trim_streak >= CAUTION_REBLOCK_STREAK_THRESHOLD
            caution_reasons = {"boost_guidance", "near_threshold_guidance"}
            if caution_unblock_reason in caution_reasons and (heavy_trend or streak_exceeds):
                short_cover_constraints["block_new_shorts"] = True
                short_cover_constraints["allow_short"] = False
                short_cover_constraints.setdefault("max_additional_short_shares", 0)
                if "target_short_shares" not in short_cover_constraints:
                    short_cover_constraints["target_short_shares"] = 0
                note_parts: list[str] = []
                if heavy_trend:
                    note_parts.append("trend {:.3f}>= {:.3f}".format(precision_trend_strength, CAUTION_REBLOCK_TREND_THRESHOLD))
                if streak_exceeds:
                    note_parts.append(f"boost-trim streak {precision_trim_streak}")
                detail = "; ".join(note_parts) if note_parts else "trend/streak guard"
                reblock_note = f"Short-cover caution unblock suppressed ({caution_unblock_reason}; {detail})"
                if reblock_note not in constraint_notes:
                    constraint_notes.append(reblock_note)
        momentum_payload = (analyst_signals.get("momentum_guardian_agent") or {}).get(ticker, {})
        momentum_constraints = dict(momentum_payload.get("constraints") or {})
        breakout_payload = (analyst_signals.get("breakout_cover_sentinel_agent") or {}).get(ticker, {})
        breakout_constraints = dict(breakout_payload.get("constraints") or {})
        range_recovery_payload = (analyst_signals.get("range_recovery_sentinel_agent") or {}).get(ticker, {})
        range_recovery_constraints = dict(range_recovery_payload.get("constraints") or {})
        trend_payload = (analyst_signals.get("trend_regime_agent") or {}).get(ticker, {})
        trend_constraints = dict(trend_payload.get("constraints") or {})
        growth_payload = (analyst_signals.get("growth_momentum_agent") or {}).get(ticker, {})
        growth_constraints = dict(growth_payload.get("constraints") or {})
        mean_rev_payload = (analyst_signals.get("stat_mean_reversion_agent") or {}).get(ticker, {})
        mean_rev_constraints = dict(mean_rev_payload.get("constraints") or {})
        regime_meta_payload = (analyst_signals.get("regime_meta_agent") or {}).get(ticker, {})
        regime_meta_constraints = dict(regime_meta_payload.get("constraints") or {})
        event_payload = (analyst_signals.get("event_catalyst_agent") or {}).get(ticker, {})
        event_constraints = dict(event_payload.get("constraints") or {})
        volatility_payload = (analyst_signals.get("macro_volatility_sentinel_agent") or {}).get(ticker, {})
        volatility_constraints = dict(volatility_payload.get("constraints") or {})
        downside_payload = (analyst_signals.get("downside_flow_sentinel_agent") or {}).get(ticker, {})
        downside_constraints = dict(downside_payload.get("constraints") or {})
        crash_allocator_payload = (analyst_signals.get("crash_short_allocator_agent") or {}).get(ticker, {})
        crash_allocator_constraints = dict(crash_allocator_payload.get("constraints") or {})
        stop_payload = (analyst_signals.get("stop_loss_guardian_agent") or {}).get(ticker, {})
        stop_constraints = stop_payload.get("constraints") or {}

        event_signal = str(event_payload.get("signal") or "").lower() if isinstance(event_payload, Mapping) else ""
        event_preferred_direction = ""
        if isinstance(event_constraints, Mapping):
            event_preferred_direction = str(event_constraints.get("preferred_direction") or "").lower()
        if not event_preferred_direction and isinstance(event_payload, Mapping):
            event_preferred_direction = str(event_payload.get("preferred_direction") or "").lower()
        event_confidence = _float_or_none(event_payload.get("confidence")) if isinstance(event_payload, Mapping) else None
        event_bearish_bias = event_preferred_direction == "short" or event_signal in DOWNSIDE_SENTINEL_SIGNALS

        downside_signal = str(downside_payload.get("signal") or "").lower() if isinstance(downside_payload, Mapping) else ""
        downside_preferred_direction = ""
        if isinstance(downside_constraints, Mapping):
            downside_preferred_direction = str(downside_constraints.get("preferred_direction") or "").lower()
        downside_confidence = _float_or_none(downside_payload.get("confidence")) if isinstance(downside_payload, Mapping) else None
        downside_bearish_bias = downside_preferred_direction == "short" or downside_signal in DOWNSIDE_SENTINEL_SIGNALS or (downside_confidence is not None and downside_confidence >= 60 and "down" in downside_signal)

        catalyst_stack_active = False
        if event_bearish_bias:
            catalyst_stack_active = True if ((event_confidence is not None and event_confidence >= 60) or downside_bearish_bias) else False

        bearish_fundamental_catalyst_stack = bearish_consensus_active and fundamentals_stack_active and catalyst_stack_active

        short_squeeze_signal = str(short_squeeze_payload.get("signal") or "").lower() if isinstance(short_squeeze_payload, Mapping) else ""
        squeeze_constraints_raw = short_squeeze_payload.get("constraints") if isinstance(short_squeeze_payload, Mapping) else {}
        if not isinstance(squeeze_constraints_raw, Mapping):
            squeeze_constraints_raw = {}
        short_squeeze_hard_block = str(squeeze_constraints_raw.get("block_new_shorts") or "").lower() in {"true", "1"}

        trend_signal = str(trend_payload.get("signal", "")).lower() if trend_payload else ""
        trend_confidence = int(trend_payload.get("confidence", 0) or 0)
        bullish_trend = trend_signal == "bullish" and trend_confidence >= BULLISH_TREND_CONVICTION
        bearish_trend = trend_signal == "bearish" and trend_confidence >= BEARISH_TREND_CONVICTION

        trend_enforced_short_block = False
        trend_forced_mean_rev_cap = False

        trend_mean_rev_original_cap: float | None = None
        trend_mean_rev_original_pref: str | None = None

        if bearish_trend:
            if mean_rev_constraints.pop("preferred_direction", None) == "long":
                constraint_notes.append("Suppressed mean reversion long bias due to bearish trend conviction")
            if "max_short_exposure_pct" in mean_rev_constraints:
                mean_rev_constraints.pop("max_short_exposure_pct", None)
                constraint_notes.append("Relaxed mean reversion short cap under bearish trend")

        if bullish_trend:
            trend_mean_rev_original_cap = _float_or_none(mean_rev_constraints.get("max_short_exposure_pct"))
            candidate_cap = trend_mean_rev_original_cap if trend_mean_rev_original_cap is not None else 0.0
            if mean_rev_constraints.get("preferred_direction") == "short":
                trend_mean_rev_original_pref = "short"
                mean_rev_constraints.pop("preferred_direction", None)
                constraint_notes.append("Suppressed mean reversion short bias due to bullish trend conviction")
            mean_rev_constraints["max_short_exposure_pct"] = min(float(candidate_cap), 0.0)
            short_squeeze_constraints.setdefault("block_new_shorts", True)
            short_squeeze_constraints.setdefault("max_additional_short_shares", 0)
            short_squeeze_constraints.setdefault("max_short_exposure_pct", 0.0)
            short_squeeze_constraints["allow_short"] = False
            constraint_notes.append("Bullish trend confirmation blocking new shorts")
            trend_enforced_short_block = True
            if trend_mean_rev_original_cap is None or trend_mean_rev_original_cap > 0.0:
                trend_forced_mean_rev_cap = True
            else:
                trend_forced_mean_rev_cap = False
        else:
            trend_forced_mean_rev_cap = False

        constraint_sources = [
            ("DownsideFlow", downside_constraints),
            ("CrashAllocator", crash_allocator_constraints),
            ("EventCatalyst", event_constraints),
            ("VolatilitySentinel", volatility_constraints),
            ("ShortSqueeze", short_squeeze_constraints),
            ("ShortCover", short_cover_constraints),
            ("BreakoutCover", breakout_constraints),
            ("RangeRecovery", range_recovery_constraints),
            ("RegimeMeta", regime_meta_constraints),
            ("Momentum", momentum_constraints),
            ("Trend", trend_constraints),
            ("GrowthMomentum", growth_constraints),
            ("MeanReversion", mean_rev_constraints),
        ]

        payload_lookup = {
            "DownsideFlow": downside_payload,
            "CrashAllocator": crash_allocator_payload,
            "EventCatalyst": event_payload,
            "VolatilitySentinel": volatility_payload,
            "ShortSqueeze": short_squeeze_payload,
            "ShortCover": short_cover_payload,
            "BreakoutCover": breakout_payload,
            "RangeRecovery": range_recovery_payload,
            "RegimeMeta": regime_meta_payload,
            "Momentum": momentum_payload,
            "Trend": trend_payload,
            "GrowthMomentum": growth_payload,
            "MeanReversion": mean_rev_payload,
        }

        regime_meta_probs: Mapping[str, Any] = {}
        if isinstance(regime_meta_payload, Mapping):
            indicators = regime_meta_payload.get("indicators")
            if isinstance(indicators, Mapping):
                raw_probabilities = indicators.get("probabilities")
                if isinstance(raw_probabilities, Mapping):
                    regime_meta_probs = raw_probabilities

        crash_prob = _float_or_none(regime_meta_probs.get("crash")) if regime_meta_probs else None
        rally_prob = _float_or_none(regime_meta_probs.get("rally")) if regime_meta_probs else None

        ticker_mode = crash_mode_registry.get(ticker, {})
        crash_prob_streak = int(ticker_mode.get("streak", 0) or 0)
        crash_mode_active = bool(ticker_mode.get("active", False))

        if crash_prob is not None and crash_prob >= 0.55:
            crash_prob_streak += 1
        else:
            crash_prob_streak = 0

        if crash_prob_streak >= 2:
            crash_mode_active = True
        elif crash_mode_active:
            if crash_prob is None or crash_prob < 0.45:
                crash_mode_active = False

        crash_mode_registry[ticker] = {
            "streak": crash_prob_streak,
            "active": crash_mode_active,
            "last_prob": crash_prob,
        }

        downside_signal = ""
        if isinstance(downside_payload, Mapping):
            downside_signal = str(downside_payload.get("signal") or "").lower()

        crash_allocator_signal = str((crash_allocator_payload or {}).get("signal") or "").lower()
        raw_allocator_conf = _float_or_none((crash_allocator_payload or {}).get("confidence"))
        crash_allocator_conf_pct = 0.0
        if raw_allocator_conf is not None:
            crash_allocator_conf_pct = max(0.0, min(raw_allocator_conf / 100.0, 1.0))

        allocator_allocation = _float_or_none((crash_allocator_payload or {}).get("allocation_pct"))
        if allocator_allocation is None:
            indicators = (crash_allocator_payload or {}).get("indicators")
            if isinstance(indicators, Mapping):
                allocator_allocation = _float_or_none(indicators.get("allocation_pct"))
        allocator_allocation = allocator_allocation or 0.0

        crash_target_raw = (crash_allocator_constraints or {}).get("target_short_shares")
        crash_target_value = _float_or_none(crash_target_raw)
        crash_target_shares = int(crash_target_value) if crash_target_value is not None else None
        if crash_target_shares is not None:
            crash_target_shares = max(0, crash_target_shares)

        crash_raw_target_raw = (crash_allocator_constraints or {}).get("raw_target_short_shares")
        crash_raw_target_value = _float_or_none(crash_raw_target_raw)
        crash_raw_target_shares = int(crash_raw_target_value) if crash_raw_target_value is not None else None
        if crash_raw_target_shares is not None:
            crash_raw_target_shares = max(0, crash_raw_target_shares)
        if crash_raw_target_shares is None:
            crash_indicators = (crash_allocator_payload or {}).get("indicators")
            if isinstance(crash_indicators, Mapping):
                crash_indicator_target = _float_or_none(crash_indicators.get("raw_target_short_shares"))
                if crash_indicator_target is not None:
                    crash_raw_target_shares = max(0, int(crash_indicator_target))

        crash_allocator_target_for_push = crash_raw_target_shares
        if crash_allocator_target_for_push is None:
            crash_allocator_target_for_push = crash_target_shares

        crash_allocator_pushes_short = crash_allocator_target_for_push is not None and crash_allocator_target_for_push > existing_short and crash_allocator_signal in {"crash_short", "short_bias"}

        downside_crash_signal = downside_signal in {"crash_flow"}
        crash_prob_bias = crash_prob is not None and crash_prob >= 0.5

        crash_bias_weight = 0.0
        if crash_prob_bias and crash_prob is not None:
            crash_bias_weight += min(0.6, max(0.0, crash_prob - 0.5) * 2.4)
        if crash_prob is not None and rally_prob is not None:
            crash_bias_weight += min(0.3, max(0.0, crash_prob - rally_prob))
        if crash_allocator_pushes_short:
            if crash_allocator_conf_pct >= 0.55:
                crash_bias_weight += min(0.5, crash_allocator_conf_pct * 0.6 + allocator_allocation * 1.4)
            else:
                crash_bias_weight += min(0.3, crash_allocator_conf_pct * 0.5 + allocator_allocation)
        if downside_crash_signal:
            crash_bias_weight += 0.2

        crash_bias_weight = min(1.2, max(0.0, crash_bias_weight))

        crash_override_active = crash_allocator_pushes_short and (crash_prob_bias or downside_crash_signal or crash_allocator_conf_pct >= 0.7)

        if trend_enforced_short_block and bearish_fundamental_catalyst_stack and not crash_override_active and not crash_mode_active:
            for key in ("block_new_shorts", "max_additional_short_shares", "max_short_exposure_pct"):
                short_squeeze_constraints.pop(key, None)
            short_squeeze_constraints["allow_short"] = True
            if trend_mean_rev_original_cap is None:
                mean_rev_constraints.pop("max_short_exposure_pct", None)
            else:
                mean_rev_constraints["max_short_exposure_pct"] = trend_mean_rev_original_cap
            if trend_mean_rev_original_pref is not None:
                mean_rev_constraints["preferred_direction"] = trend_mean_rev_original_pref
            _remove_note(constraint_notes, "Bullish trend confirmation blocking new shorts")
            _remove_note(constraint_notes, "Suppressed mean reversion short bias due to bullish trend conviction")
            constraint_notes.append("Bearish fundamentals + catalyst stack override bullish trend short block")
            trend_enforced_short_block = False
            trend_forced_mean_rev_cap = False

        long_vote_threshold_base = 0.5 + crash_bias_weight

        if crash_mode_active:
            crash_mode_note = "Crash probability streak activated crash-mode override"
            if crash_mode_note not in constraint_notes:
                constraint_notes.append(crash_mode_note)
            short_relaxed = False
            for key in ("block_new_shorts", "max_additional_short_shares"):
                if key in short_squeeze_constraints:
                    short_squeeze_constraints.pop(key, None)
                    short_relaxed = True
            if short_squeeze_constraints.get("allow_short") is False:
                short_squeeze_constraints.pop("allow_short", None)
                short_relaxed = True
            short_cap_val = short_squeeze_constraints.get("max_short_exposure_pct")
            if isinstance(short_cap_val, (int, float)) and short_cap_val <= 0:
                short_squeeze_constraints.pop("max_short_exposure_pct", None)
                short_relaxed = True
            if short_relaxed:
                relax_note = "Crash mode relaxes ShortSqueeze guardrails"
                if relax_note not in constraint_notes:
                    constraint_notes.append(relax_note)

        if (crash_override_active or crash_mode_active) and trend_enforced_short_block:
            for key in ("block_new_shorts", "max_additional_short_shares", "max_short_exposure_pct"):
                short_squeeze_constraints.pop(key, None)
            if short_squeeze_constraints.get("allow_short") is False:
                short_squeeze_constraints.pop("allow_short", None)
            note = "Bullish trend confirmation blocking new shorts"
            while note in constraint_notes:
                constraint_notes.remove(note)
            override_note = "Crash allocator override ignores bullish trend short block"
            if crash_mode_active and not crash_override_active:
                override_note = "Crash-mode override ignores bullish trend short block"
            elif crash_override_active and crash_mode_active:
                override_note = "Crash override (allocator + mode) ignores bullish trend short block"
            constraint_notes.append(override_note)

        if (crash_override_active or crash_mode_active) and trend_forced_mean_rev_cap:
            if trend_mean_rev_original_cap is None:
                mean_rev_constraints.pop("max_short_exposure_pct", None)
            else:
                mean_rev_constraints["max_short_exposure_pct"] = trend_mean_rev_original_cap
            restore_note = "Crash allocator override restores mean reversion short cap"
            if crash_mode_active and not crash_override_active:
                restore_note = "Crash-mode override restores mean reversion short cap"
            elif crash_override_active and crash_mode_active:
                restore_note = "Crash override (allocator + mode) restores mean reversion short cap"
            constraint_notes.append(restore_note)

        short_cap_pct: float | None = None
        short_cap_source: str | None = None
        short_cap_force_deficit: int | None = None
        short_cap_target: int | None = None
        crash_short_cap_pct_value = _float_or_none(crash_allocator_constraints.get("max_short_exposure_pct"))
        crash_short_cap_pct_value = max(0.0, float(crash_short_cap_pct_value)) if crash_short_cap_pct_value is not None else None
        crash_cap_override_applied = False
        for name, constraint in constraint_sources:
            value = constraint.get("max_short_exposure_pct")
            if isinstance(value, (int, float)) and value >= 0:
                candidate_pct = float(value)
                if bearish_consensus_active and name in CONSENSUS_RELAX_BLOCKERS:
                    if name == "EventCatalyst" and event_confidence is not None and event_confidence > 65:
                        pass
                    elif name == "ShortSqueeze" and (short_squeeze_signal in {"squeeze_warning", "elevated_risk"} or short_squeeze_hard_block):
                        pass
                    elif name == "ShortCover":
                        short_cover_payload = payload_lookup.get("ShortCover")
                        if _short_cover_near_trigger(short_cover_payload):
                            note = "ShortCover retains short cap near squeeze trigger"
                            if note not in constraint_notes:
                                constraint_notes.append(note)
                        else:
                            consensus_relaxed_blocks.append(name)
                            continue
                    else:
                        consensus_relaxed_blocks.append(name)
                        continue
                if short_cap_pct is None or candidate_pct < short_cap_pct:
                    short_cap_pct = candidate_pct
                    short_cap_source = name

        if (crash_override_active or crash_mode_active) and crash_short_cap_pct_value is not None and crash_short_cap_pct_value > 0.0:
            if short_cap_pct is None or short_cap_pct <= 0.0:
                short_cap_pct = crash_short_cap_pct_value
                short_cap_source = "CrashAllocator"
                crash_cap_override_applied = True
            elif crash_short_cap_pct_value < short_cap_pct:
                short_cap_pct = crash_short_cap_pct_value
                short_cap_source = "CrashAllocator"
                crash_cap_override_applied = True
            elif short_cap_source != "CrashAllocator" and math.isclose(short_cap_pct, crash_short_cap_pct_value, rel_tol=1e-9, abs_tol=1e-9):
                short_cap_source = "CrashAllocator"
                crash_cap_override_applied = True

        if short_cap_pct is not None:
            short_cap_pct = max(0.0, short_cap_pct)
            if short_cap_source:
                note = f"{short_cap_source} short cap {short_cap_pct:.1%}"
                if crash_cap_override_applied and short_cap_source == "CrashAllocator":
                    note += " (override)"
                constraint_notes.append(note)

            effective_value = max(0.0, total_portfolio_value)
            short_cap_value = effective_value * short_cap_pct
            if current_price > 0:
                short_cap_shares = max(0, int(short_cap_value / current_price))
            else:
                short_cap_shares = 0

            if risk_limit_shares is not None:
                short_cap_shares = min(short_cap_shares, risk_limit_shares)

            additional_capacity = max(0, short_cap_shares - existing_short)
            if risk_remaining_shares is not None:
                additional_capacity = min(additional_capacity, risk_remaining_shares)

            existing_max_add = overrides.get("max_additional_short_shares")
            if isinstance(existing_max_add, (int, float)):
                try:
                    existing_cap = max(0, int(existing_max_add))
                except (TypeError, ValueError):
                    existing_cap = 0
                additional_capacity = min(additional_capacity, existing_cap)

            overrides["max_additional_short_shares"] = int(additional_capacity)
            if int(additional_capacity) <= 0:
                overrides["block_new_shorts"] = True

            current_target_short = overrides.get("target_short_shares")
            if isinstance(current_target_short, (int, float)):
                try:
                    current_target_int = max(0, int(current_target_short))
                except (TypeError, ValueError):
                    current_target_int = short_cap_shares
                else:
                    current_target_int = min(current_target_int, short_cap_shares)
            else:
                current_target_int = short_cap_shares

            overrides["target_short_shares"] = current_target_int
            short_cap_target = current_target_int

            if existing_short > current_target_int:
                short_cap_force_deficit = existing_short - current_target_int

        event_reduce_change = bool(event_constraints.get("reduce_position_change"))
        event_long_cap: int | None = None
        max_long_add = event_constraints.get("max_long_add")
        if isinstance(max_long_add, (int, float)):
            event_long_cap = existing_long + max(0, int(max_long_add))
        if event_reduce_change and event_long_cap is None:
            event_long_cap = existing_long

        downside_reduce_change = bool(downside_constraints.get("reduce_position_change"))
        downside_long_cap: int | None = None
        downside_cap_note: str | None = None

        max_additional_downside = downside_constraints.get("max_additional_long_shares")
        if isinstance(max_additional_downside, (int, float)):
            candidate = existing_long + max(0, int(max_additional_downside))
            downside_long_cap = candidate
            downside_cap_note = f"DownsideFlow cap long adds to {candidate} shares"

        max_long_add_downside = downside_constraints.get("max_long_add")
        if isinstance(max_long_add_downside, (int, float)):
            candidate = existing_long + max(0, int(max_long_add_downside))
            if downside_long_cap is None or candidate < downside_long_cap:
                downside_long_cap = candidate
                downside_cap_note = f"DownsideFlow cap long adds to {candidate} shares"

        if downside_reduce_change:
            if downside_long_cap is None or downside_long_cap > existing_long:
                downside_long_cap = existing_long
            downside_cap_note = "DownsideFlow freezing long adds during crash flow"

        if downside_long_cap is not None:
            if event_long_cap is None or downside_long_cap < event_long_cap:
                event_long_cap = downside_long_cap
            if downside_cap_note:
                constraint_notes.append(downside_cap_note)

        sentinel_reduce_change = bool(volatility_constraints.get("reduce_position_change"))
        sentinel_long_cap: int | None = None
        sentinel_cap_note: str | None = None

        max_additional_longs = volatility_constraints.get("max_additional_long_shares")
        if isinstance(max_additional_longs, (int, float)):
            candidate = existing_long + max(0, int(max_additional_longs))
            sentinel_long_cap = candidate
            sentinel_cap_note = f"VolatilitySentinel cap long adds to {candidate} shares"

        max_long_add_sentinel = volatility_constraints.get("max_long_add")
        if isinstance(max_long_add_sentinel, (int, float)):
            candidate = existing_long + max(0, int(max_long_add_sentinel))
            if sentinel_long_cap is None or candidate < sentinel_long_cap:
                sentinel_long_cap = candidate
                sentinel_cap_note = f"VolatilitySentinel cap long adds to {candidate} shares"

        if sentinel_reduce_change:
            if sentinel_long_cap is None or sentinel_long_cap > existing_long:
                sentinel_long_cap = existing_long
            sentinel_cap_note = "VolatilitySentinel freezing long adds during volatility spike"

        if sentinel_long_cap is not None:
            if event_long_cap is None or sentinel_long_cap < event_long_cap:
                event_long_cap = sentinel_long_cap
            if sentinel_cap_note:
                constraint_notes.append(sentinel_cap_note)

        max_long_caps_pct: list[tuple[float, str]] = []
        for name, constraint in constraint_sources:
            value = constraint.get("max_long_exposure_pct")
            if isinstance(value, (int, float)) and value >= 0:
                max_long_caps_pct.append((float(value), name))

        if max_long_caps_pct:
            cap_pct, cap_source = min(max_long_caps_pct, key=lambda x: x[0])
            capped_limit = total_portfolio_value * cap_pct
            if capped_limit < position_limit:
                position_limit = capped_limit
                constraint_notes.append(f"{cap_source} long cap {cap_pct:.1%}")

        allow_short_blocks: list[str] = [name for name, constraint in constraint_sources if constraint.get("allow_short") is False]

        if bearish_consensus_active and allow_short_blocks:
            filtered_allow_blocks: list[str] = []
            for name in allow_short_blocks:
                if name not in CONSENSUS_RELAX_BLOCKERS:
                    filtered_allow_blocks.append(name)
                    continue
                if name == "EventCatalyst":
                    if event_confidence is not None and event_confidence > 65:
                        filtered_allow_blocks.append(name)
                        continue
                if name == "ShortSqueeze":
                    if short_squeeze_signal in {"squeeze_warning", "elevated_risk"} or short_squeeze_hard_block:
                        filtered_allow_blocks.append(name)
                        continue
                if name == "ShortCover":
                    short_cover_payload = payload_lookup.get("ShortCover")
                    if _short_cover_near_trigger(short_cover_payload):
                        filtered_allow_blocks.append(name)
                        note = "ShortCover retains short block near squeeze trigger"
                        if note not in constraint_notes:
                            constraint_notes.append(note)
                        continue
                consensus_relaxed_blocks.append(name)
            allow_short_blocks = filtered_allow_blocks

        block_short_sources: list[str] = []

        if allow_short_blocks and (crash_override_active or crash_mode_active):
            directional_short_blocks = {"Trend", "Momentum", "EventCatalyst", "GrowthMomentum"}
            ignored_blocks = [name for name in allow_short_blocks if name in directional_short_blocks]
            allow_short_blocks = [name for name in allow_short_blocks if name not in directional_short_blocks]
            if ignored_blocks:
                override_label = "Crash override"
                if crash_mode_active and not crash_override_active:
                    override_label = "Crash-mode override"
                elif crash_override_active and crash_mode_active:
                    override_label = "Crash override (allocator + mode)"
                constraint_notes.append(f"{override_label} ignores directional short blocks from " + ", ".join(sorted(ignored_blocks)))

        if allow_short_blocks:
            overrides["block_new_shorts"] = True
            overrides["max_additional_short_shares"] = 0
            for name in allow_short_blocks:
                note = f"{name} blocks new shorts"
                if note not in constraint_notes:
                    constraint_notes.append(note)
        else:
            block_short_sources = [name for name, constraint in constraint_sources if constraint.get("block_new_shorts")]
            if bearish_consensus_active and block_short_sources:
                filtered_block_sources: list[str] = []
                for name in block_short_sources:
                    if name not in CONSENSUS_RELAX_BLOCKERS:
                        filtered_block_sources.append(name)
                        continue
                    if name == "EventCatalyst":
                        if event_confidence is not None and event_confidence > 65:
                            filtered_block_sources.append(name)
                            continue
                    if name == "ShortSqueeze":
                        if short_squeeze_signal in {"squeeze_warning", "elevated_risk"} or short_squeeze_hard_block:
                            filtered_block_sources.append(name)
                            continue
                    if name == "ShortCover":
                        short_cover_payload = payload_lookup.get("ShortCover")
                        if _short_cover_near_trigger(short_cover_payload):
                            filtered_block_sources.append(name)
                            note = "ShortCover retains short block near squeeze trigger"
                            if note not in constraint_notes:
                                constraint_notes.append(note)
                            continue
                    consensus_relaxed_blocks.append(name)
                block_short_sources = filtered_block_sources
            if block_short_sources and (crash_override_active or crash_mode_active):
                directional_block_sources = {"Trend", "Momentum", "EventCatalyst", "GrowthMomentum"}
                ignored_blocks = [name for name in block_short_sources if name in directional_block_sources]
                block_short_sources = [name for name in block_short_sources if name not in directional_block_sources]
                if ignored_blocks:
                    override_label = "Crash override"
                    if crash_mode_active and not crash_override_active:
                        override_label = "Crash-mode override"
                    elif crash_override_active and crash_mode_active:
                        override_label = "Crash override (allocator + mode)"
                    constraint_notes.append(f"{override_label} ignores block_new_shorts from " + ", ".join(sorted(ignored_blocks)))

            if block_short_sources:
                overrides["block_new_shorts"] = True
                overrides.setdefault("max_additional_short_shares", 0)
                for name in block_short_sources:
                    note = f"{name} blocks new shorts"
                    if note not in constraint_notes:
                        constraint_notes.append(note)

        if consensus_relaxed_blocks:
            relaxed_list = ", ".join(sorted(set(consensus_relaxed_blocks)))
            constraint_notes.append(f"Bearish consensus relaxed blocks from {relaxed_list}")

        target_long_candidates: list[tuple[int, str]] = []
        for name, constraint in constraint_sources:
            target_long = constraint.get("target_long_shares")
            if isinstance(target_long, (int, float)):
                target_long_candidates.append((max(0, int(target_long)), name))

        if target_long_candidates:
            target_long, target_source = max(target_long_candidates, key=lambda x: x[0])
            if target_long > existing_long:
                pending_target_longs.append((target_long, f"{target_source} target {target_long} longs", None))

        target_short_candidates: list[tuple[int, str]] = []
        for name, constraint in constraint_sources:
            target = constraint.get("target_short_shares")
            if not isinstance(target, (int, float)):
                continue

            candidate = max(0, int(target))
            if candidate == 0:
                if (crash_override_active or crash_mode_active) and name in DIRECTIONAL_SHORT_SOURCES:
                    continue
                if bearish_consensus_active and name in CONSENSUS_RELAX_BLOCKERS:
                    if name == "ShortCover":
                        short_cover_payload = payload_lookup.get("ShortCover")
                        if _short_cover_near_trigger(short_cover_payload):
                            note = "ShortCover holds zero short target near squeeze trigger"
                            if note not in constraint_notes:
                                constraint_notes.append(note)
                        else:
                            consensus_relaxed_blocks.append(name)
                            continue
                    else:
                        consensus_relaxed_blocks.append(name)
                        continue

            target_short_candidates.append((candidate, name))

        if target_short_candidates:
            target_short, target_source = min(target_short_candidates, key=lambda x: x[0])
            overrides["target_short_shares"] = target_short
            constraint_notes.append(f"{target_source} target {target_short} shorts")
            if event_reduce_change and target_short > existing_short:
                if crash_override_active or crash_mode_active:
                    override_label = "Crash allocator override" if crash_override_active else "Crash-mode override"
                    if crash_override_active and crash_mode_active:
                        override_label = "Crash override (allocator + mode)"
                    constraint_notes.append(f"{override_label} bypasses EventCatalyst short freeze")
                else:
                    overrides["target_short_shares"] = existing_short
                    constraint_notes.append("EventCatalyst holding short exposure steady pre-event")

        preferred_direction = None
        direction_votes = {"long": 0.0, "short": 0.0}
        for name, constraint in constraint_sources:
            preference = constraint.get("preferred_direction")
            if preference and not preferred_direction:
                preferred_direction = str(preference)
                constraint_notes.append(f"{name} prefers {preferred_direction}")
            if preference:
                payload_conf = payload_lookup.get(name, {}).get("confidence")
                weight = float(payload_conf) / 100.0 if payload_conf is not None else 0.5
                if preference.lower() in direction_votes:
                    direction_votes[preference.lower()] += weight

        if consensus_long_vote:
            direction_votes["long"] += consensus_long_vote
        if consensus_short_vote:
            direction_votes["short"] += consensus_short_vote

        if bearish_consensus_active and preferred_direction:
            lower_pref = preferred_direction.lower()
            if lower_pref == "long":
                can_flip_short = not allow_short_blocks and not block_short_sources and not overrides.get("block_new_shorts")
                preferred_direction = "short" if can_flip_short else "neutral"
                constraint_notes.append(f"Bearish consensus adjusted preferred direction to {preferred_direction}")

        force_cover_candidates: list[tuple[int, str, str | None]] = []
        if existing_short > 0:
            for name, constraint in constraint_sources:
                qty = constraint.get("force_cover_qty")
                if isinstance(qty, (int, float)) and qty > 0:
                    forced = min(existing_short, int(qty))
                    if forced > 0:
                        force_cover_candidates.append((forced, name, constraint.get("force_cover_reason")))

        if force_cover_candidates:
            forced_qty, source_name, force_reason = max(force_cover_candidates, key=lambda x: x[0])
            overrides.setdefault("force_cover_qty", forced_qty)
            if force_reason:
                overrides.setdefault("force_cover_reason", str(force_reason))
            else:
                overrides.setdefault(
                    "force_cover_reason",
                    f"{source_name} forcing cover of {forced_qty} shares",
                )

        if short_cap_force_deficit and short_cap_force_deficit > 0:
            existing_force = overrides.get("force_cover_qty")
            if isinstance(existing_force, (int, float)):
                try:
                    force_qty = max(int(existing_force), short_cap_force_deficit)
                except (TypeError, ValueError):
                    force_qty = short_cap_force_deficit
            else:
                force_qty = short_cap_force_deficit

            overrides["force_cover_qty"] = force_qty
            cap_reason = f"Short exposure above cap of {short_cap_target} shares" if short_cap_target is not None else "Short exposure above cap threshold"
            overrides.setdefault("force_cover_reason", cap_reason)

        if preferred_direction:
            overrides["preferred_direction"] = preferred_direction
            if preferred_direction.lower() == "long" and existing_short > 0:
                overrides.setdefault("force_cover_qty", existing_short)
                overrides.setdefault("force_cover_reason", "Trend regime enforcing long bias")

        force_cover_raw = overrides.get("force_cover_qty")
        if isinstance(force_cover_raw, (int, float)) and existing_short > 0:
            try:
                force_cover_int = max(0, int(force_cover_raw))
            except (TypeError, ValueError):
                force_cover_int = 0
            if force_cover_int > 0:
                remaining_short = max(existing_short - force_cover_int, 0)
                target_short_raw = overrides.get("target_short_shares")
                target_short_int: int | None = None
                if isinstance(target_short_raw, (int, float)):
                    try:
                        target_short_int = max(0, int(target_short_raw))
                    except (TypeError, ValueError):
                        target_short_int = None

                if target_short_int is None:
                    overrides["target_short_shares"] = remaining_short
                elif target_short_int > remaining_short:
                    overrides["target_short_shares"] = remaining_short
                    note = f"Force cover of {force_cover_int} trims target shorts to {remaining_short} shares"
                    if note not in constraint_notes:
                        constraint_notes.append(note)

        growth_inds = (growth_payload.get("indicators") or {}) if growth_payload else {}
        prob_up = growth_inds.get("prob_up")
        base_rate = growth_inds.get("base_rate")
        if prob_up is not None and base_rate is not None and prob_up >= min(0.99, base_rate * 1.25) and position.get("short", 0) > 0:
            overrides["force_cover_qty"] = int(position.get("short", 0))
            overrides["force_cover_reason"] = f"Growth momentum classifier prob_up {prob_up:.2f} exceeds baseline {base_rate:.2f}"

        # If long-directed analysts collectively outweigh shorts, enforce conservative short caps
        long_vote_delta = direction_votes["long"] - direction_votes["short"]
        vote_threshold = min(1.5, max(0.5, long_vote_threshold_base))
        if bearish_consensus_active and vote_threshold > 0.5:
            delta_adjust = max(0.0, consensus_short_vote - consensus_long_vote)
            delta_adjust = min(0.4, delta_adjust * 0.2)
            if delta_adjust:
                vote_threshold = max(0.5, vote_threshold - delta_adjust)

        if long_vote_delta >= vote_threshold:
            if crash_override_active or crash_mode_active:
                override_label = "Crash allocator override" if crash_override_active else "Crash-mode override"
                if crash_override_active and crash_mode_active:
                    override_label = "Crash override (allocator + mode)"
                constraint_notes.append(f"{override_label} keeps short adds open despite long vote delta {long_vote_delta:.2f} (thr {vote_threshold:.2f})")
            else:
                overrides["block_new_shorts"] = True
                overrides["max_additional_short_shares"] = 0
                if position.get("short", 0) > 0:
                    overrides.setdefault("force_cover_qty", int(position.get("short", 0)))
                    overrides.setdefault(
                        "force_cover_reason",
                        "Long-bias consensus across analysts",
                    )
                constraint_notes.append(f"Direction votes long delta {long_vote_delta:.2f} >= {vote_threshold:.2f}; blocking new shorts")

        long_weight = 0.0
        if prob_up is not None and base_rate is not None:
            long_weight += max(0.0, prob_up - base_rate)
        mean_rev_inds = (mean_rev_payload.get("indicators") or {}) if mean_rev_payload else {}
        prob_revert_up = mean_rev_inds.get("prob_revert_up")
        mean_rev_long_bias = 0.0
        if prob_revert_up is not None:
            mean_rev_long_bias = max(0.0, prob_revert_up - 0.5)
            if bearish_trend:
                mean_rev_long_bias = 0.0
        if mean_rev_long_bias:
            long_weight += mean_rev_long_bias
        regime_meta_inds = (regime_meta_payload.get("indicators") or {}).get("probabilities", {}) if regime_meta_payload else {}
        rally_prob = regime_meta_inds.get("rally") if isinstance(regime_meta_inds, dict) else None
        crash_prob = regime_meta_inds.get("crash") if isinstance(regime_meta_inds, dict) else None
        if rally_prob is not None and crash_prob is not None:
            try:
                long_weight += max(0.0, float(rally_prob) - float(crash_prob))
            except (TypeError, ValueError):
                pass
        elif rally_prob is not None:
            try:
                long_weight += max(0.0, float(rally_prob) - 0.5)
            except (TypeError, ValueError):
                pass

        if long_weight > 0 and current_price > 0:
            desired_long = min(position_limit, total_portfolio_value * min(0.3, long_weight))
            target_long_shares = int(desired_long / current_price)
            if target_long_shares > existing_long:
                pending_target_longs.append(
                    (
                        target_long_shares,
                        f"Allocating long exposure based on prob_up delta {long_weight:.2f}",
                        "Probabilistic long allocation",
                    )
                )

        short_bias_active = False
        if preferred_direction and str(preferred_direction).lower() == "short":
            short_bias_active = True
        elif direction_votes["short"] - direction_votes["long"] >= 0.5:
            short_bias_active = True

        if short_bias_active:
            overrides.pop("target_long_shares", None)
            if "target_long_shares" not in overrides and "force_buy_reason" in overrides:
                overrides.pop("force_buy_reason", None)
        elif pending_target_longs:
            best_target, note, reason = max(pending_target_longs, key=lambda x: x[0])
            if event_long_cap is not None:
                capped_target = min(best_target, event_long_cap)
                if capped_target < best_target:
                    if capped_target <= existing_long:
                        constraint_notes.append("EventCatalyst deferring incremental long adds pre-event")
                    else:
                        delta = max(0, capped_target - existing_long)
                        if delta > 0:
                            constraint_notes.append(f"EventCatalyst cap long add to +{delta} shares")
                best_target = capped_target
            forced_cover_active = bool(overrides.get("force_cover_qty") and existing_short > 0)
            if best_target > existing_long:
                if forced_cover_active:
                    constraint_notes.append(f"{note} (deferred until forced cover completes)")
                else:
                    overrides["target_long_shares"] = best_target
                    if reason:
                        overrides.setdefault("force_buy_reason", reason)
                    constraint_notes.append(note)

        capped_long_share_sources: list[tuple[int, str]] = []
        sentinel_max_shares = volatility_constraints.get("max_long_shares")
        if isinstance(sentinel_max_shares, (int, float)):
            capped_long_share_sources.append((max(0, int(sentinel_max_shares)), "VolatilitySentinel"))
        downside_max_shares = downside_constraints.get("max_long_shares")
        if isinstance(downside_max_shares, (int, float)):
            capped_long_share_sources.append((max(0, int(downside_max_shares)), "DownsideFlow"))

        if capped_long_share_sources:
            capped_shares, cap_source = min(capped_long_share_sources, key=lambda item: item[0])
            current_target = overrides.get("target_long_shares")
            effective_target = int(current_target) if isinstance(current_target, (int, float)) else existing_long
            if capped_shares < effective_target:
                overrides["target_long_shares"] = capped_shares
                constraint_notes.append(f"{cap_source} trims long exposure to {capped_shares} shares")
                if capped_shares <= existing_long:
                    overrides.pop("force_buy_reason", None)

        if stop_constraints.get("force_cover"):
            target_qty = stop_constraints.get("force_cover_qty")
            if isinstance(target_qty, (int, float)):
                forced_qty = int(max(0, target_qty))
            else:
                forced_qty = position.get("short", 0)
            existing_short_qty = int(position.get("short", 0) or 0)
            forced_qty = min(existing_short_qty, forced_qty)
            if forced_qty > 0:
                existing_force = overrides.get("force_cover_qty")
                if isinstance(existing_force, (int, float)):
                    try:
                        forced_qty = max(forced_qty, int(existing_force))
                    except (TypeError, ValueError):
                        forced_qty = max(forced_qty, existing_short_qty)
                overrides["force_cover_qty"] = forced_qty
                overrides["force_cover_reason"] = stop_payload.get("reasoning")

                stop_target_raw = stop_constraints.get("target_short_shares")
                if isinstance(stop_target_raw, (int, float)):
                    stop_target = max(0, int(stop_target_raw))
                else:
                    stop_target = existing_short_qty

                current_target_raw = overrides.get("target_short_shares")
                if isinstance(current_target_raw, (int, float)):
                    try:
                        current_target = max(0, int(current_target_raw))
                    except (TypeError, ValueError):
                        current_target = stop_target
                    stop_target = min(current_target, stop_target)

                overrides["target_short_shares"] = stop_target

            constraint_notes.append("Stop-loss triggered")

        elif "target_short_shares" in stop_constraints:
            target_shares = stop_constraints.get("target_short_shares")
            if isinstance(target_shares, (int, float)):
                overrides["target_short_shares"] = max(0, int(target_shares))

        if event_reduce_change:
            if crash_override_active or crash_mode_active:
                override_label = "Crash allocator override" if crash_override_active else "Crash-mode override"
                if crash_override_active and crash_mode_active:
                    override_label = "Crash override (allocator + mode)"
                constraint_notes.append(f"{override_label} bypasses EventCatalyst churn freeze")
            else:
                current_cap = overrides.get("max_additional_short_shares")
                if current_cap is None or (isinstance(current_cap, (int, float)) and current_cap > 0):
                    overrides["max_additional_short_shares"] = 0
                overrides.setdefault("block_new_shorts", True)
                constraint_notes.append("EventCatalyst limiting position churn ahead of catalyst")

        existing_short = int(position.get("short", 0) or 0)
        force_cover_raw = overrides.get("force_cover_qty")
        if isinstance(force_cover_raw, (int, float)) and existing_short > 0:
            try:
                force_cover_int = max(0, int(force_cover_raw))
            except (TypeError, ValueError):
                force_cover_int = 0
            if force_cover_int > 0:
                remaining_short = max(existing_short - force_cover_int, 0)
                target_short_raw = overrides.get("target_short_shares")
                target_short_int: int | None = None
                if isinstance(target_short_raw, (int, float)):
                    try:
                        target_short_int = max(0, int(target_short_raw))
                    except (TypeError, ValueError):
                        target_short_int = None

                if target_short_int is None or target_short_int > remaining_short:
                    overrides["target_short_shares"] = remaining_short

        raw_short_cap = overrides.get("max_additional_short_shares")
        short_cap_int: int | None
        if isinstance(raw_short_cap, (int, float)):
            try:
                short_cap_int = int(raw_short_cap)
            except (TypeError, ValueError):
                short_cap_int = None
            else:
                if short_cap_int < 0:
                    short_cap_int = 0
        else:
            short_cap_int = None

        short_adds_blocked = bool(overrides.get("block_new_shorts"))
        if short_cap_int is not None and short_cap_int <= 0:
            short_adds_blocked = True

        target_short_override = overrides.get("target_short_shares")
        target_short_int: int | None = None
        if isinstance(target_short_override, (int, float)):
            try:
                target_short_int = max(0, int(target_short_override))
            except (TypeError, ValueError):
                target_short_int = None

        if short_adds_blocked and target_short_int is not None and target_short_int > existing_short:
            overrides["target_short_shares"] = existing_short
            if "Short target clamped to existing exposure under blocked short adds" not in constraint_notes:
                constraint_notes.append("Short target clamped to existing exposure under blocked short adds")
            target_short_int = existing_short

        if target_short_int is not None and short_cap_int is not None and short_cap_int >= 0 and target_short_int > existing_short + short_cap_int:
            reachable_short = existing_short + short_cap_int
            overrides["target_short_shares"] = reachable_short
            additional_capacity = max(0, reachable_short - existing_short)
            existing_cap_value = overrides.get("max_additional_short_shares")
            try:
                existing_cap_int = max(0, int(existing_cap_value))
            except (TypeError, ValueError):
                existing_cap_int = None
            if existing_cap_int is None or existing_cap_int > additional_capacity:
                overrides["max_additional_short_shares"] = additional_capacity
            if additional_capacity <= 0:
                overrides["block_new_shorts"] = True
                note = "Short target clamped to existing exposure under zero short capacity"
            else:
                note = f"Short target trimmed to {reachable_short} shares to respect short capacity"
            if note not in constraint_notes:
                constraint_notes.append(note)
            target_short_int = reachable_short

        preferred_short = overrides.get("preferred_direction")
        if short_adds_blocked and str(preferred_short or "").lower() == "short" and existing_short <= 0:
            overrides["preferred_direction"] = "neutral"
            constraint_notes.append("Short bias relaxed to neutral because short adds are blocked and no short exposure remains")

        if current_price > 0:
            final_limit_shares = max(0, int(math.floor(position_limit / current_price)))
            final_remaining_value = max(0.0, position_limit - current_position_value)
            final_remaining_shares = max(0, int(math.floor(final_remaining_value / current_price)))
        else:
            final_limit_shares = None
            final_remaining_shares = None

        if final_limit_shares is not None:
            target_short_raw = overrides.get("target_short_shares")
            target_short_int: int | None = None
            if isinstance(target_short_raw, (int, float)):
                try:
                    target_short_int = max(0, int(target_short_raw))
                except (TypeError, ValueError):
                    target_short_int = None

            existing_cap_raw = overrides.get("max_additional_short_shares")
            existing_cap_int: int | None = None
            if isinstance(existing_cap_raw, (int, float)):
                try:
                    existing_cap_int = max(0, int(existing_cap_raw))
                except (TypeError, ValueError):
                    existing_cap_int = None

            max_total_short = final_limit_shares
            if final_remaining_shares is not None:
                max_total_short = min(max_total_short, existing_short + final_remaining_shares)
            if existing_cap_int is not None:
                max_total_short = min(max_total_short, existing_short + existing_cap_int)

            if target_short_int is not None and target_short_int > max_total_short:
                overrides["target_short_shares"] = max_total_short
                note = f"Risk limit caps short exposure to {max_total_short} shares"
                if note not in constraint_notes:
                    constraint_notes.append(note)
                target_short_int = max_total_short

            allowed_additional = max(0, final_limit_shares - existing_short)
            if final_remaining_shares is not None:
                allowed_additional = min(allowed_additional, final_remaining_shares)

            new_cap = int(allowed_additional)
            if existing_cap_int is None or new_cap < existing_cap_int:
                overrides["max_additional_short_shares"] = new_cap

            if new_cap <= 0:
                overrides["block_new_shorts"] = True
                note = "Risk limit blocks new short adds"
                if note not in constraint_notes:
                    constraint_notes.append(note)

            if existing_short > final_limit_shares:
                deficit = existing_short - final_limit_shares
                existing_force = overrides.get("force_cover_qty")
                existing_force_int: int | None = None
                if isinstance(existing_force, (int, float)):
                    try:
                        existing_force_int = max(0, int(existing_force))
                    except (TypeError, ValueError):
                        existing_force_int = None
                if existing_force_int is None or deficit > existing_force_int:
                    overrides["force_cover_qty"] = deficit
                    overrides.setdefault(
                        "force_cover_reason",
                        f"Exposure above risk limit of {final_limit_shares} shares",
                    )

        if crash_raw_target_shares is not None:
            overrides.setdefault(
                "crash_allocator_raw_target_shares",
                int(crash_raw_target_shares),
            )
            if crash_allocator_target_for_push is not None and crash_allocator_target_for_push > existing_short:
                clip_reasons: list[str] = []
                if overrides.get("block_new_shorts"):
                    clip_reasons.append("block_new_shorts guardrails")
                max_short_add = overrides.get("max_additional_short_shares")
                if isinstance(max_short_add, (int, float)) and int(max_short_add) <= 0:
                    clip_reasons.append("zero incremental short capacity")
                if short_cap_source and short_cap_pct is not None:
                    clip_reasons.append(f"{short_cap_source} cap {short_cap_pct:.1%}")
                if clip_reasons:
                    clip_note = f"Crash allocator raw target {int(crash_raw_target_shares)} clipped by " + ", ".join(dict.fromkeys(clip_reasons))
                    if clip_note not in constraint_notes:
                        constraint_notes.append(clip_note)

        position_limit = max(0.0, position_limit)
        effective_limit_pct = position_limit / total_portfolio_value if total_portfolio_value > 0 else 0.0

        # Calculate remaining limit for this position
        remaining_position_limit = max(0.0, position_limit - current_position_value)

        # Ensure we don't exceed available cash
        max_position_size = min(remaining_position_limit, portfolio.get("cash", 0))

        risk_snapshot = {
            "remaining_position_limit": float(max_position_size),
            "current_price": float(current_price),
            "volatility_metrics": {
                "daily_volatility": float(vol_data.get("daily_volatility", 0.05)),
                "annualized_volatility": float(vol_data.get("annualized_volatility", 0.25)),
                "volatility_percentile": float(vol_data.get("volatility_percentile", 100)),
                "data_points": int(vol_data.get("data_points", 0)),
            },
            "correlation_metrics": corr_metrics,
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position_value": float(current_position_value),
                "base_position_limit_pct": float(vol_adjusted_limit_pct),
                "correlation_multiplier": float(corr_multiplier),
                "combined_position_limit_pct": float(effective_limit_pct),
                "position_limit": float(position_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
                "risk_adjustment": (f"Volatility x Correlation adjusted: {effective_limit_pct:.1%} " f"(base {vol_adjusted_limit_pct:.1%})"),
                "crash_context": {
                    "override_active": crash_override_active,
                    "mode_active": crash_mode_active,
                    "probability_streak": crash_prob_streak,
                    "raw_target_shares": int(crash_raw_target_shares) if crash_raw_target_shares is not None else None,
                },
            },
        }

        final_target_raw = overrides.get("target_short_shares")
        final_cap_raw = overrides.get("max_additional_short_shares")
        final_target_int: int | None = None
        final_cap_int: int | None = None

        if isinstance(final_target_raw, (int, float)):
            try:
                final_target_int = max(0, int(final_target_raw))
            except (TypeError, ValueError):
                final_target_int = None

        if isinstance(final_cap_raw, (int, float)):
            try:
                final_cap_int = max(0, int(final_cap_raw))
            except (TypeError, ValueError):
                final_cap_int = None

        if final_target_int is not None:
            max_short_allowed = existing_short
            if final_cap_int is not None:
                max_short_allowed += final_cap_int
            if overrides.get("block_new_shorts"):
                max_short_allowed = existing_short
            if final_cap_int is None and not overrides.get("block_new_shorts"):
                max_short_allowed = max(final_target_int, max_short_allowed)
            if final_target_int > max_short_allowed:
                harmonised_target = max_short_allowed
                overrides["target_short_shares"] = harmonised_target
                if final_cap_int is not None:
                    overrides["max_additional_short_shares"] = max(0, harmonised_target - existing_short)
                if harmonised_target <= existing_short:
                    overrides["block_new_shorts"] = True
                note = f"Short target harmonised to {harmonised_target} shares based on available capacity"
                if note not in constraint_notes:
                    constraint_notes.append(note)

        if constraint_notes:
            risk_snapshot["reasoning"]["constraints"] = constraint_notes

        if overrides:
            risk_snapshot["overrides"] = overrides

        risk_analysis[ticker] = risk_snapshot

        override_logger.write(
            {
                "logged_at": datetime.now(timezone.utc).isoformat(),
                "ticker": ticker,
                "start_date": data.get("start_date"),
                "end_date": data.get("end_date"),
                "portfolio": {
                    "cash": float(portfolio.get("cash", 0.0)),
                    "position": {
                        "long": int(position.get("long", 0) or 0),
                        "short": int(position.get("short", 0) or 0),
                    },
                },
                "risk_snapshot": risk_snapshot,
                "analyst_inputs": {
                    "crash_short_allocator": crash_allocator_payload,
                    "regime_meta": regime_meta_payload,
                    "momentum_guardian": momentum_payload,
                    "trend_regime": trend_payload,
                    "growth_momentum": growth_payload,
                    "stat_mean_reversion": mean_rev_payload,
                    "event_catalyst": event_payload,
                },
            }
        )

        await progress.aupdate_status(agent_id, ticker, f"Adj. limit: {effective_limit_pct:.1%}, Available: ${max_position_size:.0f}")

    await progress.aupdate_status(agent_id, None, "Done")

    message = HumanMessage(
        content=json.dumps(risk_analysis),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Volatility-Adjusted Risk Management Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = risk_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def risk_management_agent(state: AgentState, agent_id: str = "risk_management_agent"):
    """Synchronous adapter that runs the async risk manager implementation."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_risk_management_agent_impl(state, agent_id))
    raise RuntimeError("risk_management_agent cannot run inside an active event loop; use risk_management_agent_async instead.")


async def risk_management_agent_async(state: AgentState, agent_id: str = "risk_management_agent"):
    """Async risk manager that updates shared agent state without thread offloading."""
    clone_state: AgentState = {
        "messages": list(state.get("messages", [])),
        "data": copy.deepcopy(state.get("data", {})),
        "metadata": copy.deepcopy(state.get("metadata", {})),
    }

    result = await _risk_management_agent_impl(clone_state, agent_id)

    analyst_payload = clone_state.get("data", {}).get("analyst_signals", {}).get(agent_id, {})
    await update_analyst_signals_async(state, agent_id, analyst_payload)

    risk_state = clone_state.get("data", {}).get("risk_manager_state")
    if risk_state is not None:
        await update_risk_state_async(state, copy.deepcopy(risk_state))

    if isinstance(result, dict) and result.get("messages"):
        latest_message = result["messages"][-1]
    else:
        latest_message = HumanMessage(content=json.dumps(analyst_payload), name=agent_id)

    messages = list(state.get("messages", [])) + [latest_message]

    return {
        "messages": messages,
        "data": state["data"],
    }


def calculate_volatility_metrics(prices_df: pd.DataFrame, lookback_days: int = 60) -> dict:
    """Calculate comprehensive volatility metrics from price data."""
    if len(prices_df) < 2:
        return {"daily_volatility": 0.05, "annualized_volatility": 0.05 * np.sqrt(252), "volatility_percentile": 100, "data_points": len(prices_df)}

    # Calculate daily returns
    daily_returns = prices_df["close"].pct_change().dropna()

    if len(daily_returns) < 2:
        return {"daily_volatility": 0.05, "annualized_volatility": 0.05 * np.sqrt(252), "volatility_percentile": 100, "data_points": len(daily_returns)}

    # Use the most recent lookback_days for volatility calculation
    recent_returns = daily_returns.tail(min(lookback_days, len(daily_returns)))

    # Calculate volatility metrics
    daily_vol = recent_returns.std()
    annualized_vol = daily_vol * np.sqrt(252)  # Annualize assuming 252 trading days

    # Calculate percentile rank of recent volatility vs historical volatility
    if len(daily_returns) >= 30:  # Need sufficient history for percentile calculation
        # Calculate 30-day rolling volatility for the full history
        rolling_vol = daily_returns.rolling(window=30).std().dropna()
        if len(rolling_vol) > 0:
            # Compare current volatility against historical rolling volatilities
            current_vol_percentile = (rolling_vol <= daily_vol).mean() * 100
        else:
            current_vol_percentile = 50  # Default to median
    else:
        current_vol_percentile = 50  # Default to median if insufficient data

    return {"daily_volatility": float(daily_vol) if not np.isnan(daily_vol) else 0.025, "annualized_volatility": float(annualized_vol) if not np.isnan(annualized_vol) else 0.25, "volatility_percentile": float(current_vol_percentile) if not np.isnan(current_vol_percentile) else 50.0, "data_points": len(recent_returns)}


def calculate_volatility_adjusted_limit(annualized_volatility: float) -> float:
    """
    Calculate position limit as percentage of portfolio based on volatility.

    Logic:
    - Low volatility (<15%): Up to 25% allocation
    - Medium volatility (15-30%): 15-20% allocation
    - High volatility (>30%): 10-15% allocation
    - Very high volatility (>50%): Max 10% allocation
    """
    base_limit = 0.20  # 20% baseline

    if annualized_volatility < 0.15:  # Low volatility
        # Allow higher allocation for stable stocks
        vol_multiplier = 1.25  # Up to 25%
    elif annualized_volatility < 0.30:  # Medium volatility
        # Standard allocation with slight adjustment based on volatility
        vol_multiplier = 1.0 - (annualized_volatility - 0.15) * 0.5  # 20% -> 12.5%
    elif annualized_volatility < 0.50:  # High volatility
        # Reduce allocation significantly
        vol_multiplier = 0.75 - (annualized_volatility - 0.30) * 0.5  # 15% -> 5%
    else:  # Very high volatility (>50%)
        # Minimum allocation for very risky stocks
        vol_multiplier = 0.50  # Max 10%

    # Apply bounds to ensure reasonable limits
    vol_multiplier = max(0.25, min(1.25, vol_multiplier))  # 5% to 25% range

    return base_limit * vol_multiplier


def calculate_correlation_multiplier(avg_correlation: float) -> float:
    """Map average correlation to an adjustment multiplier.
    - Very high correlation (>= 0.8): reduce limit sharply (0.7x)
    - High correlation (0.6-0.8): reduce (0.85x)
    - Moderate correlation (0.4-0.6): neutral (1.0x)
    - Low correlation (0.2-0.4): slight increase (1.05x)
    - Very low correlation (< 0.2): increase (1.10x)
    """
    if avg_correlation >= 0.80:
        return 0.70
    if avg_correlation >= 0.60:
        return 0.85
    if avg_correlation >= 0.40:
        return 1.00
    if avg_correlation >= 0.20:
        return 1.05
    return 1.10
