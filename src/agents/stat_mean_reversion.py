from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.persona_utils import async_persona_from_observations, persona_from_observations
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices_async, prices_to_df
from src.utils.api_key import get_api_key_from_state
from src.utils.async_state import update_analyst_signals_async
from src.utils.calibration import apply_platt, get_platt_parameters
from src.utils.progress import progress


def _extend_start(start_date: str | None, buffer_days: int = 120) -> str | None:
    if not start_date:
        return None
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        return start_date
    return (start_dt - timedelta(days=buffer_days)).strftime("%Y-%m-%d")


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


PERSONA_NAME = "Mira"
PERSONA_ROLE = "a mean-reversion sleuth translating z-scores into positioning guidance"
PERSONA_BACKSTORY = "Mira built statistical arbitrage desks and now narrates when stretched prices should mean-revert."
PERSONA_INSTRUCTIONS = (
    "Weigh the z-score extreme, calibrated reversion probabilities, and recent volatility context before taking a stance.",
    "Use the batch probability table to compare names so you understand which extremes dominate the set.",
    "Tie your recommendation back to the current portfolio snapshot—protect when we are offsides and press when exposure is light.",
)
ALLOWED_SIGNALS = ["bullish", "bearish", "neutral"]
MIN_OBSERVATIONS = 30


def _format_number(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "-"
    try:
        if not np.isfinite(value):
            return "-"
    except TypeError:
        return "-"
    return f"{value:.{decimals}f}"


def _format_percent(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "-"
    try:
        if not np.isfinite(value):
            return "-"
    except TypeError:
        return "-"
    return f"{value * 100:.{decimals}f}%"


def _build_mean_reversion_snapshot(
    closes: pd.Series,
    calibration_params: dict[str, float] | None,
) -> dict[str, Any] | None:
    if closes is None or closes.empty:
        return None

    closes = closes.astype(float)
    lookback = len(closes)
    rolling_window = min(63, lookback)
    mean = closes.rolling(rolling_window).mean()
    std = closes.rolling(rolling_window).std(ddof=0)

    latest_mean = mean.iloc[-1]
    latest_std = std.iloc[-1]
    latest_close = closes.iloc[-1]
    if np.isnan(latest_mean) or np.isnan(latest_std) or latest_std == 0:
        return None

    z_score = (latest_close - latest_mean) / latest_std
    raw_prob_revert_up = _norm_cdf(-z_score)

    if calibration_params:
        prob_revert_up = apply_platt(z_score, calibration_params)
    else:
        prob_revert_up = raw_prob_revert_up
    prob_revert_down = 1.0 - prob_revert_up

    indicators = {
        "latest_close": float(latest_close),
        "z_score": float(z_score),
        "rolling_mean": float(latest_mean),
        "rolling_std": float(latest_std),
        "prob_revert_up": float(prob_revert_up),
        "prob_revert_down": float(prob_revert_down),
        "raw_prob_revert_up": float(raw_prob_revert_up),
        "calibration_applied": bool(calibration_params),
    }
    if calibration_params:
        indicators["calibration"] = {key: float(value) for key, value in calibration_params.items()}

    notes: list[str] = []
    if abs(z_score) >= 2.0:
        notes.append("Price sits more than 2 standard deviations from the rolling mean.")
    elif abs(z_score) >= 1.0:
        notes.append("Moderate deviation (>1σ) from the rolling mean.")
    if prob_revert_up >= 0.7:
        notes.append("Reversion-up probability above 70%; bullish fade favored if portfolio allows.")
    elif prob_revert_down >= 0.7:
        notes.append("Reversion-down probability above 70%; bearish fade favored if exposure manageable.")
    if prob_revert_up <= 0.55 and prob_revert_down <= 0.55:
        notes.append("Probabilities clustered near 50%; signal quality is marginal.")

    return {
        "indicators": indicators,
        "probabilities": {
            "revert_up": float(prob_revert_up),
            "revert_down": float(prob_revert_down),
            "raw_revert_up": float(raw_prob_revert_up),
        },
        "calibration_applied": bool(calibration_params),
        "calibration_params": {key: float(value) for key, value in calibration_params.items()} if calibration_params else None,
        "rolling_window": rolling_window,
        "lookback_observations": lookback,
        "notes": notes,
    }


def _build_probability_table(tickers: list[str], snapshots: dict[str, dict[str, Any]]) -> str:
    headers = [
        "Ticker",
        "Last Price",
        "Z-Score",
        "Prob Revert Up",
        "Prob Revert Down",
        "Raw Prob Up",
        "Rolling Mean",
        "Rolling Std",
    ]
    col_widths = [len(h) for h in headers]
    rows: list[list[str]] = []

    for ticker in tickers:
        snapshot = snapshots.get(ticker)
        indicators = snapshot.get("indicators") if snapshot else None

        row = [
            ticker,
            _format_number(indicators.get("latest_close")) if indicators else "-",
            _format_number(indicators.get("z_score")) if indicators else "-",
            _format_percent(indicators.get("prob_revert_up")) if indicators else "-",
            _format_percent(indicators.get("prob_revert_down")) if indicators else "-",
            _format_percent(indicators.get("raw_prob_revert_up")) if indicators else "-",
            _format_number(indicators.get("rolling_mean")) if indicators else "-",
            _format_number(indicators.get("rolling_std")) if indicators else "-",
        ]
        col_widths = [max(col_widths[idx], len(value)) for idx, value in enumerate(row)]
        rows.append(row)

    def fmt(row_values: list[str]) -> str:
        return " | ".join(row_values[idx].ljust(col_widths[idx]) for idx in range(len(headers)))

    header_line = fmt(headers)
    divider = "-+-".join("-" * width for width in col_widths)
    body = [fmt(row) for row in rows]
    return "\n".join([header_line, divider, *body])


def _build_portfolio_table(
    tickers: list[str],
    portfolio: dict[str, Any],
    snapshots: dict[str, dict[str, Any]],
) -> str:
    headers = ["Ticker", "Last Price", "Long", "Short", "Net"]
    col_widths = [len(h) for h in headers]
    rows: list[list[str]] = []

    positions = (portfolio.get("positions") or {}) if isinstance(portfolio, dict) else {}

    for ticker in tickers:
        pos = positions.get(ticker, {}) or {}
        long_shares = int(pos.get("long", 0) or 0)
        short_shares = int(pos.get("short", 0) or 0)
        net = long_shares - short_shares
        indicators = snapshots.get(ticker, {}).get("indicators", {})
        last_price = indicators.get("latest_close")

        row = [
            ticker,
            _format_number(last_price),
            str(long_shares),
            str(short_shares),
            str(net),
        ]
        col_widths = [max(col_widths[idx], len(value)) for idx, value in enumerate(row)]
        rows.append(row)

    def fmt(row_values: list[str]) -> str:
        return " | ".join(row_values[idx].ljust(col_widths[idx]) for idx in range(len(headers)))

    header_line = fmt(headers)
    divider = "-+-".join("-" * width for width in col_widths)
    body = [fmt(row) for row in rows]
    return "\n".join([header_line, divider, *body])


def _build_portfolio_snapshot(portfolio: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    if not isinstance(portfolio, dict):
        return snapshot

    for key in ("cash", "gross_exposure", "net_exposure", "leverage"):
        if key in portfolio and portfolio[key] is not None:
            snapshot[key] = float(portfolio[key])

    positions = portfolio.get("positions") or {}
    if isinstance(positions, dict):
        total_long = sum(int((pos or {}).get("long", 0) or 0) for pos in positions.values())
        total_short = sum(int((pos or {}).get("short", 0) or 0) for pos in positions.values())
        snapshot["positions_count"] = len(positions)
        snapshot["total_long_shares"] = total_long
        snapshot["total_short_shares"] = total_short
    return snapshot


def _build_peer_summary(snapshots: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    indicators_map = {ticker: snap.get("indicators") for ticker, snap in snapshots.items() if snap and snap.get("indicators")}

    def order(metric: str, reverse: bool) -> list[str]:
        pairs = [(ticker, indicators.get(metric)) for ticker, indicators in indicators_map.items() if indicators.get(metric) is not None]
        pairs.sort(key=lambda item: item[1], reverse=reverse)
        return [ticker for ticker, _ in pairs]

    return {
        "z_score_desc": order("z_score", True),
        "z_score_asc": order("z_score", False),
        "prob_revert_up_desc": order("prob_revert_up", True),
        "prob_revert_down_desc": order("prob_revert_down", True),
    }


def _peer_rankings_for_ticker(ticker: str, summary: dict[str, list[str]]) -> dict[str, Any]:
    rankings: dict[str, Any] = {}
    for key, ordered in summary.items():
        if not ordered:
            continue
        if ticker in ordered:
            rankings[key] = {"rank": ordered.index(ticker) + 1, "total": len(ordered)}
    return rankings


def _fallback_payload(*, confidence: int, reason: str) -> dict[str, Any]:
    return {
        "signal": "neutral",
        "confidence": confidence,
        "reasoning": reason,
        "constraints": {},
    }


async def _collect_mean_reversion_inputs(
    *,
    agent_id: str,
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
    api_key: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Fetch price history and derive mean reversion snapshots for tickers."""

    extended_start = _extend_start(start_date)
    snapshots: dict[str, dict[str, Any]] = {}
    signals: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching price history")

        prices = await get_prices_async(
            ticker=ticker,
            start_date=extended_start or start_date,
            end_date=end_date,
            api_key=api_key,
        )

        if not prices:
            signals[ticker] = _fallback_payload(confidence=40, reason="Missing price data for mean reversion analysis.")
            continue

        df = prices_to_df(prices)
        if df.empty or len(df) < MIN_OBSERVATIONS:
            signals[ticker] = _fallback_payload(
                confidence=42,
                reason="Need at least 30 observations to evaluate mean deviation.",
            )
            continue

        df = df.sort_index()
        closes = df["close"].astype(float)
        calibration_params = get_platt_parameters("stat_mean_reversion", ticker)
        snapshot = _build_mean_reversion_snapshot(closes, calibration_params)

        if snapshot is None:
            signals[ticker] = _fallback_payload(
                confidence=45,
                reason="Rolling statistics unavailable; staying neutral.",
            )
            continue

        snapshots[ticker] = snapshot

    return snapshots, signals


def _finalize_mean_reversion_sync(
    *,
    state: AgentState,
    agent_id: str,
    snapshots: dict[str, dict[str, Any]],
    signals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate persona decisions synchronously and persist results."""

    data = state.get("data", {})
    tickers = data.get("tickers", [])
    portfolio = data.get("portfolio", {}) or {}

    probability_table = _build_probability_table(tickers, snapshots)
    portfolio_table = _build_portfolio_table(tickers, portfolio, snapshots)
    portfolio_snapshot = _build_portfolio_snapshot(portfolio)
    peer_summary = _build_peer_summary(snapshots)

    for ticker in tickers:
        snapshot = snapshots.get(ticker)
        if snapshot is None:
            continue

        indicators = snapshot["indicators"]
        observations = {
            "ticker": ticker,
            "tickers_in_batch": tickers,
            "z_score": indicators["z_score"],
            "probabilities": snapshot["probabilities"],
            "calibration_applied": snapshot["calibration_applied"],
            "calibration_parameters": snapshot["calibration_params"],
            "rolling_window": snapshot["rolling_window"],
            "lookback_observations": snapshot["lookback_observations"],
            "notes": snapshot["notes"],
            "batch_tables": {
                "mean_reversion": probability_table,
                "portfolio_positions": portfolio_table,
            },
            "portfolio_snapshot": portfolio_snapshot,
            "peer_summary": peer_summary,
            "peer_rankings": _peer_rankings_for_ticker(ticker, peer_summary),
        }

        decision = persona_from_observations(
            state=state,
            agent_id=agent_id,
            persona_name=PERSONA_NAME,
            persona_role=PERSONA_ROLE,
            persona_backstory=PERSONA_BACKSTORY,
            allowed_signals=ALLOWED_SIGNALS,
            observations=observations,
            persona_instructions=PERSONA_INSTRUCTIONS,
            default_signal="neutral",
            default_confidence=55.0,
            default_reasoning="Defaulted to neutral after observation-only fallback.",
        )

        confidence = int(np.clip(decision.confidence, 0, 100))
        constraints = decision.constraints or {}
        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": confidence,
            "reasoning": decision.reasoning,
            "constraints": constraints,
            "indicators": indicators,
            "probabilities": snapshot["probabilities"],
            "meta": {
                "observations": observations,
            },
        }

        progress.update_status(
            agent_id,
            ticker,
            f"z={indicators['z_score']:.2f} → {payload['signal'].upper()} ({payload['confidence']})",
        )

        signals[ticker] = payload

    message = HumanMessage(content=json.dumps(signals), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(signals, "Statistical Mean Reversion Analyst")

    state["data"].setdefault("analyst_signals", {})[agent_id] = signals
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


async def _finalize_mean_reversion_async(
    *,
    state: AgentState,
    agent_id: str,
    snapshots: dict[str, dict[str, Any]],
    signals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate persona decisions asynchronously and persist results."""

    data = state.get("data", {})
    tickers = data.get("tickers", [])
    portfolio = data.get("portfolio", {}) or {}

    probability_table = _build_probability_table(tickers, snapshots)
    portfolio_table = _build_portfolio_table(tickers, portfolio, snapshots)
    portfolio_snapshot = _build_portfolio_snapshot(portfolio)
    peer_summary = _build_peer_summary(snapshots)

    for ticker in tickers:
        snapshot = snapshots.get(ticker)
        if snapshot is None:
            continue

        indicators = snapshot["indicators"]
        observations = {
            "ticker": ticker,
            "tickers_in_batch": tickers,
            "z_score": indicators["z_score"],
            "probabilities": snapshot["probabilities"],
            "calibration_applied": snapshot["calibration_applied"],
            "calibration_parameters": snapshot["calibration_params"],
            "rolling_window": snapshot["rolling_window"],
            "lookback_observations": snapshot["lookback_observations"],
            "notes": snapshot["notes"],
            "batch_tables": {
                "mean_reversion": probability_table,
                "portfolio_positions": portfolio_table,
            },
            "portfolio_snapshot": portfolio_snapshot,
            "peer_summary": peer_summary,
            "peer_rankings": _peer_rankings_for_ticker(ticker, peer_summary),
        }

        decision = await async_persona_from_observations(
            state=state,
            agent_id=agent_id,
            persona_name=PERSONA_NAME,
            persona_role=PERSONA_ROLE,
            persona_backstory=PERSONA_BACKSTORY,
            allowed_signals=ALLOWED_SIGNALS,
            observations=observations,
            persona_instructions=PERSONA_INSTRUCTIONS,
            default_signal="neutral",
            default_confidence=55.0,
            default_reasoning="Defaulted to neutral after observation-only fallback.",
        )

        confidence = int(np.clip(decision.confidence, 0, 100))
        constraints = decision.constraints or {}
        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": confidence,
            "reasoning": decision.reasoning,
            "constraints": constraints,
            "indicators": indicators,
            "probabilities": snapshot["probabilities"],
            "meta": {
                "observations": observations,
            },
        }

        progress.update_status(
            agent_id,
            ticker,
            f"z={indicators['z_score']:.2f} → {payload['signal'].upper()} ({payload['confidence']})",
        )

        signals[ticker] = payload

    message = HumanMessage(content=json.dumps(signals), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(signals, "Statistical Mean Reversion Analyst")

    await update_analyst_signals_async(state, agent_id, signals)
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


##### Statistical Mean Reversion Analyst #####
def stat_mean_reversion_agent(state: AgentState, agent_id: str = "stat_mean_reversion_agent"):
    """Uses z-score deviations from rolling means to assess reversion probabilities."""

    data = state.get("data", {})
    snapshots, signals = asyncio.run(
        _collect_mean_reversion_inputs(
            agent_id=agent_id,
            tickers=data.get("tickers", []),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            api_key=get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY"),
        )
    )

    return _finalize_mean_reversion_sync(
        state=state,
        agent_id=agent_id,
        snapshots=snapshots,
        signals=signals,
    )


async def stat_mean_reversion_agent_async(state: AgentState, agent_id: str = "stat_mean_reversion_agent"):
    """Async mean reversion analyst."""

    data = state.get("data", {})
    snapshots, signals = await _collect_mean_reversion_inputs(
        agent_id=agent_id,
        tickers=data.get("tickers", []),
        start_date=data.get("start_date"),
        end_date=data.get("end_date"),
        api_key=get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY"),
    )

    return await _finalize_mean_reversion_async(
        state=state,
        agent_id=agent_id,
        snapshots=snapshots,
        signals=signals,
    )
