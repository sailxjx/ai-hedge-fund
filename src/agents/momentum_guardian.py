import asyncio
import json
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.persona_utils import async_persona_from_observations, persona_from_observations
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices_async, prices_to_df
from src.utils.api_key import get_api_key_from_state
from src.utils.async_state import update_analyst_signals_async
from src.utils.progress import progress


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, (float, int)):
            return float(value)
        return float(value)
    except (TypeError, ValueError):
        return default


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _recent_ohlc_records(prices: pd.DataFrame, limit: int = 12) -> list[dict[str, Any]]:
    recent = prices.tail(limit)[["open", "high", "low", "close", "volume"]]
    records: list[dict[str, Any]] = []
    for index, row in recent.iterrows():
        ts = index.tz_localize(None) if isinstance(index, pd.Timestamp) and index.tz is not None else index
        date_str = ts.date().isoformat() if hasattr(ts, "date") else str(ts)
        records.append(
            {
                "date": date_str,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"] or 0.0),
            }
        )
    return records


def _recent_return_records(prices: pd.DataFrame, limit: int = 12) -> list[dict[str, Any]]:
    returns = prices["close"].pct_change().tail(limit).dropna()
    records: list[dict[str, Any]] = []
    for index, value in returns.items():
        ts = index.tz_localize(None) if isinstance(index, pd.Timestamp) and index.tz is not None else index
        date_str = ts.date().isoformat() if hasattr(ts, "date") else str(ts)
        records.append({"date": date_str, "daily_return": float(value)})
    return records


PERSONA_NAME = "Orion"
PERSONA_ROLE = "a momentum guardian who protects the book from fighting dominant trends"
PERSONA_BACKSTORY = "Orion spent a decade trading trend-following strategies and now advises when our short book must respect momentum."
PERSONA_INSTRUCTIONS = (
    "Use the EMA alignment, price distance from trend anchors, momentum slopes, and RSI to judge whether shorts should be capped.",
    "Cite the raw OHLC tape and daily return stream to justify the call—highlight inflection days and volume surges instead of quoting hard-coded thresholds.",
    "Compare each name against the batch momentum table to understand which trends dominate or lag.",
    "Incorporate the shared portfolio snapshot so your judgement links to actual position risk and avoids deterministic caps.",
    "Call out elevated turnover or recent flips and recommend easing rotation instead of reversing the book day-after-day.",
    "Respect regime/macro context: when risk and regime signals skew bullish, prioritise trimming shorts over adding new ones unless momentum evidence is overwhelming.",
)
ALLOWED_SIGNALS = ["bullish_momentum", "neutral", "bearish_momentum"]
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


def _compute_momentum_snapshot(prices: pd.DataFrame) -> dict[str, Any] | None:
    if prices is None or prices.empty:
        return None

    close = prices["close"].dropna()
    if close.empty:
        return None

    current_price = _safe_float(close.iloc[-1])
    ema_fast = close.ewm(span=21, adjust=False).mean()
    ema_slow = close.ewm(span=55, adjust=False).mean()
    ema_long = close.ewm(span=200, adjust=False).mean() if len(close) >= 200 else None
    roc_10 = close.pct_change(periods=10)
    rsi_14 = _calculate_rsi(close, period=14)

    fast_val = _safe_float(ema_fast.iloc[-1], current_price)
    slow_val = _safe_float(ema_slow.iloc[-1], current_price)
    long_val: float | None
    if ema_long is not None:
        long_val = _safe_float(ema_long.iloc[-1], slow_val)
    else:
        long_val = None

    roc_val = _safe_float(roc_10.iloc[-1], 0.0)
    rsi_val = _safe_float(rsi_14.iloc[-1], 50.0)

    slope_mid_raw = float(ema_slow.diff().iloc[-1]) if len(ema_slow) > 1 else 0.0
    slope_long_raw = float(ema_long.diff().iloc[-1]) if ema_long is not None and len(ema_long) > 1 else 0.0

    slope_mid_pct = slope_mid_raw / slow_val if slow_val else 0.0
    slope_long_pct = slope_long_raw / long_val if long_val else 0.0

    price_vs_ema55_pct = (current_price - slow_val) / slow_val if slow_val else None
    price_vs_ema200_pct = (current_price - long_val) / long_val if long_val else None

    bullish_stack = long_val is not None and current_price > slow_val > long_val and slope_mid_pct >= -0.001 and slope_long_pct >= -0.0005 and roc_val >= -0.01
    bearish_stack = long_val is not None and current_price < slow_val < long_val and slope_mid_pct <= 0.001 and slope_long_pct <= 0.0005 and roc_val <= 0.01

    metrics = {
        "price": current_price,
        "ema21": fast_val,
        "ema55": slow_val,
        "ema200": long_val,
        "roc_10": roc_val,
        "rsi_14": rsi_val,
        "ema55_slope_pct": slope_mid_pct,
        "ema200_slope_pct": slope_long_pct,
        "price_vs_ema55_pct": price_vs_ema55_pct,
        "price_vs_ema200_pct": price_vs_ema200_pct,
    }

    trend_flags = {
        "bullish_stack_alignment": bool(bullish_stack),
        "bearish_stack_alignment": bool(bearish_stack),
        "price_above_ema55": bool(slow_val and current_price >= slow_val),
        "price_above_ema200": bool(long_val and current_price >= long_val),
        "ema21_above_ema55": bool(fast_val and slow_val and fast_val >= slow_val),
    }

    notes: list[str] = []
    if long_val is None:
        notes.append("Less than 200 trading days available; EMA200 substituted with EMA55.")
    if abs(roc_val) < 0.005:
        notes.append("10-day rate of change is close to flat; expect choppy trend-following signals.")
    if rsi_val > 70:
        notes.append("RSI > 70 indicates stretched bullish momentum.")
    elif rsi_val < 30:
        notes.append("RSI < 30 indicates stretched bearish momentum.")

    weekly_returns = close.pct_change(periods=5).dropna()
    weekly_return_value = float(weekly_returns.iloc[-1]) if not weekly_returns.empty else None
    weekly_vol = close.pct_change().rolling(5).std(ddof=0).dropna()
    weekly_vol_value = float(weekly_vol.iloc[-1]) if not weekly_vol.empty else None

    return {
        "metrics": metrics,
        "trend_flags": trend_flags,
        "notes": notes,
        "lookback_observations": len(close),
        "window_snapshot": {
            "lookback_days": 5,
            "compound_return": weekly_return_value,
            "realised_volatility": weekly_vol_value,
        },
        "recent_ohlc": _recent_ohlc_records(prices),
        "recent_daily_returns": _recent_return_records(prices),
    }


def _build_momentum_table(tickers: list[str], snapshots: dict[str, dict[str, Any]]) -> str:
    headers = [
        "Ticker",
        "Price",
        "EMA21",
        "EMA55",
        "EMA200",
        "Price-EMA55",
        "Price-EMA200",
        "ROC10",
        "RSI14",
        "EMA55 Δ",
        "EMA200 Δ",
    ]
    col_widths = [len(h) for h in headers]
    rows: list[list[str]] = []

    for ticker in tickers:
        snapshot = snapshots.get(ticker)
        metrics = snapshot.get("metrics") if snapshot else None

        row = [
            ticker,
            _format_number(metrics.get("price")) if metrics else "-",
            _format_number(metrics.get("ema21")) if metrics else "-",
            _format_number(metrics.get("ema55")) if metrics else "-",
            _format_number(metrics.get("ema200")) if metrics else "-",
            _format_percent(metrics.get("price_vs_ema55_pct")) if metrics else "-",
            _format_percent(metrics.get("price_vs_ema200_pct")) if metrics else "-",
            _format_percent(metrics.get("roc_10")) if metrics else "-",
            _format_number(metrics.get("rsi_14")) if metrics else "-",
            _format_percent(metrics.get("ema55_slope_pct")) if metrics else "-",
            _format_percent(metrics.get("ema200_slope_pct")) if metrics else "-",
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
    headers = ["Ticker", "Price", "Long", "Short", "Net"]
    col_widths = [len(h) for h in headers]
    rows: list[list[str]] = []

    positions = (portfolio.get("positions") or {}) if isinstance(portfolio, dict) else {}

    for ticker in tickers:
        pos = positions.get(ticker, {}) or {}
        long_shares = int(pos.get("long", 0) or 0)
        short_shares = int(pos.get("short", 0) or 0)
        net = long_shares - short_shares
        price = snapshots.get(ticker, {}).get("metrics", {}).get("price")

        row = [
            ticker,
            _format_number(price),
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
    metrics_map = {ticker: snap.get("metrics") for ticker, snap in snapshots.items() if snap and snap.get("metrics")}

    def order(metric: str, reverse: bool) -> list[str]:
        pairs = [(ticker, metrics.get(metric)) for ticker, metrics in metrics_map.items() if metrics.get(metric) is not None]
        pairs.sort(key=lambda item: item[1], reverse=reverse)
        return [ticker for ticker, _ in pairs]

    return {
        "roc_10_desc": order("roc_10", True),
        "roc_10_asc": order("roc_10", False),
        "rsi_14_desc": order("rsi_14", True),
        "price_vs_ema55_desc": order("price_vs_ema55_pct", True),
        "price_vs_ema200_desc": order("price_vs_ema200_pct", True),
        "ema55_slope_desc": order("ema55_slope_pct", True),
    }


def _peer_rankings_for_ticker(ticker: str, summary: dict[str, list[str]]) -> dict[str, Any]:
    rankings: dict[str, Any] = {}
    for key, ordered in summary.items():
        if not ordered:
            continue
        if ticker in ordered:
            rankings[key] = {"rank": ordered.index(ticker) + 1, "total": len(ordered)}
    return rankings


def _neutral_payload(reason: str) -> dict[str, Any]:
    return {
        "signal": "neutral",
        "confidence": 0,
        "reasoning": reason,
        "constraints": {},
    }


async def _collect_momentum_inputs(
    *,
    agent_id: str,
    tickers: list[str],
    start_date: str,
    end_date: str,
    api_key: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Fetch price history and derive momentum snapshots for tickers."""

    snapshots: dict[str, dict[str, Any]] = {}
    momentum_view: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Assessing momentum regime")
        prices = await get_prices_async(ticker=ticker, start_date=start_date, end_date=end_date, api_key=api_key)
        if not prices:
            momentum_view[ticker] = _neutral_payload("No price history available for momentum assessment.")
            continue

        prices_df = prices_to_df(prices)
        if prices_df.empty or len(prices_df) < MIN_OBSERVATIONS:
            momentum_view[ticker] = _neutral_payload(f"Insufficient observations ({len(prices_df)}) for momentum guard.")
            continue

        snapshot = _compute_momentum_snapshot(prices_df)
        if snapshot is None:
            momentum_view[ticker] = _neutral_payload("Unable to derive momentum metrics from price history.")
            continue

        snapshots[ticker] = snapshot

    return snapshots, momentum_view


def _finalize_momentum_sync(
    *,
    state: AgentState,
    agent_id: str,
    snapshots: dict[str, dict[str, Any]],
    momentum_view: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate persona decisions synchronously and persist results."""

    data = state["data"]
    tickers: list[str] = data["tickers"]
    portfolio = data.get("portfolio", {}) or {}

    momentum_table = _build_momentum_table(tickers, snapshots)
    portfolio_table = _build_portfolio_table(tickers, portfolio, snapshots)
    portfolio_snapshot = _build_portfolio_snapshot(portfolio)
    peer_summary = _build_peer_summary(snapshots)

    for ticker in tickers:
        snapshot = snapshots.get(ticker)
        if snapshot is None:
            continue

        observations = {
            "ticker": ticker,
            "tickers_in_batch": tickers,
            "momentum_metrics": snapshot["metrics"],
            "trend_flags": snapshot["trend_flags"],
            "notes": snapshot["notes"],
            "lookback_observations": snapshot["lookback_observations"],
            "batch_tables": {
                "momentum_metrics": momentum_table,
                "portfolio_positions": portfolio_table,
            },
            "portfolio_snapshot": portfolio_snapshot,
            "peer_summary": peer_summary,
            "peer_rankings": _peer_rankings_for_ticker(ticker, peer_summary),
            "window_snapshot": snapshot.get("window_snapshot"),
            "recent_ohlc": snapshot.get("recent_ohlc"),
            "recent_daily_returns": snapshot.get("recent_daily_returns"),
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
            default_reasoning="Defaulted to neutral after missing persona response.",
        )

        trend_flags = snapshot["trend_flags"]
        signal = decision.signal if decision.signal in ALLOWED_SIGNALS else "neutral"
        raw_confidence = decision.confidence if isinstance(decision.confidence, (int, float)) else 0.0
        confidence = int(np.clip(raw_confidence, 0, 100))
        reasoning = decision.reasoning or "Defaulted to neutral after missing persona response."
        constraints = dict(decision.constraints or {})

        payload = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "constraints": constraints,
            "metrics": snapshot["metrics"],
            "trend_flags": trend_flags,
            "meta": {"observations": observations},
        }
        momentum_view[ticker] = payload
        progress.update_status(agent_id, ticker, f"{payload['signal']}@{payload['confidence']}")

    message = HumanMessage(content=json.dumps(momentum_view), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(momentum_view, "Momentum Regime Analyst")

    state["data"].setdefault("analyst_signals", {})[agent_id] = momentum_view
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


async def _finalize_momentum_async(
    *,
    state: AgentState,
    agent_id: str,
    snapshots: dict[str, dict[str, Any]],
    momentum_view: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate persona decisions asynchronously and persist results."""

    data = state["data"]
    tickers: list[str] = data["tickers"]
    portfolio = data.get("portfolio", {}) or {}

    momentum_table = _build_momentum_table(tickers, snapshots)
    portfolio_table = _build_portfolio_table(tickers, portfolio, snapshots)
    portfolio_snapshot = _build_portfolio_snapshot(portfolio)
    peer_summary = _build_peer_summary(snapshots)

    for ticker in tickers:
        snapshot = snapshots.get(ticker)
        if snapshot is None:
            continue

        observations = {
            "ticker": ticker,
            "tickers_in_batch": tickers,
            "momentum_metrics": snapshot["metrics"],
            "trend_flags": snapshot["trend_flags"],
            "notes": snapshot["notes"],
            "lookback_observations": snapshot["lookback_observations"],
            "batch_tables": {
                "momentum_metrics": momentum_table,
                "portfolio_positions": portfolio_table,
            },
            "portfolio_snapshot": portfolio_snapshot,
            "peer_summary": peer_summary,
            "peer_rankings": _peer_rankings_for_ticker(ticker, peer_summary),
            "window_snapshot": snapshot.get("window_snapshot"),
            "recent_ohlc": snapshot.get("recent_ohlc"),
            "recent_daily_returns": snapshot.get("recent_daily_returns"),
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
            default_reasoning="Defaulted to neutral after missing persona response.",
        )

        trend_flags = snapshot["trend_flags"]
        signal = decision.signal if decision.signal in ALLOWED_SIGNALS else "neutral"
        raw_confidence = decision.confidence if isinstance(decision.confidence, (int, float)) else 0.0
        confidence = int(np.clip(raw_confidence, 0, 100))
        reasoning = decision.reasoning or "Defaulted to neutral after missing persona response."
        constraints = dict(decision.constraints or {})

        payload = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "constraints": constraints,
            "metrics": snapshot["metrics"],
            "trend_flags": trend_flags,
            "meta": {"observations": observations},
        }
        momentum_view[ticker] = payload
        progress.update_status(agent_id, ticker, f"{payload['signal']}@{payload['confidence']}")

    message = HumanMessage(content=json.dumps(momentum_view), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(momentum_view, "Momentum Regime Analyst")

    await update_analyst_signals_async(state, agent_id, momentum_view)
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


##### Momentum Regime Analyst #####
def momentum_guardian_agent(state: AgentState, agent_id: str = "momentum_guardian_agent"):
    """Evaluate momentum regime and gate short exposure when trends turn bullish."""

    data = state["data"]
    snapshots, momentum_view = asyncio.run(
        _collect_momentum_inputs(
            agent_id=agent_id,
            tickers=data["tickers"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            api_key=get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY"),
        )
    )

    return _finalize_momentum_sync(
        state=state,
        agent_id=agent_id,
        snapshots=snapshots,
        momentum_view=momentum_view,
    )


async def momentum_guardian_agent_async(state: AgentState, agent_id: str = "momentum_guardian_agent"):
    """Async momentum regime analyst to guard shorts."""

    data = state["data"]
    snapshots, momentum_view = await _collect_momentum_inputs(
        agent_id=agent_id,
        tickers=data["tickers"],
        start_date=data["start_date"],
        end_date=data["end_date"],
        api_key=get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY"),
    )

    return await _finalize_momentum_async(
        state=state,
        agent_id=agent_id,
        snapshots=snapshots,
        momentum_view=momentum_view,
    )
