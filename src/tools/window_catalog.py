from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.tools.api import get_prices, prices_to_df


@dataclass(frozen=True)
class WindowMetrics:
    start_date: date
    end_date: date
    mean_return: float
    min_return: float
    max_return: float
    dispersion: float
    mean_volatility: float
    per_ticker_returns: dict[str, float]
    per_ticker_volatility: dict[str, float]


def _coerce_date(value: datetime | pd.Timestamp) -> date:
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_convert(None)
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    raise TypeError(f"Unsupported date type: {type(value)}")


def _fetch_price_frame(ticker: str, start: str, end: str) -> pd.DataFrame:
    prices = get_prices(ticker=ticker, start_date=start, end_date=end)
    if not prices:
        return pd.DataFrame()

    df = prices_to_df(prices)
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = df.sort_index()
    return df[["open", "high", "low", "close", "volume"]].rename(columns={col: f"{ticker}_{col}" for col in ("open", "high", "low", "close", "volume")})


def _join_price_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    base = pd.concat(frames, axis=1, join="outer")
    base = base.sort_index()
    return base


def _compute_window_metrics(
    *,
    price_panel: pd.DataFrame,
    tickers: Sequence[str],
    window_days: int,
) -> list[WindowMetrics]:
    closes = pd.DataFrame(
        {ticker: price_panel[f"{ticker}_close"] for ticker in tickers if f"{ticker}_close" in price_panel}
    ).dropna()
    if closes.empty:
        return []

    daily_returns = closes.pct_change()
    compound_returns = (1.0 + daily_returns).rolling(window_days).apply(np.prod, raw=False) - 1.0
    rolling_volatility = daily_returns.rolling(window_days).std(ddof=0)

    metrics: list[WindowMetrics] = []
    index = closes.index
    for idx in range(window_days - 1, len(index)):
        end_dt = index[idx]
        start_dt = index[idx - window_days + 1]

        per_returns = compound_returns.iloc[idx]
        per_vol = rolling_volatility.iloc[idx]

        if per_returns.isna().any():
            continue

        mean_return = float(per_returns.mean())
        min_return = float(per_returns.min())
        max_return = float(per_returns.max())
        dispersion = max_return - min_return
        mean_vol = float(per_vol.mean(skipna=True)) if not per_vol.isna().all() else 0.0

        metrics.append(
            WindowMetrics(
                start_date=_coerce_date(start_dt),
                end_date=_coerce_date(end_dt),
                mean_return=mean_return,
                min_return=min_return,
                max_return=max_return,
                dispersion=dispersion,
                mean_volatility=mean_vol,
                per_ticker_returns={ticker: float(per_returns[ticker]) for ticker in per_returns.index},
                per_ticker_volatility={ticker: float(per_vol[ticker]) if not np.isnan(per_vol[ticker]) else 0.0 for ticker in per_returns.index},
            )
        )
    return metrics


def _select_non_overlapping(
    candidates: Iterable[tuple[str, WindowMetrics]],
    *,
    limit: int,
    min_gap_days: int = 2,
) -> list[tuple[str, WindowMetrics]]:
    selected: list[tuple[str, WindowMetrics]] = []
    for category, metrics in candidates:
        if len(selected) >= limit:
            break
        overlap = False
        for _, existing in selected:
            if metrics.start_date <= existing.end_date + timedelta(days=min_gap_days) and metrics.end_date >= existing.start_date - timedelta(days=min_gap_days):
                overlap = True
                break
        if not overlap:
            selected.append((category, metrics))
    return selected


def build_window_catalog(
    *,
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    window_days: int = 5,
    top_n: int = 6,
) -> dict:
    frames = [_fetch_price_frame(ticker, start_date, end_date) for ticker in tickers]
    frames = [frame for frame in frames if not frame.empty]
    price_panel = _join_price_frames(frames)
    if price_panel.empty:
        raise RuntimeError("No price data available for the requested tickers.")

    metrics = _compute_window_metrics(price_panel=price_panel, tickers=tickers, window_days=window_days)
    if not metrics:
        raise RuntimeError("Unable to compute window metrics from price history.")

    sorted_by_mean = sorted(metrics, key=lambda m: m.mean_return, reverse=True)
    sorted_by_loss = sorted(metrics, key=lambda m: m.mean_return)
    sorted_by_dispersion = sorted(metrics, key=lambda m: m.dispersion, reverse=True)
    sorted_by_vol = sorted(metrics, key=lambda m: m.mean_volatility, reverse=True)

    selections: list[tuple[str, WindowMetrics]] = []
    selections.extend(_select_non_overlapping((("broad_rally", m) for m in sorted_by_mean), limit=top_n))
    selections.extend(_select_non_overlapping((("broad_selloff", m) for m in sorted_by_loss), limit=top_n))
    selections.extend(_select_non_overlapping((("dispersion", m) for m in sorted_by_dispersion), limit=top_n))
    selections.extend(_select_non_overlapping((("volatility_spike", m) for m in sorted_by_vol), limit=top_n))

    seen: set[tuple[date, date]] = set()
    windows: list[dict] = []
    for category, window in selections:
        key = (window.start_date, window.end_date)
        if key in seen:
            continue
        seen.add(key)

        per_returns = window.per_ticker_returns
        best = max(per_returns.items(), key=lambda item: item[1])
        worst = min(per_returns.items(), key=lambda item: item[1])
        vol_leader = max(window.per_ticker_volatility.items(), key=lambda item: item[1])

        summary = (
            f"Avg {window_days}-day return {window.mean_return:.2%}; "
            f"leader {best[0]} {best[1]:.2%}, laggard {worst[0]} {worst[1]:.2%}. "
            f"Mean volatility {window.mean_volatility:.2%} (peak {vol_leader[0]} {vol_leader[1]:.2%})."
        )

        windows.append(
            {
                "category": category,
                "start_date": window.start_date.isoformat(),
                "end_date": window.end_date.isoformat(),
                "tickers": list(tickers),
                "summary": summary,
                "metrics": {
                    "mean_return": window.mean_return,
                    "min_return": window.min_return,
                    "max_return": window.max_return,
                    "dispersion": window.dispersion,
                    "mean_volatility": window.mean_volatility,
                    "per_ticker_returns": window.per_ticker_returns,
                    "per_ticker_volatility": window.per_ticker_volatility,
                },
            }
        )

    windows.sort(key=lambda item: item["start_date"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tickers": list(tickers),
        "window_days": window_days,
        "start_date": start_date,
        "end_date": end_date,
        "methodology": {
            "metrics": [
                "mean_return",
                "min_return",
                "max_return",
                "dispersion",
                "mean_volatility",
            ],
            "selection": {
                "categories": ["broad_rally", "broad_selloff", "dispersion", "volatility_spike"],
                "top_n": top_n,
                "min_gap_days": 2,
            },
        },
        "windows": windows,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build diversified backtest window catalog.")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of tickers to analyze.")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument("--window-days", type=int, default=5, help="Rolling window length in trading days.")
    parser.add_argument("--top-n", type=int, default=6, help="Number of top windows to sample per category before deduplication.")
    parser.add_argument("--output", type=Path, help="Optional output path for the JSON catalog.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    catalog = build_window_catalog(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        window_days=args.window_days,
        top_n=args.top_n,
    )
    output_path: Path | None = args.output
    if output_path is None:
        print(json.dumps(catalog, indent=2))
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(catalog, indent=2))
        print(f"Wrote catalog with {len(catalog['windows'])} windows to {output_path}")


if __name__ == "__main__":
    main()
