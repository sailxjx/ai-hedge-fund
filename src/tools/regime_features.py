"""CLI utility to compute regime classification features for meta-model research."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.tools.api import get_prices, prices_to_df


def _extend_start(start_date: str, buffer_days: int = 120) -> str:
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Start date must be YYYY-MM-DD, received '{start_date}'.") from exc
    extended = start_dt - timedelta(days=buffer_days)
    return extended.strftime("%Y-%m-%d")


def _fetch_price_frame(ticker: str, start_date: str, end_date: str, api_key: str | None) -> pd.DataFrame:
    prices = get_prices(ticker=ticker, start_date=start_date, end_date=end_date, api_key=api_key)
    if not prices:
        return pd.DataFrame()
    df = prices_to_df(prices)
    return df.sort_index()


def _annualized_volatility(returns: pd.Series, window: int) -> pd.Series:
    rolling_std = returns.rolling(window=window).std(ddof=0)
    return rolling_std * np.sqrt(252.0)


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    closes = df["close"].astype(float)
    returns = closes.pct_change().fillna(0.0)

    vol_30 = _annualized_volatility(returns, window=30)
    vol_60 = _annualized_volatility(returns, window=60)
    vol_slope = vol_30 - vol_60

    rolling_max_60 = closes.rolling(window=60).max()
    drawdown_depth = closes / rolling_max_60 - 1.0

    returns_5 = closes.pct_change(periods=5)
    return_skew = returns_5.rolling(window=20).skew()

    regime_features = pd.DataFrame(
        {
            "date": closes.index,
            "close": closes.values,
            "return_1d": returns.values,
            "vol_30": vol_30.values,
            "vol_60": vol_60.values,
            "volatility_slope": vol_slope.values,
            "drawdown_depth": drawdown_depth.values,
            "return5_skew20": return_skew.values,
        }
    )
    return regime_features.dropna().reset_index(drop=True)


def _write_outputs(dataset: pd.DataFrame, output_dir: Path, ticker: str, label: str) -> Path:
    ticker_dir = output_dir / ticker / label
    ticker_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = ticker_dir / "regime_features.csv"
    dataset.to_csv(dataset_path, index=False)

    drawdown_min = float(dataset["drawdown_depth"].min()) if not dataset.empty else 0.0
    summary_stats = {
        "sample_count": int(len(dataset)),
        "volatility_slope_mean": float(dataset["volatility_slope"].mean()) if not dataset.empty else 0.0,
        "drawdown_depth_min": drawdown_min,
        "return5_skew20_mean": float(dataset["return5_skew20"].mean()) if not dataset.empty else 0.0,
    }
    (ticker_dir / "summary.json").write_text(json.dumps(summary_stats, indent=2))
    return dataset_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute regime features for meta-model development.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., TSLA)")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--label", default="window", help="Label for output folder (e.g., rally)")
    parser.add_argument(
        "--output-dir",
        default="log/regime_features",
        help="Directory root for persisted feature tables.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional override for FINANCIAL_DATASETS_API_KEY.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    extended_start = _extend_start(args.start_date)
    prices = _fetch_price_frame(
        ticker=args.ticker,
        start_date=extended_start,
        end_date=args.end_date,
        api_key=args.api_key,
    )
    if prices.empty or len(prices) < 70:
        raise SystemExit(f"Insufficient price history for {args.ticker} between {args.start_date} and {args.end_date}.")

    features = _compute_features(prices)
    output_dir = Path(args.output_dir)
    dataset_path = _write_outputs(features, output_dir, args.ticker.upper(), args.label)
    print(f"Saved regime features to {dataset_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
