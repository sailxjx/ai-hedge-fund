"""CLI to export ML analyst datasets and compute calibration diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.agents.growth_momentum import _extend_start as growth_extend_start
from src.agents.growth_momentum import _logistic_regression, _prepare_features
from src.agents.stat_mean_reversion import _extend_start as mean_rev_extend
from src.tools.api import get_prices, prices_to_df

BIN_COUNT = 10
CalibrationBin = dict[str, float]


@dataclass
class Window:
    """Represents a backtest calibration window."""

    label: str
    start_date: str
    end_date: str


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def _compute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    labels = y_true.astype(float)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return math.nan
    ranks = np.argsort(np.argsort(scores)) + 1
    pos_ranks = ranks[labels == 1]
    pos_count = pos_ranks.size
    neg_count = len(labels) - pos_count
    numerator = pos_ranks.sum() - pos_count * (pos_count + 1) / 2
    denominator = pos_count * neg_count
    return float(numerator / denominator)


def _compute_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs - y_true) ** 2))


def _reliability_table(y: np.ndarray, p: np.ndarray) -> list[CalibrationBin]:
    df = pd.DataFrame({"prob": p, "label": y})
    if df.empty:
        return []

    df = df.sort_values("prob").reset_index(drop=True)
    bin_count = min(BIN_COUNT, max(1, len(df)))
    if bin_count == 1:
        mean_prob = float(df["prob"].mean())
        return [
            {
                "mean_probability": mean_prob,
                "observed_positive_rate": float(df["label"].mean()),
                "count": int(len(df)),
            }
        ]

    df["bin"] = pd.qcut(df.index, q=bin_count, duplicates="drop")

    table: list[CalibrationBin] = []
    for _, group in df.groupby("bin"):
        avg_prob = float(group["prob"].mean())
        true_rate = float(group["label"].mean())
        table.append(
            {
                "mean_probability": avg_prob,
                "observed_positive_rate": true_rate,
                "count": int(group.shape[0]),
            }
        )
    return table


def _fetch_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    api_key: str | None,
) -> pd.DataFrame:
    try:
        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )
    except Exception as exc:  # pragma: no cover - defensive network guard
        template = "Failed price fetch for {ticker} ({start}->{end}): {exc}"
        message = template.format(
            ticker=ticker,
            start=start_date,
            end=end_date,
            exc=exc,
        )
        print(message)
        return pd.DataFrame()
    return prices_to_df(prices) if prices else pd.DataFrame()


def _export_growth_features(
    ticker: str,
    window: Window,
    output_dir: Path,
    *,
    api_key: str | None,
) -> dict[str, float | int | list[dict[str, float]]]:
    extended_start = growth_extend_start(window.start_date)
    df = _fetch_prices(
        ticker,
        extended_start or window.start_date,
        window.end_date,
        api_key=api_key,
    )

    if df.empty or len(df) < 90:
        return {
            "ticker": ticker,
            "window": window.label,
            "sample_count": int(len(df)),
            "note": "Insufficient observations for logistic calibration.",
        }

    df = df.sort_index()
    features_df, labels = _prepare_features(df["close"])

    if features_df.empty or labels.nunique() < 2:
        return {
            "ticker": ticker,
            "window": window.label,
            "sample_count": int(len(features_df)),
            "note": "Insufficient label variety for logistic calibration.",
        }

    dataset = features_df.copy()
    dataset.insert(0, "date", dataset.index)
    dataset["label"] = labels.values

    window_dir = output_dir / ticker / window.label
    window_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(window_dir / "growth_momentum_dataset.csv", index=False)

    feature_columns: list[str] = []
    for column in dataset.columns:
        if column not in {"date", "label"}:
            feature_columns.append(column)
    feature_matrix = dataset[feature_columns].to_numpy(dtype=float)
    intercept = np.ones((feature_matrix.shape[0], 1))
    X = np.hstack([intercept, feature_matrix])
    y = labels.to_numpy(dtype=float)

    weights = _logistic_regression(X, y)
    logits = X @ weights
    probs = _sigmoid(logits)

    metrics = {
        "ticker": ticker,
        "window": window.label,
        "sample_count": int(len(dataset)),
        "positive_rate": float(y.mean()),
        "auc": _compute_auc(y, probs),
        "brier_score": _compute_brier_score(y, probs),
    }

    reliability = _reliability_table(y, probs)
    diagnostics_path = window_dir / "growth_momentum_diagnostics.json"
    diagnostics = {
        "metrics": metrics,
        "reliability": reliability,
        "weights": weights.tolist(),
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2))

    return metrics


def _export_mean_reversion_features(
    ticker: str,
    window: Window,
    output_dir: Path,
    *,
    api_key: str | None,
) -> dict[str, float | int | list[dict[str, float]]]:
    extended_start = mean_rev_extend(window.start_date)
    df = _fetch_prices(
        ticker,
        extended_start or window.start_date,
        window.end_date,
        api_key=api_key,
    )

    if df.empty or len(df) < 30:
        note_msg = "Insufficient history for mean reversion calibration."
        return {
            "ticker": ticker,
            "window": window.label,
            "sample_count": int(len(df)),
            "note": note_msg,
        }

    closes = df.sort_index()["close"].astype(float)
    window_length = min(63, len(closes))
    rolling_mean = closes.rolling(window_length).mean()
    rolling_std = closes.rolling(window_length).std(ddof=0)
    future_returns = closes.pct_change().shift(-1)

    dataset = pd.DataFrame(
        {
            "date": closes.index,
            "close": closes.values,
            "rolling_mean": rolling_mean.values,
            "rolling_std": rolling_std.values,
            "future_return": future_returns.values,
        }
    ).dropna()

    if dataset.empty:
        return {
            "ticker": ticker,
            "window": window.label,
            "sample_count": 0,
            "note": "Rolling statistics produced no valid rows.",
        }

    mean_diff = dataset["close"] - dataset["rolling_mean"]
    dataset["z_score"] = mean_diff / dataset["rolling_std"]
    erf_argument = -dataset["z_score"] / math.sqrt(2.0)
    erf_values = np.vectorize(math.erf)(erf_argument)
    dataset["prob_revert_up"] = 0.5 * (1.0 + erf_values)
    dataset["up_move"] = (dataset["future_return"] > 0).astype(int)

    window_dir = output_dir / ticker / window.label
    window_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(window_dir / "mean_reversion_dataset.csv", index=False)

    probs = dataset["prob_revert_up"].to_numpy(dtype=float)
    labels = dataset["up_move"].to_numpy(dtype=float)
    metrics = {
        "ticker": ticker,
        "window": window.label,
        "sample_count": int(len(dataset)),
        "positive_rate": float(labels.mean()),
        "auc": _compute_auc(labels, probs),
        "brier_score": _compute_brier_score(labels, probs),
    }

    reliability = _reliability_table(labels, probs)
    diagnostics_path = window_dir / "mean_reversion_diagnostics.json"
    diagnostics = {"metrics": metrics, "reliability": reliability}
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2))

    return metrics


def parse_windows(window_args: Sequence[str]) -> list[Window]:
    windows: list[Window] = []
    for arg in window_args:
        try:
            label, start, end = arg.split(":")
        except ValueError as exc:  # pragma: no cover - defensive guard
            template = "Window spec '{}' must follow <label>:<start>:<end>"
            raise ValueError(template.format(arg)) from exc
        windows.append(Window(label=label, start_date=start, end_date=end))
    return windows


def main(argv: Sequence[str] | None = None) -> None:
    parser_description = "Export ML analyst datasets and diagnostics."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma separated tickers, e.g. TSLA,MSFT,QQQ",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        required=True,
        help="One or more <label>:<start>:<end> windows (YYYY-MM-DD dates)",
    )
    parser.add_argument(
        "--output-dir",
        default="log/ml_diagnostics",
        help="Directory root for exported datasets and diagnostics.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("FINANCIAL_DATASETS_API_KEY"),
        help="Override API key (defaults to FINANCIAL_DATASETS_API_KEY env).",
    )

    args = parser.parse_args(argv)

    tickers: list[str] = []
    for ticker in args.tickers.split(","):
        cleaned = ticker.strip()
        if cleaned:
            tickers.append(cleaned.upper())
    windows = parse_windows(args.windows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, dict[str, float]]] = {}

    for ticker in tickers:
        summary[ticker] = {}
        for window in windows:
            growth_metrics = _export_growth_features(
                ticker,
                window,
                output_dir,
                api_key=args.api_key,
            )
            mean_rev_metrics = _export_mean_reversion_features(
                ticker,
                window,
                output_dir,
                api_key=args.api_key,
            )
            summary[ticker][window.label] = {
                "growth_momentum": growth_metrics,
                "stat_mean_reversion": mean_rev_metrics,
            }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
