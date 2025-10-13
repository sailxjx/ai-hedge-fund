from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtesting.metrics import PerformanceMetricsCalculator


def _build_values(values: list[float]):
    start = datetime(2024, 1, 1)
    points = []
    for i, v in enumerate(values):
        points.append(
            {
                "Date": start + timedelta(days=i),
                "Portfolio Value": v,
                "Long Exposure": 0.0,
                "Short Exposure": 0.0,
                "Gross Exposure": 0.0,
                "Net Exposure": 0.0,
                "Long/Short Ratio": np.inf,
            }
        )
    return points


def _build_benchmark(values: list[float]):
    start = datetime(2024, 1, 1)
    points = []
    for i, v in enumerate(values):
        points.append(
            {
                "Date": start + timedelta(days=i),
                "Benchmark Value": v,
            }
        )
    return points


def test_metrics_insufficient_data_no_update():
    calc = PerformanceMetricsCalculator()
    metrics = {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}
    calc.update_metrics(metrics, _build_values([100_000.0]))
    assert metrics["sharpe_ratio"] is None
    assert metrics["sortino_ratio"] is None
    assert metrics["max_drawdown"] is None
    assert metrics.get("information_ratio") is None


def test_metrics_basic_sharpe_sortino_and_drawdown():
    calc = PerformanceMetricsCalculator(annual_trading_days=2, annual_rf_rate=0.0)
    # Values: up then down → non-zero volatility; drawdown occurs on last day
    vals = _build_values([100.0, 110.0, 99.0])
    metrics = {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}
    calc.update_metrics(metrics, vals)
    assert metrics["sharpe_ratio"] is not None
    assert metrics["sortino_ratio"] is not None
    assert metrics["max_drawdown"] < 0.0
    assert isinstance(metrics.get("max_drawdown_date"), str)
    assert metrics.get("information_ratio") is None


def test_metrics_zero_volatility_sharpe_zero():
    calc = PerformanceMetricsCalculator(annual_trading_days=252, annual_rf_rate=0.0)
    # Constant portfolio value → zero volatility → Sharpe 0
    vals = _build_values([100.0, 100.0, 100.0, 100.0])
    metrics = {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}
    calc.update_metrics(metrics, vals)
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics.get("information_ratio") is None


def test_information_ratio_with_benchmark():
    calc = PerformanceMetricsCalculator(annual_trading_days=2, annual_rf_rate=0.0)
    portfolio_vals = _build_values([100.0, 110.0, 120.0])
    benchmark_vals = _build_benchmark([100.0, 105.0, 110.0])
    metrics = {"sharpe_ratio": None, "sortino_ratio": None, "max_drawdown": None}
    calc.update_metrics(metrics, portfolio_vals, benchmark_vals)

    df = pd.DataFrame(portfolio_vals).set_index("Date")
    df["Daily Return"] = df["Portfolio Value"].pct_change()
    bench_df = pd.DataFrame(benchmark_vals).set_index("Date").reindex(df.index).ffill()
    combined = pd.concat([df["Daily Return"], bench_df["Benchmark Value"].pct_change()], axis=1).dropna()
    combined.columns = ["portfolio", "benchmark"]
    active = combined["portfolio"] - combined["benchmark"]
    expected_info = np.sqrt(2) * (active.mean() / active.std())

    assert metrics.get("information_ratio") is not None
    assert metrics["information_ratio"] == pytest.approx(expected_info)
