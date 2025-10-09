from __future__ import annotations

from typing import Sequence

from .data_types import PerformanceMetrics, PortfolioValuePoint


class PerformanceMetricsCalculator:
    """Concrete metrics calculator like sharpe ratio, sortino ratio, max drawdown, etc."""

    def __init__(self, *, annual_trading_days: int = 252, annual_rf_rate: float = 0.0434) -> None:
        self.annual_trading_days = annual_trading_days
        self.annual_rf_rate = annual_rf_rate

    def update_metrics(
        self,
        metrics: PerformanceMetrics,
        values: Sequence[PortfolioValuePoint],
        benchmark_values: Sequence[dict] | None = None,
    ) -> None:
        """Deprecated: mutate provided dict. Kept for backward compatibility."""
        computed = self.compute_metrics(values, benchmark_values)
        if not computed:
            return
        metrics.update(computed)  # type: ignore[arg-type]

    def compute_metrics(
        self,
        values: Sequence[PortfolioValuePoint],
        benchmark_values: Sequence[dict] | None = None,
    ) -> PerformanceMetrics:
        import pandas as pd
        import numpy as np

        if not values:
            return {
                "sharpe_ratio": None,
                "sortino_ratio": None,
                "max_drawdown": None,
                "max_drawdown_date": None,
                "information_ratio": None,
            }

        df = pd.DataFrame(values)
        if df.empty or "Portfolio Value" not in df:
            return {
                "sharpe_ratio": None,
                "sortino_ratio": None,
                "max_drawdown": None,
                "max_drawdown_date": None,
                "information_ratio": None,
            }

        df = df.set_index("Date")
        df = df[~df.index.duplicated(keep="last")]
        df["Daily Return"] = df["Portfolio Value"].pct_change()
        clean_returns = df["Daily Return"].dropna()
        if len(clean_returns) < 2:
            return {
                "sharpe_ratio": None,
                "sortino_ratio": None,
                "max_drawdown": None,
                "max_drawdown_date": None,
                "information_ratio": None,
            }

        daily_rf = self.annual_rf_rate / self.annual_trading_days
        excess = clean_returns - daily_rf
        mean_excess = excess.mean()
        std_excess = excess.std()

        if std_excess > 1e-12:
            sharpe = float(np.sqrt(self.annual_trading_days) * (mean_excess / std_excess))
        else:
            sharpe = 0.0

        negative_excess = excess[excess < 0]
        if len(negative_excess) > 0:
            downside_std = negative_excess.std()
            if downside_std > 1e-12:
                sortino = float(np.sqrt(self.annual_trading_days) * (mean_excess / downside_std))
            else:
                sortino = float("inf") if mean_excess > 0 else 0.0
        else:
            sortino = float("inf") if mean_excess > 0 else 0.0

        rolling_max = df["Portfolio Value"].cummax()
        drawdown = (df["Portfolio Value"] - rolling_max) / rolling_max
        if len(drawdown) > 0:
            min_dd = float(drawdown.min())
            max_drawdown = float(min_dd * 100.0)
            if min_dd < 0:
                max_drawdown_date = drawdown.idxmin().strftime("%Y-%m-%d")
            else:
                max_drawdown_date = None
        else:
            max_drawdown = 0.0
            max_drawdown_date = None

        information_ratio = None
        if benchmark_values:
            bench_df = pd.DataFrame(benchmark_values)
            if not bench_df.empty and "Benchmark Value" in bench_df:
                bench_df = bench_df.set_index("Date").sort_index()
                bench_df = bench_df[~bench_df.index.duplicated(keep="last")]
                bench_df = bench_df.reindex(df.index).ffill()
                if not bench_df.empty:
                    bench_returns = bench_df["Benchmark Value"].pct_change()
                    combined = pd.concat(
                        [df["Daily Return"], bench_returns], axis=1, join="inner"
                    ).dropna()
                    if not combined.empty:
                        combined.columns = ["portfolio", "benchmark"]
                        active = combined["portfolio"] - combined["benchmark"]
                        tracking_error = active.std()
                        if tracking_error > 1e-12:
                            information_ratio = float(
                                np.sqrt(self.annual_trading_days)
                                * (active.mean() / tracking_error)
                            )

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "max_drawdown_date": max_drawdown_date,
            "information_ratio": information_ratio,
        }
