from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd
from dateutil.relativedelta import relativedelta

from src.tools.api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_price_data,
    get_prices,
)
from src.utils.llm import LLM_ASYNC_MAX_CONCURRENCY
from src.utils.runtime import async_personas_enabled

from .benchmarks import BenchmarkCalculator
from .controller import AgentController
from .data_types import PerformanceMetrics, PortfolioSnapshot, PortfolioValuePoint
from .metrics import PerformanceMetricsCalculator
from .output import OutputBuilder
from .portfolio import Portfolio
from .trader import TradeExecutor
from .valuation import calculate_portfolio_value, compute_exposures


def _parse_max_days_from_env(env_var: str, default: int | None) -> int | None:
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else None


class BacktestEngine:
    """Coordinates the backtest loop using the new components.

    This implementation mirrors the semantics of src/backtester.py while
    avoiding any changes to that file. It orchestrates agent decisions,
    trade execution, valuation, exposures and performance metrics.
    """

    def __init__(
        self,
        *,
        agent,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        model_name: str,
        model_provider: str,
        selected_analysts: list[str] | None,
        initial_margin_requirement: float,
        portfolio_seed: PortfolioSnapshot | None = None,
        initial_long_pct: float = 0.0,
    ) -> None:
        self._agent = agent
        self._tickers = tickers
        self._start_date = start_date
        self._end_date = end_date
        self._initial_capital = float(initial_capital)
        self._model_name = model_name
        self._model_provider = model_provider
        self._selected_analysts = selected_analysts
        self._portfolio_seed = portfolio_seed
        self._initial_long_pct = max(0.0, min(initial_long_pct, 100.0))
        self._initial_long_applied = False

        self._portfolio = Portfolio(
            tickers=tickers,
            initial_cash=initial_capital,
            margin_requirement=initial_margin_requirement,
            initial_snapshot=portfolio_seed,
        )
        self._executor = TradeExecutor()
        self._agent_controller = AgentController()
        self._perf = PerformanceMetricsCalculator()
        self._results = OutputBuilder(initial_capital=self._initial_capital)

        # Benchmark calculator
        self._benchmark = BenchmarkCalculator()
        self._benchmark_ticker: str | None = "SPY"
        self._benchmark_series: pd.Series | None = None
        self._benchmark_values: list[dict[str, float | None]] = []

        self._portfolio_values: list[PortfolioValuePoint] = []
        self._table_rows: list[list] = []
        self._performance_metrics: PerformanceMetrics = {
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": None,
            "long_short_ratio": None,
            "gross_exposure": None,
            "net_exposure": None,
            "turnover_rate": None,
        }
        # Default to no cap so backtests honor requested ranges unless overridden
        self._max_backtest_days = _parse_max_days_from_env("BACKTEST_MAX_DAYS", None)
        self._base_metadata_overrides = self._build_metadata_overrides()
        self._risk_manager_state: dict[str, Any] | None = None
        self._turnover_notional = 0.0
        self._turnover_value_sum = 0.0
        self._timing_totals: dict[str, float] = defaultdict(float)
        self._timing_counts: dict[str, int] = defaultdict(int)
        self._agent_timings: dict[str, list[float]] = defaultdict(list)
        self._agent_updates: dict[str, int] = defaultdict(int)
        self._async_enabled = async_personas_enabled()
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._run_label = self._base_metadata_overrides.get("run_label", "backtest_run")
        timing_dir = Path("log/backtest_timings")
        timing_dir.mkdir(parents=True, exist_ok=True)
        self._timing_log_path = timing_dir / f"{self._run_label}.json"
        self._timing_summary: dict[str, Any] | None = None

    def _build_metadata_overrides(self) -> dict[str, str | bool]:
        tickers_slug = "_".join(self._tickers) if self._tickers else "portfolio"
        window_slug = f"{self._start_date}_to_{self._end_date}".replace("-", "")
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_label = f"backtest_{tickers_slug}_{window_slug}_{timestamp_slug}"
        log_path = Path("log/risk_overrides") / f"{run_label}.jsonl"
        overrides = {
            "enable_override_logging": True,
            "run_label": run_label,
            "risk_override_log_path": str(log_path),
        }
        if self._portfolio_seed:
            overrides["portfolio_seeded"] = True
        return overrides

    @contextmanager
    def _timed(self, label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._timing_totals[label] += elapsed
            self._timing_counts[label] += 1

    def _prefetch_data(self) -> None:
        end_date_dt = datetime.strptime(self._end_date, "%Y-%m-%d")
        start_date_dt = end_date_dt - relativedelta(years=1)
        start_date_str = start_date_dt.strftime("%Y-%m-%d")

        for ticker in self._tickers:
            get_prices(ticker, start_date_str, self._end_date)
            get_financial_metrics(ticker, self._end_date, limit=10)
            get_insider_trades(ticker, self._end_date, start_date=self._start_date, limit=1000)
            get_company_news(ticker, self._end_date, start_date=self._start_date, limit=1000)

        # Preload data for SPY for benchmark comparison when available without an API key
        if self._benchmark_ticker:
            try:
                bench_df = get_price_data(self._benchmark_ticker, self._start_date, self._end_date)
                if bench_df.empty:
                    raise ValueError("Benchmark data unavailable")
                bench_df = bench_df.sort_index()
                bench_df.index = pd.to_datetime(bench_df.index).tz_localize(None)
                self._benchmark_series = bench_df["close"].astype(float)
            except Exception as exc:  # pragma: no cover - graceful degradation path
                error_message = str(exc)
                if "Missing API key" in error_message or "available" in error_message:
                    print(f"Skipping benchmark {self._benchmark_ticker} because the data vendor requires an API key.")
                else:
                    print(f"Skipping benchmark {self._benchmark_ticker} due to data fetch error: {error_message}")
                self._benchmark_ticker = None
                self._benchmark_series = None

    def _apply_initial_positions(self, first_trading_day: pd.Timestamp | None) -> None:
        """Seed an initial long allocation when requested."""

        if self._initial_long_applied:
            return
        if self._portfolio_seed:
            self._initial_long_applied = True
            return
        if self._initial_long_pct <= 0.0:
            self._initial_long_applied = True
            return
        if not self._tickers or first_trading_day is None:
            self._initial_long_applied = True
            return

        available_cash = self._portfolio.get_cash()
        if available_cash <= 0.0:
            self._initial_long_applied = True
            return

        target_cash = available_cash * (self._initial_long_pct / 100.0)
        if target_cash <= 0.0:
            self._initial_long_applied = True
            return

        remaining_cash = target_cash
        first_day_str = first_trading_day.strftime("%Y-%m-%d")
        total_tickers = len(self._tickers)

        for index, ticker in enumerate(self._tickers):
            if remaining_cash <= 0.0:
                break

            tickers_left = total_tickers - index
            if tickers_left <= 0:
                break

            desired_cash = remaining_cash / tickers_left

            price_df = get_price_data(ticker, self._start_date, first_day_str)
            if price_df.empty:
                continue

            try:
                price_slice = price_df.loc[:first_day_str]
                if price_slice.empty:
                    continue
                price = float(price_slice.iloc[-1]["close"])
            except (KeyError, IndexError):
                continue

            if price <= 0.0:
                continue

            desired_shares = int(desired_cash // price)
            if desired_shares <= 0:
                continue

            executed = self._portfolio.apply_long_buy(ticker, desired_shares, price)
            if executed > 0:
                remaining_cash -= executed * price

        self._initial_long_applied = True

    def _invoke_trading_agent(
        self,
        *,
        tickers: Sequence[str],
        start_date: str,
        end_date: str,
        metadata_overrides: dict[str, Any] | None,
    ) -> AgentOutput:
        controller_kwargs = dict(
            agent=self._agent,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            portfolio=self._portfolio,
            model_name=self._model_name,
            model_provider=self._model_provider,
            selected_analysts=self._selected_analysts,
            metadata_overrides=metadata_overrides,
        )
        if self._async_enabled:
            loop = self._async_loop
            if loop is None:
                loop = asyncio.new_event_loop()
                self._async_loop = loop
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._agent_controller.run_agent_async(**controller_kwargs))
        return self._agent_controller.run_agent(**controller_kwargs)

    def _record_agent_timings(self, timings: dict[str, Any] | None) -> None:
        if not timings:
            return
        for agent_name, payload in timings.items():
            if not isinstance(payload, dict):
                continue
            elapsed = payload.get("elapsed_seconds")
            if elapsed is not None:
                try:
                    elapsed_val = float(elapsed)
                except (TypeError, ValueError):
                    elapsed_val = None
                if elapsed_val is not None:
                    self._agent_timings[agent_name].append(elapsed_val)
            updates = payload.get("updates")
            if updates is not None:
                try:
                    updates_val = int(updates)
                except (TypeError, ValueError):
                    updates_val = 0
                self._agent_updates[agent_name] += max(0, updates_val)

    def run_backtest(self) -> PerformanceMetrics:
        with self._timed("prefetch_data"):
            self._prefetch_data()

        dates = pd.date_range(self._start_date, self._end_date, freq="B")
        if self._max_backtest_days and len(dates) > self._max_backtest_days:
            dates = dates[-self._max_backtest_days :]
        if len(dates) > 0:
            self._portfolio_values = [{"Date": dates[0], "Portfolio Value": self._initial_capital}]
        else:
            self._portfolio_values = []

        if len(dates) > 0 and self._benchmark_series is not None:
            try:
                initial_slice = self._benchmark_series.loc[: dates[0]]
                if not initial_slice.empty:
                    initial_benchmark = float(initial_slice.iloc[-1])
                else:
                    initial_benchmark = None
            except KeyError:
                initial_benchmark = None
            self._benchmark_values = [{"Date": dates[0], "Benchmark Value": initial_benchmark}]
        elif len(dates) > 0:
            self._benchmark_values = [{"Date": dates[0], "Benchmark Value": None}]
        else:
            self._benchmark_values = []

        if len(dates) > 0:
            self._apply_initial_positions(dates[0])

        for current_date in dates:
            lookback_start = (current_date - relativedelta(months=1)).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")
            previous_date_str = (current_date - relativedelta(days=1)).strftime("%Y-%m-%d")
            if lookback_start == current_date_str:
                continue

            try:
                current_prices: Dict[str, float] = {}
                missing_data = False
                for ticker in self._tickers:
                    try:
                        price_data = get_price_data(ticker, previous_date_str, current_date_str)
                        if price_data.empty:
                            missing_data = True
                            break
                        current_prices[ticker] = float(price_data.iloc[-1]["close"])
                    except Exception:
                        missing_data = True
                        break
                if missing_data:
                    continue
            except Exception:
                continue

            metadata_overrides = dict(self._base_metadata_overrides)
            if self._risk_manager_state is not None:
                metadata_overrides["risk_manager_state"] = copy.deepcopy(self._risk_manager_state)

            with self._timed("agent_invoke"):
                agent_output = self._invoke_trading_agent(
                    tickers=self._tickers,
                    start_date=lookback_start,
                    end_date=current_date_str,
                    metadata_overrides=metadata_overrides,
                )
            self._record_agent_timings(agent_output.get("timings"))
            self._risk_manager_state = copy.deepcopy(agent_output.get("risk_manager_state"))
            decisions = agent_output["decisions"]

            executed_trades: Dict[str, int] = {}
            with self._timed("execute_trades"):
                for ticker in self._tickers:
                    d = decisions.get(ticker, {"action": "hold", "quantity": 0})
                    action = d.get("action", "hold")
                    qty = d.get("quantity", 0)
                    executed_qty = self._executor.execute_trade(
                        ticker,
                        action,
                        qty,
                        current_prices[ticker],
                        self._portfolio,
                    )
                    executed_trades[ticker] = executed_qty
            day_turnover_value = 0.0
            for ticker in self._tickers:
                day_turnover_value += abs(executed_trades.get(ticker, 0)) * current_prices[ticker]
            self._turnover_notional += day_turnover_value

            with self._timed("valuation"):
                total_value = calculate_portfolio_value(self._portfolio, current_prices)
                exposures = compute_exposures(self._portfolio, current_prices)
            if total_value > 0:
                self._turnover_value_sum += total_value
            if self._turnover_value_sum > 0:
                self._performance_metrics["turnover_rate"] = self._turnover_notional / self._turnover_value_sum
            else:
                self._performance_metrics["turnover_rate"] = None

            benchmark_close = None
            if self._benchmark_series is not None:
                try:
                    benchmark_slice = self._benchmark_series.loc[:current_date]
                    if not benchmark_slice.empty:
                        benchmark_close = float(benchmark_slice.iloc[-1])
                except KeyError:
                    benchmark_close = None
            self._benchmark_values.append({"Date": current_date, "Benchmark Value": benchmark_close})

            if self._benchmark_series is not None:
                bench_slice = self._benchmark_series.loc[:current_date]
                if not bench_slice.empty:
                    first_close = bench_slice.iloc[0]
                    last_close = bench_slice.iloc[-1]
                    benchmark_return_pct = (float(last_close) / float(first_close) - 1.0) * 100.0 if first_close not in (None, 0) else None
                else:
                    benchmark_return_pct = None
            elif self._benchmark_ticker:
                benchmark_return_pct = self._benchmark.get_return_pct(self._benchmark_ticker, self._start_date, current_date_str)
            else:
                benchmark_return_pct = None

            point: PortfolioValuePoint = {
                "Date": current_date,
                "Portfolio Value": total_value,
                "Long Exposure": exposures["Long Exposure"],
                "Short Exposure": exposures["Short Exposure"],
                "Gross Exposure": exposures["Gross Exposure"],
                "Net Exposure": exposures["Net Exposure"],
                "Long/Short Ratio": exposures["Long/Short Ratio"],
            }
            self._portfolio_values.append(point)

            # Build daily rows (stateless usage)
            with self._timed("build_rows"):
                rows = self._results.build_day_rows(
                    date_str=current_date_str,
                    tickers=self._tickers,
                    agent_output=agent_output,
                    executed_trades=executed_trades,
                    current_prices=current_prices,
                    portfolio=self._portfolio,
                    performance_metrics=self._performance_metrics,
                    total_value=total_value,
                    benchmark_return_pct=benchmark_return_pct,
                )
            # Prepend today's rows to historical rows so latest day is on top
            self._table_rows = rows + self._table_rows
            # Print full history with latest day first (matches backtester.py behavior)
            with self._timed("print_rows"):
                self._results.print_rows(self._table_rows)

            # Update performance metrics after printing (match original timing)
            if len(self._portfolio_values) > 3:
                with self._timed("compute_metrics"):
                    computed = self._perf.compute_metrics(self._portfolio_values, self._benchmark_values)
                if computed:
                    self._performance_metrics.update(computed)

        if self._turnover_value_sum > 0:
            self._performance_metrics["turnover_rate"] = self._turnover_notional / self._turnover_value_sum
        else:
            self._performance_metrics["turnover_rate"] = 0.0
        if self._async_loop is not None:
            self._async_loop.close()
            asyncio.set_event_loop(None)
            self._async_loop = None
        self._write_timing_summary()

        return self._performance_metrics

    def get_portfolio_values(self) -> Sequence[PortfolioValuePoint]:
        return list(self._portfolio_values)

    def get_timing_log_path(self) -> Path:
        """Return the path where timing summaries are persisted."""

        return self._timing_log_path

    def get_timing_summary(self) -> dict[str, Any] | None:
        """Return an in-memory copy of the most recent timing summary."""

        if self._timing_summary is not None:
            return copy.deepcopy(self._timing_summary)
        if self._timing_log_path.exists():
            try:
                payload = json.loads(self._timing_log_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
            if isinstance(payload, dict):
                return payload
        return None

    def _async_meta_from_summary(self, payload: Mapping[str, Any]) -> dict[str, float | int | None]:
        """Compute derived async concurrency metrics from a timing summary."""

        per_agent = payload.get("per_agent")
        if not isinstance(per_agent, Mapping):
            return {}
        agent_total = 0.0
        slowest_avg = 0.0
        slowest_agent = None
        for agent_name, stats in per_agent.items():
            if not isinstance(stats, Mapping):
                continue
            total = float(stats.get("total_seconds", 0.0) or 0.0)
            avg = float(stats.get("avg_seconds", 0.0) or 0.0)
            agent_total += max(0.0, total)
            if avg > slowest_avg:
                slowest_avg = avg
                slowest_agent = agent_name

        agent_invoke_summary = payload.get("agent_invoke")
        if isinstance(agent_invoke_summary, Mapping):
            agent_invoke_total = float(agent_invoke_summary.get("total_seconds", 0.0) or 0.0)
        else:
            agent_invoke_total = 0.0

        concurrency_ratio = None
        if agent_invoke_total > 0.0 and agent_total > 0.0:
            concurrency_ratio = agent_total / agent_invoke_total

        semaphore_limit = max(int(LLM_ASYNC_MAX_CONCURRENCY or 0), 1)
        semaphore_utilization = None
        if concurrency_ratio is not None and semaphore_limit > 0:
            semaphore_utilization = min(concurrency_ratio / semaphore_limit, 1.0)

        return {
            "agent_invoke_total_seconds": agent_invoke_total or None,
            "per_agent_total_seconds": agent_total or None,
            "aggregate_concurrency": concurrency_ratio,
            "slowest_agent": slowest_agent,
            "slowest_agent_avg_seconds": slowest_avg or None,
            "concurrency_limit": semaphore_limit,
            "semaphore_utilization": semaphore_utilization,
        }

    def _write_timing_summary(self) -> None:
        if not self._timing_totals:
            return
        summary = {}
        for label, total in self._timing_totals.items():
            count = self._timing_counts.get(label, 0)
            avg = total / count if count else 0.0
            summary[label] = {
                "total_seconds": total,
                "count": count,
                "avg_seconds": avg,
            }
        if self._agent_timings:
            per_agent: dict[str, dict[str, Any]] = {}
            for agent_name, durations in self._agent_timings.items():
                total = sum(durations)
                count = len(durations)
                avg = total / count if count else 0.0
                per_agent[agent_name] = {
                    "total_seconds": total,
                    "count": count,
                    "avg_seconds": avg,
                    "updates": self._agent_updates.get(agent_name, 0),
                }
            summary["per_agent"] = per_agent
        summary["_meta"] = {
            "turnover_notional": self._turnover_notional,
            "turnover_value_sum": self._turnover_value_sum,
        }
        async_meta = self._async_meta_from_summary(summary)
        if async_meta:
            summary["async_meta"] = async_meta
        self._timing_summary = summary
        try:
            self._timing_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._timing_log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except OSError:
            pass
