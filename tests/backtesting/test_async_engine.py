from __future__ import annotations

import asyncio

import pandas as pd

from src.backtesting.controller import AgentController
from src.backtesting.engine import BacktestEngine


def test_backtest_engine_async_smoke(monkeypatch):
    monkeypatch.setenv("ASYNC_PERSONAS", "1")

    async def fake_run_agent_async(self, agent, **kwargs):
        await asyncio.sleep(0.01)
        tickers = kwargs["tickers"]
        decisions = {ticker: {"action": "hold", "quantity": 0} for ticker in tickers}
        return {
            "decisions": decisions,
            "analyst_signals": {"test_agent": {ticker: {"signal": "neutral"} for ticker in tickers}},
            "risk_manager_state": {"updated": True},
            "timings": {"test_agent": {"elapsed_seconds": 0.01, "updates": 2}},
        }

    def fake_run_agent(self, agent, **kwargs):
        tickers = kwargs["tickers"]
        return {
            "decisions": {ticker: {"action": "hold", "quantity": 0} for ticker in tickers},
            "analyst_signals": {},
        }

    monkeypatch.setattr(AgentController, "run_agent_async", fake_run_agent_async)
    monkeypatch.setattr(AgentController, "run_agent", fake_run_agent)
    monkeypatch.setattr("src.backtesting.engine.BacktestEngine._prefetch_data", lambda self: None)

    price_df = pd.DataFrame({"close": [100.0, 101.0]}, index=pd.date_range("2024-01-02", periods=2, freq="B"))
    monkeypatch.setattr("src.tools.api.get_price_data", lambda ticker, start, end: price_df.copy())
    monkeypatch.setattr("src.tools.api.get_prices", lambda ticker, start_date, end_date: [])
    monkeypatch.setattr("src.tools.api.get_financial_metrics", lambda *args, **kwargs: [])
    monkeypatch.setattr("src.tools.api.get_insider_trades", lambda *args, **kwargs: [])
    monkeypatch.setattr("src.tools.api.get_company_news", lambda *args, **kwargs: [])

    engine = BacktestEngine(
        agent=lambda **kwargs: {},
        tickers=["TEST"],
        start_date="2024-01-02",
        end_date="2024-01-03",
        initial_capital=1_000_000,
        model_name="test-model",
        model_provider="azure",
        selected_analysts=None,
        initial_margin_requirement=0.5,
    )

    metrics = engine.run_backtest()
    assert metrics["turnover_rate"] == 0.0
    assert engine.get_portfolio_values()
    summary = engine.get_timing_summary()
    assert summary is not None
    timing_path = engine.get_timing_log_path()
    assert timing_path.exists()
