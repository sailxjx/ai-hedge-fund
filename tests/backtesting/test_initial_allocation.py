from __future__ import annotations

import pandas as pd

from src.backtesting.engine import BacktestEngine


def _hold_agent(**kwargs):
    tickers = kwargs.get("tickers", [])
    decisions = {ticker: {"action": "hold", "quantity": 0} for ticker in tickers}
    return {"decisions": decisions, "analyst_signals": {}}


def test_initial_long_allocation(monkeypatch):
    """Ensure initial_long_pct seeds a starting long position."""

    monkeypatch.setattr("src.backtesting.engine.get_prices", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.backtesting.engine.get_financial_metrics", lambda *args, **kwargs: [])
    monkeypatch.setattr("src.backtesting.engine.get_insider_trades", lambda *args, **kwargs: [])
    monkeypatch.setattr("src.backtesting.engine.get_company_news", lambda *args, **kwargs: [])

    def fake_price_data(ticker: str, start: str, end: str, api_key=None):
        idx = pd.date_range(start, end, freq="B")
        if idx.empty:
            return pd.DataFrame(columns=["close"])  # pragma: no cover - defensive
        return pd.DataFrame({"close": [200.0 for _ in idx]}, index=idx)

    monkeypatch.setattr("src.backtesting.engine.get_price_data", fake_price_data)

    engine = BacktestEngine(
        agent=_hold_agent,
        tickers=["TSLA"],
        start_date="2024-04-18",
        end_date="2024-04-22",
        initial_capital=100_000.0,
        model_name="test-model",
        model_provider="test-provider",
        selected_analysts=[],
        initial_margin_requirement=0.0,
        initial_long_pct=100.0,
    )

    engine.run_backtest()

    snapshot = engine._portfolio.get_snapshot()
    tsla_position = snapshot["positions"]["TSLA"]

    assert tsla_position["long"] == 500  # 100_000 / 200
    assert tsla_position["short"] == 0
    assert tsla_position["long_cost_basis"] == 200.0
    assert snapshot["cash"] == 0.0
