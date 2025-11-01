from __future__ import annotations

import copy

import pandas as pd

from src.backtesting.engine import BacktestEngine


def _stub_prices(*_args, **_kwargs):
    return None


def _stub_metrics(*_args, **_kwargs):
    return []


class TrackingAgent:
    def __init__(self) -> None:
        self.calls: list[dict | None] = []

    def __call__(
        self,
        *,
        tickers,
        start_date,
        end_date,
        portfolio,
        model_name,
        model_provider,
        selected_analysts,
        metadata_overrides,
    ):
        self.calls.append(copy.deepcopy(metadata_overrides.get("risk_manager_state")))
        decisions = {ticker: {"action": "hold", "quantity": 0} for ticker in tickers}
        streak = len(self.calls)
        return {
            "decisions": decisions,
            "analyst_signals": {},
            "risk_manager_state": {"crash_mode": {tickers[0]: {"streak": streak}}},
        }


def test_backtest_engine_persists_risk_manager_state(monkeypatch):
    monkeypatch.setattr("src.backtesting.engine.get_prices", _stub_prices)
    monkeypatch.setattr("src.backtesting.engine.get_financial_metrics", _stub_metrics)
    monkeypatch.setattr("src.backtesting.engine.get_insider_trades", _stub_metrics)
    monkeypatch.setattr("src.backtesting.engine.get_company_news", _stub_metrics)

    def fake_price_data(ticker: str, start: str, end: str, api_key=None):
        index = pd.date_range(start, end, freq="B")
        if index.empty:
            return pd.DataFrame(columns=["close"])
        closes = [100.0 + i for i in range(len(index))]
        return pd.DataFrame({"close": closes}, index=index)

    monkeypatch.setattr("src.backtesting.engine.get_price_data", fake_price_data)

    agent = TrackingAgent()

    engine = BacktestEngine(
        agent=agent,
        tickers=["TSLA"],
        start_date="2024-04-01",
        end_date="2024-04-05",
        initial_capital=100_000.0,
        model_name="test-model",
        model_provider="test-provider",
        selected_analysts=[],
        initial_margin_requirement=0.0,
    )

    engine.run_backtest()

    assert len(agent.calls) >= 2
    assert agent.calls[0] is None
    # The second call should receive the first state's streak of 1
    assert agent.calls[1] == {"crash_mode": {"TSLA": {"streak": 1}}}
