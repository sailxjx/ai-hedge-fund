from src.backtesting.engine import BacktestEngine
from src.backtesting.portfolio import Portfolio


def test_portfolio_applies_short_seed():
    seed = {
        "cash": 95_000.0,
        "margin_used": 3_000.0,
        "margin_requirement": 0.5,
        "positions": {
            "TSLA": {
                "long": 0,
                "short": 6,
                "long_cost_basis": 0.0,
                "short_cost_basis": 310.0,
                "short_margin_used": 3_000.0,
            }
        },
        "realized_gains": {"TSLA": {"long": 0.0, "short": 100.0}},
    }

    portfolio = Portfolio(
        tickers=["TSLA"],
        initial_cash=100_000.0,
        margin_requirement=0.5,
        initial_snapshot=seed,
    )

    snapshot = portfolio.get_snapshot()
    assert snapshot["cash"] == seed["cash"]
    assert snapshot["margin_used"] == seed["margin_used"]
    assert snapshot["positions"]["TSLA"]["short"] == 6
    assert snapshot["positions"]["TSLA"]["short_cost_basis"] == 310.0
    assert snapshot["realized_gains"]["TSLA"]["short"] == 100.0


def test_portfolio_seed_adds_additional_tickers():
    seed = {
        "cash": 50_000.0,
        "margin_used": 0.0,
        "margin_requirement": 0.2,
        "positions": {
            "NVDA": {
                "long": 4,
                "short": 0,
                "long_cost_basis": 120.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
        },
        "realized_gains": {"NVDA": {"long": 25.0, "short": 0.0}},
    }

    portfolio = Portfolio(
        tickers=["TSLA"],
        initial_cash=60_000.0,
        margin_requirement=0.2,
        initial_snapshot=seed,
    )

    snapshot = portfolio.get_snapshot()
    assert "NVDA" in snapshot["positions"]
    assert snapshot["positions"]["NVDA"]["long"] == 4
    assert snapshot["realized_gains"]["NVDA"]["long"] == 25.0
    assert "TSLA" in snapshot["positions"]
    assert snapshot["positions"]["TSLA"]["long"] == 0


def test_backtest_engine_registers_portfolio_seed():
    seed = {
        "cash": 95_000.0,
        "margin_used": 1_500.0,
        "margin_requirement": 0.5,
        "positions": {
            "TSLA": {
                "long": 0,
                "short": 3,
                "long_cost_basis": 0.0,
                "short_cost_basis": 305.0,
                "short_margin_used": 1_500.0,
            }
        },
        "realized_gains": {"TSLA": {"long": 0.0, "short": 0.0}},
    }

    def dummy_agent(**kwargs):
        return {"decisions": {}, "analyst_signals": {}}

    engine = BacktestEngine(
        agent=dummy_agent,
        tickers=["TSLA"],
        start_date="2025-07-01",
        end_date="2025-07-05",
        initial_capital=100_000.0,
        model_name="test-model",
        model_provider="azure",
        selected_analysts=[],
        initial_margin_requirement=0.5,
        portfolio_seed=seed,
    )

    assert engine._base_metadata_overrides.get("portfolio_seeded") is True
