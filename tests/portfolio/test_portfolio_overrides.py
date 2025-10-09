from src.agents.portfolio_manager import (
    PortfolioDecision,
    compute_allowed_actions,
    generate_trading_decision,
)


def _base_portfolio():
    return {
        "cash": 100_000.0,
        "margin_requirement": 0.5,
        "margin_used": 0.0,
        "equity": 100_000.0,
        "positions": {
            "TSLA": {
                "long": 0,
                "long_cost_basis": 0.0,
                "short": 0,
                "short_cost_basis": 0.0,
            }
        },
    }


def test_compute_allowed_actions_blocks_new_shorts():
    portfolio = _base_portfolio()
    portfolio["positions"]["TSLA"]["short"] = 2
    allowed = compute_allowed_actions(
        tickers=["TSLA"],
        current_prices={"TSLA": 250.0},
        max_shares={"TSLA": 10},
        portfolio=portfolio,
        overrides={"TSLA": {"block_new_shorts": True, "max_additional_short_shares": 0}},
    )
    assert "short" not in allowed["TSLA"]
    assert allowed["TSLA"]["cover"] >= 2


def test_compute_allowed_actions_respects_target_short_cap():
    portfolio = _base_portfolio()
    portfolio["positions"]["TSLA"]["short"] = 5
    allowed = compute_allowed_actions(
        tickers=["TSLA"],
        current_prices={"TSLA": 250.0},
        max_shares={"TSLA": 50},
        portfolio=portfolio,
        overrides={"TSLA": {"target_short_shares": 5}},
    )
    assert "short" not in allowed["TSLA"]


def test_compute_allowed_actions_limits_short_to_target_gap():
    portfolio = _base_portfolio()
    portfolio["positions"]["TSLA"]["short"] = 2
    allowed = compute_allowed_actions(
        tickers=["TSLA"],
        current_prices={"TSLA": 250.0},
        max_shares={"TSLA": 50},
        portfolio=portfolio,
        overrides={"TSLA": {"target_short_shares": 5}},
    )
    assert allowed["TSLA"].get("short") == 3


def test_compute_allowed_actions_caps_cover_to_target_gap():
    portfolio = _base_portfolio()
    portfolio["positions"]["TSLA"]["short"] = 10
    allowed = compute_allowed_actions(
        tickers=["TSLA"],
        current_prices={"TSLA": 250.0},
        max_shares={"TSLA": 50},
        portfolio=portfolio,
        overrides={"TSLA": {"target_short_shares": 4}},
    )
    assert allowed["TSLA"].get("cover") == 6


def test_compute_allowed_actions_blocks_cover_when_target_met():
    portfolio = _base_portfolio()
    portfolio["positions"]["TSLA"]["short"] = 7
    allowed = compute_allowed_actions(
        tickers=["TSLA"],
        current_prices={"TSLA": 300.0},
        max_shares={"TSLA": 20},
        portfolio=portfolio,
        overrides={"TSLA": {"target_short_shares": 7}},
    )
    assert "cover" not in allowed["TSLA"] or allowed["TSLA"].get("cover") is None


def test_generate_trading_decision_prefills_force_cover():
    portfolio = _base_portfolio()
    portfolio["positions"]["TSLA"]["short"] = 5
    decisions = generate_trading_decision(
        tickers=["TSLA"],
        signals_by_ticker={"TSLA": {}},
        current_prices={"TSLA": 200.0},
        max_shares={"TSLA": 10},
        portfolio=portfolio,
        overrides={"TSLA": {"force_cover_qty": 3, "force_cover_reason": "Risk override"}},
        agent_id="portfolio_manager",
        state={"messages": [], "data": {}, "metadata": {"show_reasoning": False}},
    )
    decision = decisions.decisions["TSLA"]
    assert isinstance(decision, PortfolioDecision)
    assert decision.action == "cover"
    assert decision.quantity == 3
    assert "Risk override" in decision.reasoning


def test_generate_trading_decision_prefills_target_long_buy():
    portfolio = _base_portfolio()
    portfolio["positions"]["TSLA"]["long"] = 2
    decisions = generate_trading_decision(
        tickers=["TSLA"],
        signals_by_ticker={"TSLA": {}},
        current_prices={"TSLA": 100.0},
        max_shares={"TSLA": 20},
        portfolio=portfolio,
        overrides={"TSLA": {"target_long_shares": 5, "force_buy_reason": "Long bias"}},
        agent_id="portfolio_manager",
        state={"messages": [], "data": {}, "metadata": {"show_reasoning": False}},
    )
    decision = decisions.decisions["TSLA"]
    assert decision.action == "buy"
    assert decision.quantity == 3
    assert "Long bias" in decision.reasoning

def test_generate_trading_decision_prefills_target_short_sell():
    portfolio = _base_portfolio()
    decisions = generate_trading_decision(
        tickers=["TSLA"],
        signals_by_ticker={"TSLA": {}},
        current_prices={"TSLA": 150.0},
        max_shares={"TSLA": 20},
        portfolio=portfolio,
        overrides={
            "TSLA": {
                "target_short_shares": 6,
                "max_additional_short_shares": 6,
                "force_short_reason": "Crash allocator target",
            }
        },
        agent_id="portfolio_manager",
        state={"messages": [], "data": {}, "metadata": {"show_reasoning": False}},
    )
    decision = decisions.decisions["TSLA"]
    assert decision.action == "short"
    assert decision.quantity == 6
    assert "Crash allocator target" in decision.reasoning
