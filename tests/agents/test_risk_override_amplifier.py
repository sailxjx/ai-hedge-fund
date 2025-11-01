from src.agents.risk_override_amplifier import risk_override_amplifier_agent


def _base_state(analyst_payload: dict, *, long_shares: int = 0, short_shares: int = 0):
    return {
        "messages": [],
        "data": {
            "tickers": ["TSLA"],
            "portfolio": {
                "cash": 100_000.0,
                "positions": {
                    "TSLA": {
                        "long": long_shares,
                        "short": short_shares,
                        "long_cost_basis": 0.0,
                        "short_cost_basis": 0.0,
                    }
                },
            },
            "analyst_signals": {
                "risk_management_agent": {
                    "TSLA": analyst_payload,
                }
            },
        },
        "metadata": {"show_reasoning": False},
    }


def test_neutral_when_no_overrides():
    state = _base_state({"overrides": {}}, long_shares=10, short_shares=5)

    result = risk_override_amplifier_agent(state)
    payload = result["data"]["analyst_signals"]["risk_override_amplifier"]["TSLA"]

    assert payload["signal"] == "risk_neutral"
    assert payload["confidence"] == 55


def test_force_cover_takes_priority():
    overrides = {
        "force_cover_qty": 120,
        "force_cover_reason": "Stop-loss breach",
        "target_short_shares": 0,
    }
    state = _base_state({"overrides": overrides}, short_shares=200)

    result = risk_override_amplifier_agent(state)
    payload = result["data"]["analyst_signals"]["risk_override_amplifier"]["TSLA"]

    assert payload["signal"] == "risk_force_cover"
    assert payload["confidence"] == 100
    directives = payload.get("directives", {})
    assert directives.get("action") == "cover"
    assert directives.get("target_quantity") == 120


def test_prefer_short_overrides_conflicting_long_target():
    overrides = {
        "preferred_direction": "short",
        "target_long_shares": 400,
    }
    state = _base_state({"overrides": overrides}, long_shares=350, short_shares=0)

    result = risk_override_amplifier_agent(state)
    payload = result["data"]["analyst_signals"]["risk_override_amplifier"]["TSLA"]

    assert payload["signal"] == "risk_prefer_short"
    assert payload["confidence"] >= 90
    directives = payload.get("directives", {})
    assert directives.get("preferred_direction") == "short"


def test_short_bias_guardrail_when_additions_blocked():
    overrides = {
        "preferred_direction": "short",
        "block_new_shorts": True,
        "max_additional_short_shares": 0,
        "target_short_shares": 6,
    }
    state = _base_state({"overrides": overrides}, long_shares=0, short_shares=0)

    result = risk_override_amplifier_agent(state)
    payload = result["data"]["analyst_signals"]["risk_override_amplifier"]["TSLA"]

    assert payload["signal"] == "risk_guardrail"
    assert payload["confidence"] >= 80
    directives = payload.get("directives", {})
    assert directives.get("block_new_shorts") is True


def test_target_long_builds_position_when_allowed():
    overrides = {
        "preferred_direction": "long",
        "target_long_shares": 200,
    }
    state = _base_state({"overrides": overrides}, long_shares=50, short_shares=0)

    result = risk_override_amplifier_agent(state)
    payload = result["data"]["analyst_signals"]["risk_override_amplifier"]["TSLA"]

    assert payload["signal"] == "risk_build_long"
    assert payload["confidence"] >= 90
    directives = payload.get("directives", {})
    assert directives.get("action") == "buy"
    assert directives.get("target_quantity") == 150
