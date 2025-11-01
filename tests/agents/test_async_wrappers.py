import asyncio
import json
from inspect import iscoroutinefunction

from langchain_core.messages import HumanMessage

from src.agents import portfolio_manager, risk_manager, risk_override_amplifier
from src.agents.portfolio_manager import PortfolioDecision, PortfolioManagerOutput
from src.agents.persona_utils import PersonaDecision
from src.utils.analysts import get_analyst_nodes


def _base_state():
    return {
        "messages": [],
        "data": {
            "analyst_signals": {},
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TEST": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
            },
            "tickers": ["TEST"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
        },
        "metadata": {"show_reasoning": False},
    }


def test_risk_management_agent_async(monkeypatch):
    async def run():
        state = _base_state()

        async def fake_impl(clone_state, agent_id="risk_management_agent"):
            payload = {"TEST": {"remaining_position_limit": 1000.0}}
            clone_state.setdefault("data", {}).setdefault("analyst_signals", {})[agent_id] = payload
            clone_state["data"]["risk_manager_state"] = {"crash_mode": {"TEST": {"active": True}}}
            clone_state["messages"] = [HumanMessage(content=json.dumps(payload), name=agent_id)]
            return {"messages": list(clone_state["messages"])}

        monkeypatch.setattr(risk_manager, "_risk_management_agent_impl", fake_impl)

        result = await risk_manager.risk_management_agent_async(state)

        signals = state["data"]["analyst_signals"]["risk_management_agent"]
        assert signals["TEST"]["remaining_position_limit"] == 1000.0
        assert state["data"]["risk_manager_state"]["crash_mode"]["TEST"]["active"] is True
        assert isinstance(result["messages"][-1], HumanMessage)

    asyncio.run(run())


def test_risk_override_amplifier_agent_async(monkeypatch):
    async def run():
        state = _base_state()
        state["data"]["analyst_signals"]["risk_management_agent"] = {"TEST": {"overrides": {"preferred_direction": "long"}}}

        async def fake_async_apply_persona(*, payload, **kwargs):
            payload["signal"] = "risk_guardrail"
            payload["confidence"] = 80
            payload["reasoning"] = "Stub override"
            payload["constraints"] = {}
            return PersonaDecision(signal="risk_guardrail", confidence=80, reasoning="Stub override", constraints={})

        monkeypatch.setattr(risk_override_amplifier, "async_apply_persona", fake_async_apply_persona)
        result = await risk_override_amplifier.risk_override_amplifier_agent_async(state)

        signals = state["data"]["analyst_signals"]["risk_override_amplifier"]
        assert signals["TEST"]["signal"] == "risk_guardrail"
        assert result["messages"][-1].name == "risk_override_amplifier"

    asyncio.run(run())


def test_portfolio_management_agent_async(monkeypatch):
    async def run():
        state = _base_state()
        state["data"]["analyst_signals"]["risk_management_agent"] = {
            "TEST": {
                "remaining_position_limit": 10_000.0,
                "current_price": 10.0,
                "overrides": {},
            }
        }
        state["data"]["analyst_signals"]["growth_momentum_agent"] = {
            "TEST": {"signal": "buy", "confidence": 85}
        }

        async def fake_async_generate(**kwargs):
            return PortfolioManagerOutput(
                decisions={"TEST": PortfolioDecision(action="buy", quantity=10, confidence=90, reasoning="Async stub")}
            )

        monkeypatch.setattr(portfolio_manager, "async_generate_trading_decision", fake_async_generate)

        result = await portfolio_manager.portfolio_management_agent_async(state)

        signals = state["data"]["analyst_signals"]["portfolio_manager"]
        assert signals["TEST"]["action"] == "buy"
        assert signals["TEST"]["quantity"] == 10
        assert state["data"]["current_prices"]["TEST"] == 10.0
        assert result["messages"][-1].name == "portfolio_manager"

    asyncio.run(run())


def test_all_analysts_register_async_variants(monkeypatch):
    """Ensure every analyst exposes an async callable when async personas are enabled."""

    monkeypatch.setenv("ASYNC_PERSONAS", "1")
    nodes = get_analyst_nodes(async_enabled=True)
    missing = [key for key, (_, func) in nodes.items() if not iscoroutinefunction(func)]
    assert not missing, f"Missing async variants for analysts: {missing}"
