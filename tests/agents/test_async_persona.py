import asyncio

from src.agents.persona_utils import async_persona_from_observations, PersonaDecision


def test_async_persona_from_observations(monkeypatch):
    async def fake_async_call_llm(*, prompt, pydantic_model, agent_name, state, default_factory):
        await asyncio.sleep(0.01)
        return PersonaDecision(signal="bullish", confidence=77.0, reasoning="Async path", constraints={"cap": 0.2})

    monkeypatch.setattr("src.agents.persona_utils.async_call_llm", fake_async_call_llm)
    state = {"data": {}, "metadata": {}}

    decision = asyncio.run(
        async_persona_from_observations(
            state=state,
            agent_id="test_agent",
            persona_name="Test",
            persona_role="tester",
            persona_backstory="Validates async path",
            allowed_signals=["bullish", "bearish", "neutral"],
            observations={"ticker": "TEST"},
        )
    )
    assert decision.signal == "bullish"
    assert decision.confidence == 77.0
    assert decision.constraints == {"cap": 0.2}
