from __future__ import annotations

import asyncio
import json

from langchain_core.messages import HumanMessage

from app.backend.models.schemas import GraphEdge, GraphNode
from app.backend.services import graph as graph_service
from src.graph.state import AgentState


def _build_portfolio():
    return {
        "cash": 1_000_000.0,
        "margin_requirement": 0.5,
        "margin_used": 0.0,
        "positions": {"TEST": {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "short_margin_used": 0.0}},
        "realized_gains": {
            "TEST": {"long": 0.0, "short": 0.0},
        },
    }


def _graph_nodes():
    return [
        GraphNode(id="sentiment_analyst"),
        GraphNode(id="portfolio_manager_pm0001"),
    ]


def _graph_edges():
    return [
        GraphEdge(id="edge-sentiment-risk", source="sentiment_analyst", target="portfolio_manager_pm0001"),
    ]


def _state_from_payload(payload: AgentState, *, messages=None, data=None, metadata=None) -> AgentState:
    new_messages = messages if messages is not None else list(payload["messages"])
    new_data = data if data is not None else dict(payload["data"])
    new_metadata = metadata if metadata is not None else dict(payload["metadata"])
    return {
        "messages": new_messages,
        "data": new_data,
        "metadata": new_metadata,
    }


def test_run_graph_async_with_async_personas(monkeypatch):
    calls: list[str] = []
    monkeypatch.setenv("ASYNC_PERSONAS", "1")

    async def sentiment_async(state: AgentState, agent_id: str = "sentiment_analyst"):
        calls.append(f"{agent_id}:async")
        updated_data = dict(state["data"])
        analyst_signals = dict(updated_data.get("analyst_signals", {}))
        analyst_signals[agent_id] = {"TEST": {"signal": "bullish"}}
        updated_data["analyst_signals"] = analyst_signals
        return _state_from_payload(state, data=updated_data)

    async def risk_async(state: AgentState, agent_id: str = "risk_management_agent_pm0001"):
        calls.append(f"{agent_id}:async")
        updated_data = dict(state["data"])
        updated_data["risk_manager_state"] = {"ok": True}
        return _state_from_payload(state, data=updated_data)

    async def portfolio_async(state: AgentState, agent_id: str = "portfolio_manager_pm0001"):
        calls.append(f"{agent_id}:async")
        decisions = {"TEST": {"action": "hold", "quantity": 0}}
        updated_messages = list(state["messages"]) + [HumanMessage(content=json.dumps(decisions), name=agent_id)]
        return _state_from_payload(state, messages=updated_messages)

    monkeypatch.setattr("src.agents.sentiment.sentiment_analyst_agent_async", sentiment_async)
    monkeypatch.setattr("src.agents.risk_manager.risk_management_agent_async", risk_async)
    monkeypatch.setattr("src.agents.portfolio_manager.portfolio_management_agent_async", portfolio_async)

    graph = graph_service.create_graph(_graph_nodes(), _graph_edges()).compile()

    async def _runner():
        return await graph_service.run_graph_async(
            graph=graph,
            portfolio=_build_portfolio(),
            tickers=["TEST"],
            start_date="2024-01-01",
            end_date="2024-01-05",
            model_name="test-model",
            model_provider="azure",
            request=None,
        )

    result = asyncio.run(_runner())

    assert result["data"]["analyst_signals"]["sentiment_analyst"]["TEST"]["signal"] == "bullish"
    assert calls == [
        "sentiment_analyst:async",
        "risk_management_agent_pm0001:async",
        "portfolio_manager_pm0001:async",
    ]


def test_run_graph_async_with_sync_personas(monkeypatch):
    calls: list[str] = []
    monkeypatch.setenv("ASYNC_PERSONAS", "0")

    def sentiment_sync(state: AgentState, agent_id: str = "sentiment_analyst"):
        calls.append(f"{agent_id}:sync")
        updated_data = dict(state["data"])
        analyst_signals = dict(updated_data.get("analyst_signals", {}))
        analyst_signals[agent_id] = {"TEST": {"signal": "neutral"}}
        updated_data["analyst_signals"] = analyst_signals
        return _state_from_payload(state, data=updated_data)

    def risk_sync(state: AgentState, agent_id: str = "risk_management_agent_pm0001"):
        calls.append(f"{agent_id}:sync")
        return state

    def portfolio_sync(state: AgentState, agent_id: str = "portfolio_manager_pm0001"):
        calls.append(f"{agent_id}:sync")
        decisions = {"TEST": {"action": "hold", "quantity": 0}}
        updated_messages = list(state["messages"]) + [HumanMessage(content=json.dumps(decisions), name=agent_id)]
        return _state_from_payload(state, messages=updated_messages)

    monkeypatch.setattr("src.agents.sentiment.sentiment_analyst_agent", sentiment_sync)
    monkeypatch.setattr("src.agents.risk_manager.risk_management_agent", risk_sync)
    monkeypatch.setattr("src.agents.portfolio_manager.portfolio_management_agent", portfolio_sync)
    monkeypatch.setattr(graph_service, "risk_management_agent", risk_sync)
    monkeypatch.setattr(graph_service, "portfolio_management_agent", portfolio_sync)
    monkeypatch.setitem(graph_service.ANALYST_CONFIG["sentiment_analyst"], "agent_func", sentiment_sync)

    graph = graph_service.create_graph(_graph_nodes(), _graph_edges()).compile()

    async def _runner():
        return await graph_service.run_graph_async(
            graph=graph,
            portfolio=_build_portfolio(),
            tickers=["TEST"],
            start_date="2024-01-01",
            end_date="2024-01-05",
            model_name="test-model",
            model_provider="azure",
            request=None,
        )

    result = asyncio.run(_runner())

    assert result["data"]["analyst_signals"]["sentiment_analyst"]["TEST"]["signal"] == "neutral"
    assert calls == [
        "sentiment_analyst:sync",
        "risk_management_agent_pm0001:sync",
        "portfolio_manager_pm0001:sync",
    ]
