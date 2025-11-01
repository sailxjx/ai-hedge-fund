from __future__ import annotations

import asyncio
import json
import re
from importlib import import_module
from typing import Any, Callable, Dict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app.backend.services.agent_service import create_agent_function
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.graph.state import AgentState
from src.main import start
from src.utils.analysts import ANALYST_CONFIG, get_analyst_nodes
from src.utils.runtime import async_personas_enabled


def extract_base_agent_key(unique_id: str) -> str:
    """
    Extract the base agent key from a unique node ID.

    Args:
        unique_id: The unique node ID with suffix (e.g., "warren_buffett_abc123")

    Returns:
        The base agent key (e.g., "warren_buffett")
    """
    parts = unique_id.split("_")
    if len(parts) >= 2:
        last_part = parts[-1]
        if len(last_part) == 6 and re.match(r"^[a-z0-9]+$", last_part):
            return "_".join(parts[:-1])
    return unique_id


def _resolve_agent_function(func: Callable[..., Any], async_mode: bool) -> Callable[..., Any]:
    """Return the async-aware variant of the provided agent function."""
    if not async_mode:
        return func
    module = import_module(func.__module__)
    async_candidate = getattr(module, f"{func.__name__}_async", None)
    return async_candidate or func


def _get_analyst_function_map(async_mode: bool) -> Dict[str, Callable[..., Any]]:
    """Build an analyst function mapping that respects async toggles."""
    analyst_nodes = get_analyst_nodes(async_enabled=async_mode)
    return {key: node_func for key, (_, node_func) in analyst_nodes.items()}


def _build_invocation_payload(
    *,
    portfolio: dict,
    tickers: list[str],
    start_date: str,
    end_date: str,
    model_name: str,
    model_provider: str,
    request=None,
) -> AgentState:
    metadata: Dict[str, Any] = {
        "show_reasoning": False,
        "model_name": model_name,
        "model_provider": model_provider,
    }
    if request is not None:
        metadata["request"] = request

    return {
        "messages": [
            HumanMessage(
                content="Make trading decisions based on the provided data.",
            )
        ],
        "data": {
            "tickers": tickers,
            "portfolio": portfolio,
            "start_date": start_date,
            "end_date": end_date,
            "analyst_signals": {},
        },
        "metadata": metadata,
    }


def create_graph(graph_nodes: list, graph_edges: list) -> StateGraph:
    """Create the workflow based on the React Flow graph structure."""
    async_mode = async_personas_enabled()
    graph = StateGraph(AgentState)
    graph.add_node("start_node", start)

    analyst_functions = _get_analyst_function_map(async_mode)

    agent_ids = [node.id for node in graph_nodes]
    agent_ids_set = set(agent_ids)

    portfolio_manager_nodes = set()

    for unique_agent_id in agent_ids:
        base_agent_key = extract_base_agent_key(unique_agent_id)

        if base_agent_key == "portfolio_manager":
            portfolio_manager_nodes.add(unique_agent_id)
            continue

        if base_agent_key not in ANALYST_CONFIG:
            continue

        node_func = analyst_functions.get(base_agent_key)
        if node_func is None:
            raw_config = ANALYST_CONFIG[base_agent_key]
            node_func = _resolve_agent_function(raw_config["agent_func"], async_mode)
        agent_function = create_agent_function(node_func, unique_agent_id)
        graph.add_node(unique_agent_id, agent_function)

    portfolio_manager_base = _resolve_agent_function(portfolio_management_agent, async_mode)
    risk_manager_base = _resolve_agent_function(risk_management_agent, async_mode)
    risk_manager_nodes: Dict[str, str] = {}
    for portfolio_manager_id in portfolio_manager_nodes:
        portfolio_manager_function = create_agent_function(portfolio_manager_base, portfolio_manager_id)
        graph.add_node(portfolio_manager_id, portfolio_manager_function)

        suffix = portfolio_manager_id.split("_")[-1]
        risk_manager_id = f"risk_management_agent_{suffix}"
        risk_manager_nodes[portfolio_manager_id] = risk_manager_id

        risk_manager_function = create_agent_function(risk_manager_base, risk_manager_id)
        graph.add_node(risk_manager_id, risk_manager_function)

    nodes_with_incoming_edges = set()
    direct_to_portfolio_managers: Dict[str, str] = {}

    for edge in graph_edges:
        if edge.source in agent_ids_set and edge.target in agent_ids_set:
            source_base_key = extract_base_agent_key(edge.source)
            target_base_key = extract_base_agent_key(edge.target)

            nodes_with_incoming_edges.add(edge.target)

            if source_base_key in ANALYST_CONFIG and source_base_key != "portfolio_manager" and target_base_key == "portfolio_manager":
                direct_to_portfolio_managers[edge.source] = edge.target
            else:
                graph.add_edge(edge.source, edge.target)

    for agent_id in agent_ids:
        if agent_id not in nodes_with_incoming_edges:
            base_agent_key = extract_base_agent_key(agent_id)
            if base_agent_key in ANALYST_CONFIG and base_agent_key != "portfolio_manager":
                graph.add_edge("start_node", agent_id)

    for analyst_id, portfolio_manager_id in direct_to_portfolio_managers.items():
        risk_manager_id = risk_manager_nodes[portfolio_manager_id]
        graph.add_edge(analyst_id, risk_manager_id)

    for portfolio_manager_id, risk_manager_id in risk_manager_nodes.items():
        graph.add_edge(risk_manager_id, portfolio_manager_id)

    for portfolio_manager_id in portfolio_manager_nodes:
        graph.add_edge(portfolio_manager_id, END)

    graph.set_entry_point("start_node")
    return graph


async def run_graph_async(graph, portfolio, tickers, start_date, end_date, model_name, model_provider, request=None):
    """Async wrapper that respects async persona toggles for backend execution."""
    payload = _build_invocation_payload(
        portfolio=portfolio,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        model_name=model_name,
        model_provider=model_provider,
        request=request,
    )
    if async_personas_enabled():
        return await graph.ainvoke(payload)
    return await asyncio.to_thread(graph.invoke, payload)


def run_graph(
    graph: StateGraph,
    portfolio: dict,
    tickers: list[str],
    start_date: str,
    end_date: str,
    model_name: str,
    model_provider: str,
    request=None,
) -> dict:
    """
    Run the graph with the given portfolio, tickers,
    start date, end date, show reasoning, model name,
    and model provider.
    """
    payload = _build_invocation_payload(
        portfolio=portfolio,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        model_name=model_name,
        model_provider=model_provider,
        request=request,
    )
    return graph.invoke(payload)


def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None
