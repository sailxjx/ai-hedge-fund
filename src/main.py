import asyncio
import copy
import io
import json
import re
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

from colorama import init
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.agents.risk_override_amplifier import risk_override_amplifier_agent
from src.backtesting.portfolio import Portfolio
from src.cli.input import parse_cli_inputs
from src.graph.state import AgentState
from src.utils.analysts import get_analyst_nodes
from src.utils.display import print_trading_output
from src.utils.progress import progress
from src.utils.runtime import async_personas_enabled

# Load environment variables from .env file
load_dotenv()

init(autoreset=True)


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


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    """Remove ANSI color codes from a string."""
    return ANSI_ESCAPE_RE.sub("", text)


def _resolve_agent_function(func, async_mode: bool) -> Any:
    if not async_mode:
        return func
    module = import_module(func.__module__)
    async_candidate = getattr(module, f"{func.__name__}_async", None)
    return async_candidate or func


def _build_metadata(
    *,
    show_reasoning: bool,
    model_name: str,
    model_provider: str,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = {
        "show_reasoning": show_reasoning,
        "model_name": model_name,
        "model_provider": model_provider,
    }
    if overrides:
        metadata.update({k: v for k, v in overrides.items() if v is not None})
    return metadata


def _build_override_logging_metadata(
    *,
    tickers: list[str],
    start_date: str,
    end_date: str,
    log_file: str | None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {"enable_override_logging": True}
    if log_file:
        log_path = Path(log_file).expanduser()
        overrides["log_file"] = str(log_path)
        overrides["run_label"] = log_path.stem or "hedge_fund_run"
        overrides["risk_override_log_path"] = str(log_path.with_suffix(".jsonl"))
    else:
        tickers_slug = "_".join(tickers) if tickers else "portfolio"
        window_slug = f"{start_date}_to_{end_date}".replace("-", "")
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_label = f"cli_{tickers_slug}_{window_slug}_{timestamp_slug}"
        overrides["run_label"] = run_label
        overrides["risk_override_log_path"] = str(Path("log/risk_overrides") / f"{run_label}.jsonl")
    return overrides


# Run the Hedge Fund
def _prepare_agent_run(
    *,
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool,
    selected_analysts: list[str] | None,
    model_name: str,
    model_provider: str,
    metadata_overrides: dict[str, Any] | None,
    async_mode: bool,
):
    overrides_for_metadata: dict[str, Any] | None = None
    persistent_state: dict[str, Any] | None = None
    if metadata_overrides:
        overrides_for_metadata = {key: value for key, value in metadata_overrides.items() if key != "risk_manager_state"}
        if "risk_manager_state" in metadata_overrides:
            persistent_state = copy.deepcopy(metadata_overrides["risk_manager_state"])

    workflow = create_workflow(selected_analysts if selected_analysts else None, async_mode=async_mode)
    agent = workflow.compile()

    data_payload: dict[str, Any] = {
        "tickers": tickers,
        "portfolio": portfolio,
        "start_date": start_date,
        "end_date": end_date,
        "analyst_signals": {},
    }
    if persistent_state:
        data_payload["risk_manager_state"] = persistent_state

    invocation = {
        "messages": [
            HumanMessage(
                content="Make trading decisions based on the provided data.",
            )
        ],
        "data": data_payload,
        "metadata": _build_metadata(
            show_reasoning=show_reasoning,
            model_name=model_name,
            model_provider=model_provider,
            overrides=overrides_for_metadata,
        ),
    }

    return agent, invocation


def _register_timing_handler():
    timings: dict[str, dict[str, Any]] = {}

    def handler(agent_name, ticker, status, analysis, timestamp):
        try:
            observed = datetime.fromisoformat(timestamp)
        except ValueError:
            observed = datetime.now(timezone.utc)
        entry = timings.setdefault(agent_name, {"start": None, "end": None, "updates": 0})
        if entry["start"] is None:
            entry["start"] = observed
        entry["end"] = observed
        entry["updates"] += 1

    progress.register_handler(handler)
    return timings, handler


def _finalize_timings(timings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for agent_name, meta in timings.items():
        start = meta.get("start")
        end = meta.get("end")
        if start and end:
            elapsed = (end - start).total_seconds()
        else:
            elapsed = None
        report[agent_name] = {
            "elapsed_seconds": elapsed,
            "updates": meta.get("updates", 0),
        }
    return report


def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4.1",
    model_provider: str = "OpenAI",
    metadata_overrides: dict[str, Any] | None = None,
):
    async_mode = async_personas_enabled()
    if async_mode:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                run_hedge_fund_async(
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date,
                    portfolio=portfolio,
                    show_reasoning=show_reasoning,
                    selected_analysts=selected_analysts,
                    model_name=model_name,
                    model_provider=model_provider,
                    metadata_overrides=metadata_overrides,
                )
            )
        raise RuntimeError("Async personas are enabled. Await run_hedge_fund_async (or set ASYNC_PERSONAS=0 before invoking run_hedge_fund inside an active event loop).")

    progress.start()
    try:
        agent, invocation = _prepare_agent_run(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=show_reasoning,
            selected_analysts=selected_analysts,
            model_name=model_name,
            model_provider=model_provider,
            metadata_overrides=metadata_overrides,
            async_mode=False,
        )

        timings, handler = _register_timing_handler()
        try:
            final_state = agent.invoke(invocation)
        finally:
            progress.unregister_handler(handler)
        timing_report = _finalize_timings(timings)

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
            "risk_manager_state": copy.deepcopy(final_state["data"].get("risk_manager_state")),
            "timings": timing_report,
        }
    finally:
        progress.stop()


async def run_hedge_fund_async(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] | None = None,
    model_name: str = "gpt-4.1",
    model_provider: str = "OpenAI",
    metadata_overrides: dict[str, Any] | None = None,
):
    progress.start()
    try:
        agent, invocation = _prepare_agent_run(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=show_reasoning,
            selected_analysts=selected_analysts,
            model_name=model_name,
            model_provider=model_provider,
            metadata_overrides=metadata_overrides,
            async_mode=True,
        )
        timings, handler = _register_timing_handler()
        try:
            final_state = await agent.ainvoke(invocation)
        finally:
            progress.unregister_handler(handler)
        timing_report = _finalize_timings(timings)
        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
            "risk_manager_state": copy.deepcopy(final_state["data"].get("risk_manager_state")),
            "timings": timing_report,
        }
    finally:
        progress.stop()


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None, async_mode: bool | None = None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    use_async = async_personas_enabled() if async_mode is None else async_mode
    analyst_nodes = get_analyst_nodes(async_enabled=use_async)

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", _resolve_agent_function(risk_management_agent, use_async))
    workflow.add_node("risk_override_amplifier", _resolve_agent_function(risk_override_amplifier_agent, use_async))
    workflow.add_node("portfolio_manager", _resolve_agent_function(portfolio_management_agent, use_async))

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "risk_override_amplifier")
    workflow.add_edge("risk_override_amplifier", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    workflow.set_entry_point("start_node")
    return workflow


if __name__ == "__main__":
    inputs = parse_cli_inputs(
        description="Run the hedge fund trading system",
        require_tickers=True,
        default_months_back=None,
        include_graph_flag=True,
        include_reasoning_flag=True,
    )

    tickers = inputs.tickers
    selected_analysts = inputs.selected_analysts

    # Construct portfolio here
    if inputs.portfolio_seed:
        seeded = Portfolio(
            tickers=tickers,
            initial_cash=inputs.initial_cash,
            margin_requirement=inputs.margin_requirement,
            initial_snapshot=inputs.portfolio_seed,
        )
        portfolio = seeded.get_snapshot()
    else:
        portfolio = {
            "cash": inputs.initial_cash,
            "margin_requirement": inputs.margin_requirement,
            "margin_used": 0.0,
            "positions": {
                ticker: {
                    "long": 0,
                    "short": 0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0,
                    "short_margin_used": 0.0,
                }
                for ticker in tickers
            },
            "realized_gains": {
                ticker: {
                    "long": 0.0,
                    "short": 0.0,
                }
                for ticker in tickers
            },
        }

    metadata_overrides = _build_override_logging_metadata(
        tickers=tickers,
        start_date=inputs.start_date,
        end_date=inputs.end_date,
        log_file=inputs.log_file,
    )

    result = run_hedge_fund(
        tickers=tickers,
        start_date=inputs.start_date,
        end_date=inputs.end_date,
        portfolio=portfolio,
        show_reasoning=inputs.show_reasoning,
        selected_analysts=inputs.selected_analysts,
        model_name=inputs.model_name,
        model_provider=inputs.model_provider,
        metadata_overrides=metadata_overrides,
    )
    if inputs.log_file:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_trading_output(result)
        rendered = buffer.getvalue()
        log_path = Path(inputs.log_file).expanduser()
        try:
            if log_path.parent and not log_path.parent.exists():
                log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(strip_ansi(rendered))
        except OSError as error:
            print(f"Failed to write log file '{log_path}': {error}", file=sys.stderr)
        print(rendered, end="")
    else:
        print_trading_output(result)
