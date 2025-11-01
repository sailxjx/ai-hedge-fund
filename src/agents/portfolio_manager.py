import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.async_state import update_analyst_signals_async
from src.utils.llm import async_call_llm, call_llm
from src.utils.progress import progress


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState, agent_id: str = "portfolio_manager"):
    """Makes final trading decisions and generates orders for multiple tickers"""

    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    overrides_by_ticker = {}
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Processing analyst signals")

        # Find the corresponding risk manager for this portfolio manager
        if agent_id.startswith("portfolio_manager_"):
            suffix = agent_id.split("_")[-1]
            risk_manager_id = f"risk_management_agent_{suffix}"
        else:
            risk_manager_id = "risk_management_agent"  # Fallback for CLI

        risk_data = analyst_signals.get(risk_manager_id, {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0.0)
        current_prices[ticker] = float(risk_data.get("current_price", 0.0))
        overrides_by_ticker[ticker] = risk_data.get("overrides", {}) or {}

        # Calculate maximum shares allowed based on position limit and price
        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] // current_prices[ticker])
        else:
            max_shares[ticker] = 0

        # Compress analyst signals to {sig, conf}
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if not agent.startswith("risk_management_agent") and ticker in signals:
                sig = signals[ticker].get("signal")
                conf = signals[ticker].get("confidence")
                if sig is not None and conf is not None:
                    ticker_signals[agent] = {"sig": sig, "conf": conf}
        signals_by_ticker[ticker] = ticker_signals

    state["data"]["current_prices"] = current_prices

    progress.update_status(agent_id, None, "Generating trading decisions")

    result = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        overrides=overrides_by_ticker,
        agent_id=agent_id,
        state=state,
    )
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}, "Portfolio Manager")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


async def portfolio_management_agent_async(state: AgentState, agent_id: str = "portfolio_manager"):
    """Async implementation of the portfolio manager using async LLM calls."""

    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    overrides_by_ticker = {}
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Processing analyst signals")

        if agent_id.startswith("portfolio_manager_"):
            suffix = agent_id.split("_")[-1]
            risk_manager_id = f"risk_management_agent_{suffix}"
        else:
            risk_manager_id = "risk_management_agent"

        risk_data = analyst_signals.get(risk_manager_id, {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0.0)
        current_prices[ticker] = float(risk_data.get("current_price", 0.0))
        overrides_by_ticker[ticker] = risk_data.get("overrides", {}) or {}

        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] // current_prices[ticker])
        else:
            max_shares[ticker] = 0

        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if not agent.startswith("risk_management_agent") and ticker in signals:
                sig = signals[ticker].get("signal")
                conf = signals[ticker].get("confidence")
                if sig is not None and conf is not None:
                    ticker_signals[agent] = {"sig": sig, "conf": conf}
        signals_by_ticker[ticker] = ticker_signals

    state["data"]["current_prices"] = current_prices

    progress.update_status(agent_id, None, "Generating trading decisions")

    result = await async_generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        overrides=overrides_by_ticker,
        agent_id=agent_id,
        state=state,
    )
    serialized = {ticker: decision.model_dump() for ticker, decision in result.decisions.items()}
    await update_analyst_signals_async(state, agent_id, serialized)

    message = HumanMessage(content=json.dumps(serialized), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(serialized, "Portfolio Manager")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": list(state["messages"]) + [message],
        "data": state["data"],
    }


def compute_allowed_actions(
    tickers: list[str],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float],
    overrides: dict[str, dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Compute allowed actions and max quantities per ticker deterministically, honoring overrides."""
    allowed = {}
    cash = float(portfolio.get("cash", 0.0))
    positions = portfolio.get("positions", {}) or {}
    margin_requirement = float(portfolio.get("margin_requirement", 0.5))
    margin_used = float(portfolio.get("margin_used", 0.0))
    equity = float(portfolio.get("equity", cash))

    for ticker in tickers:
        price = float(current_prices.get(ticker, 0.0))
        pos = positions.get(
            ticker,
            {"long": 0, "long_cost_basis": 0.0, "short": 0, "short_cost_basis": 0.0},
        )
        long_shares = int(pos.get("long", 0) or 0)
        short_shares = int(pos.get("short", 0) or 0)
        max_qty = int(max_shares.get(ticker, 0) or 0)
        override = overrides.get(ticker, {}) or {}

        # Start with zeros
        actions = {"buy": 0, "sell": 0, "short": 0, "cover": 0, "hold": 0}

        # Long side
        if long_shares > 0:
            actions["sell"] = long_shares
        if cash > 0 and price > 0:
            max_buy_cash = int(cash // price)
            max_buy = max(0, min(max_qty, max_buy_cash))
            if max_buy > 0:
                actions["buy"] = max_buy

        # Short side
        if short_shares > 0:
            actions["cover"] = short_shares
        if price > 0 and max_qty > 0:
            if margin_requirement <= 0.0:
                # If margin requirement is zero or unset, only cap by max_qty
                max_short = max_qty
            else:
                available_margin = max(0.0, (equity / margin_requirement) - margin_used)
                max_short_margin = int(available_margin // price)
                max_short = max(0, min(max_qty, max_short_margin))
            if max_short > 0:
                actions["short"] = max_short

        # Hold always valid
        actions["hold"] = 0

        # Apply overrides from risk controls
        preferred_direction = (override.get("preferred_direction") or "").lower()
        if preferred_direction == "long":
            actions.pop("short", None)
        elif preferred_direction == "short":
            actions.pop("buy", None)

        if override.get("block_new_shorts"):
            actions.pop("short", None)
        max_short_add = override.get("max_additional_short_shares")
        if "short" in actions and isinstance(max_short_add, (int, float)):
            capped_short = min(actions["short"], int(max_short_add))
            if capped_short > 0:
                actions["short"] = capped_short
            else:
                actions.pop("short", None)

        target_short = override.get("target_short_shares")
        if target_short is not None:
            try:
                target_short_int = max(0, int(target_short))
            except (TypeError, ValueError):
                target_short_int = short_shares

            if short_shares > target_short_int:
                desired_cover = max(0, short_shares - target_short_int)
                if desired_cover > 0:
                    actions["cover"] = min(actions.get("cover", 0), desired_cover)
            else:
                actions["cover"] = 0

            if "short" in actions:
                additional_capacity = max(0, target_short_int - short_shares)
                if additional_capacity <= 0:
                    actions.pop("short", None)
                else:
                    capped_short = min(actions["short"], additional_capacity)
                    if capped_short > 0:
                        actions["short"] = capped_short
                    else:
                        actions.pop("short", None)

        force_cover_qty = override.get("force_cover_qty")
        if short_shares > 0 and isinstance(force_cover_qty, (int, float)):
            forced = max(0, int(force_cover_qty))
            if forced > 0:
                actions["cover"] = max(actions.get("cover", 0), min(short_shares, forced))

        target_long = override.get("target_long_shares")
        if target_long is not None:
            try:
                target_long_int = max(0, int(target_long))
            except (TypeError, ValueError):
                target_long_int = long_shares
            desired_buy = max(0, target_long_int - long_shares)
            if desired_buy > 0:
                actions["buy"] = max(actions.get("buy", 0), desired_buy)

        # Prune zero-capacity actions to reduce tokens, keep hold
        pruned = {"hold": 0}
        for k, v in actions.items():
            if k != "hold" and v > 0:
                pruned[k] = v

        allowed[ticker] = pruned

    return allowed


def _compact_signals(signals_by_ticker: dict[str, dict]) -> dict[str, dict]:
    """Keep only {agent: {sig, conf}} and drop empty agents."""
    out = {}
    for t, agents in signals_by_ticker.items():
        if not agents:
            out[t] = {}
            continue
        compact = {}
        for agent, payload in agents.items():
            sig = payload.get("sig") or payload.get("signal")
            conf = payload.get("conf") if "conf" in payload else payload.get("confidence")
            if sig is not None and conf is not None:
                compact[agent] = {"sig": sig, "conf": conf}
        out[t] = compact
    return out


@dataclass
class DecisionContext:
    tickers_for_llm: list[str]
    prefilled_decisions: dict[str, PortfolioDecision]
    allowed_actions_full: dict[str, dict[str, int]]
    compact_signals_llm: dict[str, dict]
    compact_allowed_llm: dict[str, dict]
    compact_signals_all: dict[str, dict]
    overrides_full: dict[str, dict[str, Any]]
    overrides_llm: dict[str, dict[str, Any]]
    portfolio_snapshot: dict[str, Any]
    observation_table: str
    locked_summary: dict[str, dict[str, Any]]


def _summarize_signals_for_table(signals: dict[str, dict]) -> str:
    if not signals:
        return "-"
    items = sorted(signals.items(), key=lambda kv: kv[1].get("conf", 0), reverse=True)
    slices = []
    for agent, payload in items[:4]:
        sig = payload.get("sig") or payload.get("signal") or "-"
        conf = payload.get("conf") or payload.get("confidence") or 0
        slices.append(f"{agent}:{sig}@{int(conf)}")
    return "; ".join(slices)


def _summarize_allowed_for_table(allowed: dict[str, int]) -> str:
    if not allowed:
        return "-"
    parts = []
    for action, qty in allowed.items():
        if qty:
            parts.append(f"{action}:{qty}")
    return ", ".join(parts) if parts else "-"


def _summarize_overrides_for_table(override: dict[str, Any]) -> str:
    if not override:
        return "-"
    flags = []
    if override.get("block_new_shorts"):
        flags.append("block_shorts")
    pref = override.get("preferred_direction")
    if pref:
        flags.append(f"pref:{pref}")
    if override.get("target_long_shares") is not None:
        flags.append(f"target_long:{override.get('target_long_shares')}")
    if override.get("target_short_shares") is not None:
        flags.append(f"target_short:{override.get('target_short_shares')}")
    if override.get("force_cover_qty") is not None:
        flags.append(f"force_cover:{override.get('force_cover_qty')}")
    return ", ".join(flags) if flags else "-"


def _build_observation_table(
    tickers: list[str],
    current_prices: dict[str, float],
    positions: dict[str, dict[str, Any]],
    allowed_actions_full: dict[str, dict[str, int]],
    compact_signals_all: dict[str, dict],
    overrides: dict[str, dict[str, Any]],
) -> str:
    headers = ["Ticker", "Price", "Long", "Short", "Allowed", "Signals", "Overrides"]
    col_widths = [len(h) for h in headers]
    rows: list[list[str]] = []
    for ticker in tickers:
        pos = positions.get(ticker, {}) or {}
        long_shares = int(pos.get("long", 0) or 0)
        short_shares = int(pos.get("short", 0) or 0)
        price = current_prices.get(ticker)
        allowed = allowed_actions_full.get(ticker, {}) or {}
        signals = compact_signals_all.get(ticker, {}) or {}
        override = overrides.get(ticker, {}) or {}

        row = [
            ticker,
            f"{price:.2f}" if price is not None else "-",
            str(long_shares),
            str(short_shares),
            _summarize_allowed_for_table(allowed),
            _summarize_signals_for_table(signals),
            _summarize_overrides_for_table(override),
        ]
        col_widths = [max(col_widths[i], len(row[i])) for i in range(len(headers))]
        rows.append(row)

    def fmt_row(values: list[str]) -> str:
        return " | ".join(value.ljust(col_widths[idx]) for idx, value in enumerate(values))

    table_lines = [fmt_row(headers), "-+-".join("-" * w for w in col_widths)]
    table_lines.extend(fmt_row(row) for row in rows)
    return "\n".join(table_lines)


def _prepare_decision_context(
    *,
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float],
    overrides: dict[str, dict[str, Any]],
) -> tuple[DecisionContext | None, PortfolioManagerOutput | None]:
    allowed_actions_full = compute_allowed_actions(tickers, current_prices, max_shares, portfolio, overrides)

    prefilled_decisions: dict[str, PortfolioDecision] = {}
    tickers_for_llm: list[str] = []
    positions = portfolio.get("positions") or {}
    for t in tickers:
        aa = allowed_actions_full.get(t, {"hold": 0})
        override = overrides.get(t, {}) or {}
        pos = positions.get(t, {"short": 0})
        short_shares = int(pos.get("short", 0) or 0)
        long_shares = int(pos.get("long", 0) or 0)

        force_cover_qty = override.get("force_cover_qty")
        if short_shares > 0 and isinstance(force_cover_qty, (int, float)):
            qty = min(short_shares, max(0, int(force_cover_qty)))
            if qty > 0:
                reason = override.get("force_cover_reason") or "Stop-loss override"
                prefilled_decisions[t] = PortfolioDecision(action="cover", quantity=qty, confidence=100.0, reasoning=reason[:100])
                continue

        target_long = override.get("target_long_shares")
        if target_long is not None:
            try:
                target_long_int = max(0, int(target_long))
            except (TypeError, ValueError):
                target_long_int = long_shares
            desired_buy = max(0, target_long_int - long_shares)
            if desired_buy > 0 and aa.get("buy"):
                qty = min(desired_buy, aa["buy"])
                if qty > 0:
                    reason = override.get("force_buy_reason") or "Risk manager long target"
                    prefilled_decisions[t] = PortfolioDecision(action="buy", quantity=qty, confidence=100.0, reasoning=reason[:100])
                    continue

        target_short = override.get("target_short_shares")
        if target_short is not None and aa.get("short"):
            try:
                target_short_int = max(0, int(target_short))
            except (TypeError, ValueError):
                target_short_int = short_shares
            desired_short = max(0, target_short_int - short_shares)
            if desired_short > 0:
                qty = min(desired_short, aa["short"])
                if qty > 0:
                    reason = override.get("force_short_reason") or "Risk manager short target"
                    prefilled_decisions[t] = PortfolioDecision(action="short", quantity=qty, confidence=100.0, reasoning=reason[:100])
                    continue

        if set(aa.keys()) == {"hold"}:
            prefilled_decisions[t] = PortfolioDecision(action="hold", quantity=0, confidence=100.0, reasoning="No valid trade available")
        else:
            tickers_for_llm.append(t)

    if not tickers_for_llm:
        return None, PortfolioManagerOutput(decisions=prefilled_decisions)

    compact_signals_all = _compact_signals(signals_by_ticker)
    compact_signals_llm = {t: compact_signals_all.get(t, {}) for t in tickers_for_llm}
    compact_allowed_llm = {t: allowed_actions_full[t] for t in tickers_for_llm}
    overrides_full = {t: overrides.get(t, {}) or {} for t in tickers}
    overrides_llm = {t: overrides_full.get(t, {}) for t in tickers_for_llm}

    portfolio_snapshot = {
        "cash": float(portfolio.get("cash", 0.0)),
        "equity": portfolio.get("equity"),
        "margin_requirement": portfolio.get("margin_requirement"),
        "margin_used": portfolio.get("margin_used"),
        "positions": positions,
    }
    observation_table = _build_observation_table(
        tickers=tickers,
        current_prices=current_prices,
        positions=positions,
        allowed_actions_full=allowed_actions_full,
        compact_signals_all=compact_signals_all,
        overrides=overrides_full,
    )
    locked_summary = {ticker: decision.model_dump() for ticker, decision in prefilled_decisions.items()}

    return (
        DecisionContext(
            tickers_for_llm=tickers_for_llm,
            prefilled_decisions=prefilled_decisions,
            allowed_actions_full=allowed_actions_full,
            compact_signals_llm=compact_signals_llm,
            compact_allowed_llm=compact_allowed_llm,
            compact_signals_all=compact_signals_all,
            overrides_full=overrides_full,
            overrides_llm=overrides_llm,
            portfolio_snapshot=portfolio_snapshot,
            observation_table=observation_table,
            locked_summary=locked_summary,
        ),
        None,
    )


def generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float],
    overrides: dict[str, dict[str, Any]],
    agent_id: str,
    state: AgentState,
) -> PortfolioManagerOutput:
    """Get decisions from the LLM with deterministic constraints and a minimal prompt."""

    context, immediate = _prepare_decision_context(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        overrides=overrides,
    )
    if immediate is not None:
        return immediate

    # Deterministic constraints
    prefilled_decisions = context.prefilled_decisions
    tickers_for_llm = context.tickers_for_llm
    allowed_actions_full = context.allowed_actions_full
    compact_signals_llm = context.compact_signals_llm
    compact_allowed_llm = context.compact_allowed_llm

    # Minimal prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the portfolio manager for a multi-ticker book.\n"
                "Primary mandate: compound risk-adjusted returns while respecting every risk override, preserving capital, and balancing exposures across the basket.\n"
                "For each ticker, weigh analyst consensus depth, confidence dispersion, existing sizing implied by the allowed actions, and any risk-manager nudges. \n"
                "Favor diversified positioning instead of piling into a single name unless several independent personas express aligned, high-conviction signals and risk blocks are cleared. \n"
                "Scaling in/out is encouraged—partial trades are fine, and rotating capital toward the strongest relative setup while trimming stretched names keeps the book healthy. \n"
                "If risk metadata suggests caution (blocks, crash mode, squeeze caps), default to preservation even when some analysts are excited. \n"
                "Reasoning must stay under 120 characters and reference the decisive factor (e.g., consensus breadth, override, hedge). Return JSON only.",
            ),
            (
                "human",
                "Portfolio snapshot:\n{portfolio_snapshot}\n\n"
                "Observations table:\n{observation_table}\n\n"
                "Locked trades (do not modify):\n{locked}\n\n"
                "Signals (focus tickers) JSON:\n{signals_llm}\n\n"
                "Allowed (focus tickers) JSON:\n{allowed_llm}\n\n"
                "Overrides (focus tickers) JSON:\n{overrides_llm}\n\n"
                "Format:\n"
                "{{\n"
                '  "decisions": {{\n'
                '    "TICKER": {{"action":"...","quantity":int,"confidence":int,"reasoning":"..."}}\n'
                "  }}\n"
                "}}",
            ),
        ]
    )

    prompt_data = {
        "portfolio_snapshot": json.dumps(context.portfolio_snapshot, separators=(",", ":"), ensure_ascii=False),
        "observation_table": context.observation_table,
        "locked": json.dumps(context.locked_summary, separators=(",", ":"), ensure_ascii=False),
        "signals_llm": json.dumps(context.compact_signals_llm, separators=(",", ":"), ensure_ascii=False),
        "allowed_llm": json.dumps(context.compact_allowed_llm, separators=(",", ":"), ensure_ascii=False),
        "overrides_llm": json.dumps(context.overrides_llm, separators=(",", ":"), ensure_ascii=False),
    }
    prompt = template.invoke(prompt_data)

    # Default factory fills remaining tickers as hold if the LLM fails
    def create_default_portfolio_output():
        # start from prefilled
        decisions = dict(prefilled_decisions)
        for t in tickers_for_llm:
            decisions[t] = PortfolioDecision(action="hold", quantity=0, confidence=0.0, reasoning="Default decision: hold")
        return PortfolioManagerOutput(decisions=decisions)

    llm_out = call_llm(
        prompt=prompt,
        pydantic_model=PortfolioManagerOutput,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_portfolio_output,
    )

    # Merge prefilled holds with LLM results
    merged = dict(prefilled_decisions)
    merged.update(llm_out.decisions)

    debug_path = os.getenv("PORTFOLIO_MANAGER_DEBUG_LOG")
    if debug_path:
        try:
            portfolio_dict = portfolio if isinstance(portfolio, dict) else {}
            positions = portfolio_dict.get("positions", {}) if isinstance(portfolio_dict, dict) else {}
            debug_payload = {
                "logged_at": datetime.now(timezone.utc).isoformat(),
                "agent_id": agent_id,
                "start_date": state.get("data", {}).get("start_date"),
                "end_date": state.get("data", {}).get("end_date"),
                "tickers": tickers,
                "current_prices": current_prices,
                "max_shares": max_shares,
                "portfolio": {
                    "cash": float(portfolio_dict.get("cash", 0.0)) if portfolio_dict else None,
                    "equity": portfolio_dict.get("equity") if portfolio_dict else None,
                    "margin_requirement": portfolio_dict.get("margin_requirement") if portfolio_dict else None,
                    "margin_used": portfolio_dict.get("margin_used") if portfolio_dict else None,
                    "positions": positions,
                },
                "overrides": overrides,
                "allowed_actions": allowed_actions_full,
                "prefilled": {ticker: decision.model_dump() for ticker, decision in prefilled_decisions.items()},
                "decisions": {ticker: decision.model_dump() for ticker, decision in merged.items()},
            }
            with open(debug_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(debug_payload, default=str))
                handle.write("\n")
        except Exception:
            # Debug logging must not interfere with trading execution.
            pass

    return PortfolioManagerOutput(decisions=merged)


async def async_generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float],
    overrides: dict[str, dict[str, Any]],
    agent_id: str,
    state: AgentState,
) -> PortfolioManagerOutput:
    """Async counterpart to generate_trading_decision using async_call_llm."""

    context, immediate = _prepare_decision_context(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        overrides=overrides,
    )
    if immediate is not None:
        return immediate

    prefilled_decisions = context.prefilled_decisions
    tickers_for_llm = context.tickers_for_llm

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the portfolio manager for a multi-ticker book.\n"
                "Primary mandate: compound risk-adjusted returns while respecting every risk override, preserving capital, and balancing exposures across the basket.\n"
                "For each ticker, weigh analyst consensus depth, confidence dispersion, existing sizing implied by the allowed actions, and any risk-manager nudges. \n"
                "Favor diversified positioning instead of piling into a single name unless several independent personas express aligned, high-conviction signals and risk blocks are cleared. \n"
                "Scaling in/out is encouraged—partial trades are fine, and rotating capital toward the strongest relative setup while trimming stretched names keeps the book healthy. \n"
                "If risk metadata suggests caution (blocks, crash mode, squeeze caps), default to preservation even when some analysts are excited. \n"
                "Reasoning must stay under 120 characters and reference the decisive factor (e.g., consensus breadth, override, hedge). Return JSON only.",
            ),
            (
                "human",
                "Portfolio snapshot:\n{portfolio_snapshot}\n\n"
                "Observations table:\n{observation_table}\n\n"
                "Locked trades (do not modify):\n{locked}\n\n"
                "Signals (focus tickers) JSON:\n{signals_llm}\n\n"
                "Allowed (focus tickers) JSON:\n{allowed_llm}\n\n"
                "Overrides (focus tickers) JSON:\n{overrides_llm}\n\n"
                "Format:\n"
                "{{\n"
                '  "decisions": {{\n'
                '    "TICKER": {{"action":"...","quantity":int,"confidence":int,"reasoning":"..."}}\n'
                "  }}\n"
                "}}",
            ),
        ]
    )

    prompt = template.invoke(
        {
            "portfolio_snapshot": json.dumps(context.portfolio_snapshot, separators=(",", ":"), ensure_ascii=False),
            "observation_table": context.observation_table,
            "locked": json.dumps(context.locked_summary, separators=(",", ":"), ensure_ascii=False),
            "signals_llm": json.dumps(context.compact_signals_llm, separators=(",", ":"), ensure_ascii=False),
            "allowed_llm": json.dumps(context.compact_allowed_llm, separators=(",", ":"), ensure_ascii=False),
            "overrides_llm": json.dumps(context.overrides_llm, separators=(",", ":"), ensure_ascii=False),
        }
    )

    def create_default_portfolio_output():
        decisions = dict(prefilled_decisions)
        for t in tickers_for_llm:
            decisions[t] = PortfolioDecision(action="hold", quantity=0, confidence=0.0, reasoning="Default decision: hold")
        return PortfolioManagerOutput(decisions=decisions)

    llm_out = await async_call_llm(
        prompt=prompt,
        pydantic_model=PortfolioManagerOutput,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_portfolio_output,
    )

    merged = dict(prefilled_decisions)
    merged.update(llm_out.decisions)
    return PortfolioManagerOutput(decisions=merged)
