import json
import os
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm
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
            suffix = agent_id.split('_')[-1]
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
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()},
                             "Portfolio Manager")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
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

    # Deterministic constraints
    allowed_actions_full = compute_allowed_actions(
        tickers, current_prices, max_shares, portfolio, overrides
    )

    # Pre-fill pure holds to avoid sending them to the LLM at all
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
                prefilled_decisions[t] = PortfolioDecision(
                    action="cover", quantity=qty, confidence=100.0, reasoning=reason[:100]
                )
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
                    prefilled_decisions[t] = PortfolioDecision(
                        action="buy", quantity=qty, confidence=100.0, reasoning=reason[:100]
                    )
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
                    prefilled_decisions[t] = PortfolioDecision(
                        action="short", quantity=qty, confidence=100.0, reasoning=reason[:100]
                    )
                    continue

        # If only 'hold' key exists, there is no trade possible
        if set(aa.keys()) == {"hold"}:
            prefilled_decisions[t] = PortfolioDecision(
                action="hold", quantity=0, confidence=100.0, reasoning="No valid trade available"
            )
        else:
            tickers_for_llm.append(t)

    if not tickers_for_llm:
        return PortfolioManagerOutput(decisions=prefilled_decisions)

    # Build compact payloads only for tickers sent to LLM
    compact_signals = _compact_signals({t: signals_by_ticker.get(t, {}) for t in tickers_for_llm})
    compact_allowed = {t: allowed_actions_full[t] for t in tickers_for_llm}

    # Minimal prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a portfolio manager.\n"
                "Ultimate goal: compound risk-adjusted returns while fully respecting risk guardrails and capital preservation.\n"
                "Inputs per ticker: analyst signals and allowed actions with max qty (already validated).\n"
                "Pick one allowed action per ticker and a quantity ≤ the max. "
                "Keep reasoning very concise (max 100 chars). No cash or margin math. Return JSON only."
            ),
            (
                "human",
                "Signals:\n{signals}\n\n"
                "Allowed:\n{allowed}\n\n"
                "Format:\n"
                "{{\n"
                '  "decisions": {{\n'
                '    "TICKER": {{"action":"...","quantity":int,"confidence":int,"reasoning":"..."}}\n'
                "  }}\n"
                "}}"
            ),
        ]
    )

    prompt_data = {
        "signals": json.dumps(compact_signals, separators=(",", ":"), ensure_ascii=False),
        "allowed": json.dumps(compact_allowed, separators=(",", ":"), ensure_ascii=False),
    }
    prompt = template.invoke(prompt_data)

    # Default factory fills remaining tickers as hold if the LLM fails
    def create_default_portfolio_output():
        # start from prefilled
        decisions = dict(prefilled_decisions)
        for t in tickers_for_llm:
            decisions[t] = PortfolioDecision(
                action="hold", quantity=0, confidence=0.0, reasoning="Default decision: hold"
            )
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
                "logged_at": datetime.utcnow().isoformat(),
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
