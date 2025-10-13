from __future__ import annotations

import json
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import (
    get_company_news,
    get_company_news_async,
    get_financial_metrics,
    get_financial_metrics_async,
    get_insider_trades,
    get_insider_trades_async,
    get_market_cap,
    get_market_cap_async,
    search_line_items,
    search_line_items_async,
)
from src.utils.api_key import get_api_key_from_state
from src.utils.async_state import update_analyst_signals_async
from src.utils.llm import async_call_llm, call_llm
from src.utils.progress import progress


class MichaelBurrySignal(BaseModel):
    """Schema returned by the LLM."""

    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0–100
    reasoning: str


def michael_burry_agent(state: AgentState, agent_id: str = "michael_burry_agent"):
    """Analyse stocks using Michael Burry's deep‑value, contrarian framework."""
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]  # YYYY‑MM‑DD
    tickers: list[str] = data["tickers"]

    # We look one year back for insider trades / news flow
    start_date = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()

    analysis_data: dict[str, dict] = {}
    burry_analysis: dict[str, dict] = {}

    for ticker in tickers:
        # ------------------------------------------------------------------
        # Fetch raw data
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching line items")
        line_items = search_line_items(
            ticker,
            [
                "free_cash_flow",
                "net_income",
                "total_debt",
                "cash_and_equivalents",
                "total_assets",
                "total_liabilities",
                "outstanding_shares",
                "issuance_or_purchase_of_equity_shares",
            ],
            end_date,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date=end_date, start_date=start_date)

        progress.update_status(agent_id, ticker, "Fetching company news")
        news = get_company_news(ticker, end_date=end_date, start_date=start_date, limit=250)

        progress.update_status(agent_id, ticker, "Fetching market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        # ------------------------------------------------------------------
        # Run sub‑analyses
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Analyzing value")
        value_analysis = _analyze_value(metrics, line_items, market_cap)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet")
        balance_sheet_analysis = _analyze_balance_sheet(metrics, line_items)

        progress.update_status(agent_id, ticker, "Analyzing insider activity")
        insider_analysis = _analyze_insider_activity(insider_trades)

        progress.update_status(agent_id, ticker, "Analyzing contrarian sentiment")
        contrarian_analysis = _analyze_contrarian_sentiment(news)

        analysis_data[ticker] = {
            "value": value_analysis,
            "balance_sheet": balance_sheet_analysis,
            "insider_activity": insider_analysis,
            "contrarian_sentiment": contrarian_analysis,
            "context": {
                "market_cap": market_cap,
                "latest_financial_metrics": metrics[0].model_dump() if metrics else None,
                "line_items_analyzed": len(line_items),
                "news_window_start": start_date,
                "news_window_end": end_date,
            },
        }

        progress.update_status(agent_id, ticker, "Generating LLM output")
        burry_output = _generate_burry_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        burry_analysis[ticker] = {
            "signal": burry_output.signal,
            "confidence": burry_output.confidence,
            "reasoning": burry_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=burry_output.reasoning)

    # ----------------------------------------------------------------------
    # Return to the graph
    # ----------------------------------------------------------------------
    message = HumanMessage(content=json.dumps(burry_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(burry_analysis, "Michael Burry Agent")

    state["data"]["analyst_signals"][agent_id] = burry_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


async def michael_burry_agent_async(state: AgentState, agent_id: str = "michael_burry_agent"):
    """Analyse stocks using Michael Burry's deep‑value, contrarian framework."""
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]  # YYYY‑MM‑DD
    tickers: list[str] = data["tickers"]

    # We look one year back for insider trades / news flow
    start_date = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()

    analysis_data: dict[str, dict] = {}
    burry_analysis: dict[str, dict] = {}

    for ticker in tickers:
        # ------------------------------------------------------------------
        # Fetch raw data
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = await get_financial_metrics_async(ticker, end_date, period="ttm", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching line items")
        line_items = await search_line_items_async(
            ticker,
            [
                "free_cash_flow",
                "net_income",
                "total_debt",
                "cash_and_equivalents",
                "total_assets",
                "total_liabilities",
                "outstanding_shares",
                "issuance_or_purchase_of_equity_shares",
            ],
            end_date,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Fetching insider trades")
        insider_trades = await get_insider_trades_async(ticker, end_date=end_date, start_date=start_date)

        progress.update_status(agent_id, ticker, "Fetching company news")
        news = await get_company_news_async(ticker, end_date=end_date, start_date=start_date, limit=250)

        progress.update_status(agent_id, ticker, "Fetching market cap")
        market_cap = await get_market_cap_async(ticker, end_date, api_key=api_key)

        # ------------------------------------------------------------------
        # Run sub‑analyses
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Analyzing value")
        value_analysis = _analyze_value(metrics, line_items, market_cap)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet")
        balance_sheet_analysis = _analyze_balance_sheet(metrics, line_items)

        progress.update_status(agent_id, ticker, "Analyzing insider activity")
        insider_analysis = _analyze_insider_activity(insider_trades)

        progress.update_status(agent_id, ticker, "Analyzing contrarian sentiment")
        contrarian_analysis = _analyze_contrarian_sentiment(news)

        analysis_data[ticker] = {
            "value": value_analysis,
            "balance_sheet": balance_sheet_analysis,
            "insider_activity": insider_analysis,
            "contrarian_sentiment": contrarian_analysis,
            "context": {
                "market_cap": market_cap,
                "latest_financial_metrics": metrics[0].model_dump() if metrics else None,
                "line_items_analyzed": len(line_items),
                "news_window_start": start_date,
                "news_window_end": end_date,
            },
        }

        progress.update_status(agent_id, ticker, "Generating LLM output")
        burry_output = await _generate_burry_output_async(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        burry_analysis[ticker] = {
            "signal": burry_output.signal,
            "confidence": burry_output.confidence,
            "reasoning": burry_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=burry_output.reasoning)

    # ----------------------------------------------------------------------
    # Return to the graph
    # ----------------------------------------------------------------------
    message = HumanMessage(content=json.dumps(burry_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(burry_analysis, "Michael Burry Agent")

    await update_analyst_signals_async(state, agent_id, burry_analysis)

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


###############################################################################
# Sub‑analysis helpers
###############################################################################


def _latest_line_item(line_items: list):
    """Return the most recent line‑item object or *None*."""
    return line_items[0] if line_items else None


# ----- Value ----------------------------------------------------------------


def _analyze_value(metrics, line_items, market_cap):
    """Free cash‑flow yield, EV/EBIT, other classic deep‑value metrics."""

    notes: list[str] = []
    observations: dict[str, float | None] = {}

    # Free‑cash‑flow yield
    latest_item = _latest_line_item(line_items)
    fcf = getattr(latest_item, "free_cash_flow", None) if latest_item else None
    if fcf is not None and market_cap:
        fcf_yield = fcf / market_cap
        observations["free_cash_flow"] = fcf
        observations["fcf_yield_pct"] = fcf_yield * 100
        if fcf_yield >= 0.15:
            notes.append(f"Extraordinary FCF yield {fcf_yield:.1%}.")
        elif fcf_yield >= 0.12:
            notes.append(f"Very high FCF yield {fcf_yield:.1%}.")
        elif fcf_yield >= 0.08:
            notes.append(f"Respectable FCF yield {fcf_yield:.1%}.")
        else:
            notes.append(f"Low FCF yield {fcf_yield:.1%}.")
    else:
        observations["free_cash_flow"] = fcf
        observations["fcf_yield_pct"] = None
        notes.append("FCF data unavailable or market cap missing.")

    # EV/EBIT (from financial metrics)
    if metrics:
        ev_ebit = getattr(metrics[0], "ev_to_ebit", None)
        if ev_ebit is not None:
            observations["ev_to_ebit"] = ev_ebit
            if ev_ebit < 6:
                notes.append(f"EV/EBIT {ev_ebit:.1f} (<6).")
            elif ev_ebit < 10:
                notes.append(f"EV/EBIT {ev_ebit:.1f} (<10).")
            else:
                notes.append(f"High EV/EBIT {ev_ebit:.1f}.")
        else:
            observations["ev_to_ebit"] = None
            notes.append("EV/EBIT data unavailable.")
    else:
        observations["ev_to_ebit"] = None
        notes.append("Financial metrics unavailable.")

    return {"metrics": observations, "notes": notes}


# ----- Balance sheet --------------------------------------------------------


def _analyze_balance_sheet(metrics, line_items):
    """Leverage and liquidity checks."""

    notes: list[str] = []
    observations: dict[str, float | None] = {}

    latest_metrics = metrics[0] if metrics else None
    latest_item = _latest_line_item(line_items)

    debt_to_equity = getattr(latest_metrics, "debt_to_equity", None) if latest_metrics else None
    if debt_to_equity is not None:
        observations["debt_to_equity"] = debt_to_equity
        if debt_to_equity < 0.5:
            notes.append(f"Low D/E {debt_to_equity:.2f}.")
        elif debt_to_equity < 1:
            notes.append(f"Moderate D/E {debt_to_equity:.2f}.")
        else:
            notes.append(f"Elevated leverage D/E {debt_to_equity:.2f}.")
    else:
        observations["debt_to_equity"] = None
        notes.append("Debt-to-equity data unavailable.")

    # Quick liquidity sanity check (cash vs total debt)
    if latest_item is not None:
        cash = getattr(latest_item, "cash_and_equivalents", None)
        total_debt = getattr(latest_item, "total_debt", None)
        if cash is not None and total_debt is not None:
            observations["cash"] = cash
            observations["total_debt"] = total_debt
            observations["net_cash"] = cash - total_debt
            if cash > total_debt:
                notes.append("Net cash position.")
            else:
                notes.append("Net debt position.")
        else:
            notes.append("Cash/debt data unavailable.")
            observations["cash"] = cash
            observations["total_debt"] = total_debt
            observations["net_cash"] = None
    else:
        notes.append("No recent balance sheet line items.")

    return {"metrics": observations, "notes": notes}


# ----- Insider activity -----------------------------------------------------


def _analyze_insider_activity(insider_trades):
    """Net insider buying over the last 12 months acts as a hard catalyst."""

    notes: list[str] = []
    observations: dict[str, float | None] = {}

    if not insider_trades:
        notes.append("No insider trade data.")
        return {"metrics": observations, "notes": notes}

    shares_bought = sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) > 0)
    shares_sold = abs(sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) < 0))
    net = shares_bought - shares_sold
    observations["shares_bought"] = shares_bought
    observations["shares_sold"] = shares_sold
    observations["net_shares"] = net
    ratio = net / max(shares_sold, 1) if shares_sold is not None else None
    observations["buy_to_sell_ratio"] = ratio if shares_sold else None

    if net > 0:
        notes.append(f"Net insider buying of {net:,} shares.")
    elif net < 0:
        notes.append(f"Net insider selling of {abs(net):,} shares.")
    else:
        notes.append("Insider activity roughly balanced.")

    return {"metrics": observations, "notes": notes}


# ----- Contrarian sentiment -------------------------------------------------


def _analyze_contrarian_sentiment(news):
    """Very rough gauge: a wall of recent negative headlines can be a *positive* for a contrarian."""

    notes: list[str] = []
    observations: dict[str, float | int | None] = {}

    if not news:
        notes.append("No recent news.")
        return {"metrics": observations, "notes": notes}

    # Count negative sentiment articles
    sentiment_negative_count = sum(1 for n in news if n.sentiment and n.sentiment.lower() in ["negative", "bearish"])
    observations["negative_headline_count"] = sentiment_negative_count
    observations["total_headlines"] = len(news)

    if sentiment_negative_count >= 5:
        notes.append(f"{sentiment_negative_count} negative headlines (potential contrarian setup).")
    else:
        notes.append("Limited negative press.")

    return {"metrics": observations, "notes": notes}


###############################################################################
# LLM generation
###############################################################################


def _generate_burry_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> MichaelBurrySignal:
    """Call the LLM to craft the final trading signal in Burry's voice."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI agent emulating Dr. Michael J. Burry. Your mandate:
                - Hunt for deep value in US equities using hard numbers (free cash flow, EV/EBIT, balance sheet)
                - Be contrarian: hatred in the press can be your friend if fundamentals are solid
                - Focus on downside first – avoid leveraged balance sheets
                - Look for hard catalysts such as insider buying, buybacks, or asset sales
                - Communicate in Burry's terse, data‑driven style

                When providing your reasoning, be thorough and specific by:
                1. Start with the key metric(s) that drove your decision
                2. Cite concrete numbers (e.g. "FCF yield 14.7%", "EV/EBIT 5.3")
                3. Highlight risk factors and why they are acceptable (or not)
                4. Mention relevant insider activity or contrarian opportunities
                5. Use Burry's direct, number-focused communication style with minimal words
                
                For example, if bullish: "FCF yield 12.8%. EV/EBIT 6.2. Debt-to-equity 0.4. Net insider buying 25k shares. Market missing value due to overreaction to recent litigation. Strong buy."
                For example, if bearish: "FCF yield only 2.1%. Debt-to-equity concerning at 2.3. Management diluting shareholders. Pass."
                """,
            ),
            (
                "human",
                """Based on the following data, create the investment signal as Michael Burry would:

                Analysis Data for {ticker}:
                {analysis_data}

                Return the trading signal in the following JSON format exactly:
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float between 0 and 100,
                  "reasoning": "string"
                }}
                """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    # Default fallback signal in case parsing fails
    def create_default_michael_burry_signal():
        return MichaelBurrySignal(signal="neutral", confidence=0.0, reasoning="Parsing error – defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=MichaelBurrySignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_michael_burry_signal,
    )


async def _generate_burry_output_async(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> MichaelBurrySignal:
    """Async helper mirroring _generate_burry_output."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI agent emulating Dr. Michael J. Burry. Your mandate:
                - Hunt for deep value in US equities using hard numbers (free cash flow, EV/EBIT, balance sheet)
                - Be contrarian: hatred in the press can be your friend if fundamentals are solid
                - Focus on downside first – avoid leveraged balance sheets
                - Look for hard catalysts such as insider buying, buybacks, or asset sales
                - Communicate in Burry's terse, data‑driven style""",
            ),
            (
                "human",
                """Based on the following data, create the investment signal as Michael Burry would:

                Analysis Data for {ticker}:
                {analysis_data}

                Return the trading signal in JSON:
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float between 0 and 100,
                  "reasoning": "string"
                }}""",
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_michael_burry_signal():
        return MichaelBurrySignal(signal="neutral", confidence=0.0, reasoning="Parsing error – defaulting to neutral")

    return await async_call_llm(
        prompt=prompt,
        pydantic_model=MichaelBurrySignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_michael_burry_signal,
    )
