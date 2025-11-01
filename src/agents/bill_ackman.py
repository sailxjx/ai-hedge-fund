import json
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import (
    get_financial_metrics,
    get_financial_metrics_async,
    get_market_cap,
    get_market_cap_async,
    search_line_items,
    search_line_items_async,
)
from src.utils.api_key import get_api_key_from_state
from src.utils.async_state import update_analyst_signals_async
from src.utils.llm import async_call_llm, call_llm
from src.utils.progress import progress


class BillAckmanSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def bill_ackman_agent(state: AgentState, agent_id: str = "bill_ackman_agent"):
    """
    Analyzes stocks using Bill Ackman's investing principles and LLM reasoning.
    Fetches multiple periods of data for a more robust long-term view.
    Incorporates brand/competitive advantage, activism potential, and other key factors.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    ackman_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        # Request multiple periods of data (annual or TTM) for a more robust long-term view.
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                # Optional: intangible_assets if available
                # "intangible_assets"
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing business quality")
        quality_analysis = analyze_business_quality(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet and capital structure")
        balance_sheet_analysis = analyze_financial_discipline(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing activism potential")
        activism_analysis = analyze_activism_potential(financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating intrinsic value & margin of safety")
        valuation_analysis = analyze_valuation(financial_line_items, market_cap)

        analysis_data[ticker] = {
            "business_quality": quality_analysis,
            "financial_discipline": balance_sheet_analysis,
            "activism": activism_analysis,
            "valuation": valuation_analysis,
            "context": {
                "market_cap": market_cap,
                "latest_financial_metrics": metrics[0].model_dump() if metrics else None,
                "periods_analyzed": len(financial_line_items),
            },
        }

        progress.update_status(agent_id, ticker, "Generating Bill Ackman analysis")
        ackman_output = generate_ackman_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        ackman_analysis[ticker] = {"signal": ackman_output.signal, "confidence": ackman_output.confidence, "reasoning": ackman_output.reasoning}

        progress.update_status(agent_id, ticker, "Done", analysis=ackman_output.reasoning)

    # Wrap results in a single message for the chain
    message = HumanMessage(content=json.dumps(ackman_analysis), name=agent_id)

    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(ackman_analysis, "Bill Ackman Agent")

    # Add signals to the overall state
    state["data"]["analyst_signals"][agent_id] = ackman_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


async def bill_ackman_agent_async(state: AgentState, agent_id: str = "bill_ackman_agent"):
    """
    Analyzes stocks using Bill Ackman's investing principles and LLM reasoning.
    Fetches multiple periods of data for a more robust long-term view.
    Incorporates brand/competitive advantage, activism potential, and other key factors.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    ackman_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = await get_financial_metrics_async(ticker, end_date, period="annual", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        # Request multiple periods of data (annual or TTM) for a more robust long-term view.
        financial_line_items = await search_line_items_async(
            ticker,
            [
                "revenue",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                # Optional: intangible_assets if available
                # "intangible_assets"
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = await get_market_cap_async(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing business quality")
        quality_analysis = analyze_business_quality(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet and capital structure")
        balance_sheet_analysis = analyze_financial_discipline(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing activism potential")
        activism_analysis = analyze_activism_potential(financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating intrinsic value & margin of safety")
        valuation_analysis = analyze_valuation(financial_line_items, market_cap)

        analysis_data[ticker] = {
            "business_quality": quality_analysis,
            "financial_discipline": balance_sheet_analysis,
            "activism": activism_analysis,
            "valuation": valuation_analysis,
            "context": {
                "market_cap": market_cap,
                "latest_financial_metrics": metrics[0].model_dump() if metrics else None,
                "periods_analyzed": len(financial_line_items),
            },
        }

        progress.update_status(agent_id, ticker, "Generating Bill Ackman analysis")
        ackman_output = await generate_ackman_output_async(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        ackman_analysis[ticker] = {"signal": ackman_output.signal, "confidence": ackman_output.confidence, "reasoning": ackman_output.reasoning}

        progress.update_status(agent_id, ticker, "Done", analysis=ackman_output.reasoning)

    # Wrap results in a single message for the chain
    message = HumanMessage(content=json.dumps(ackman_analysis), name=agent_id)

    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(ackman_analysis, "Bill Ackman Agent")

    # Add signals to the overall state
    await update_analyst_signals_async(state, agent_id, ackman_analysis)

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def analyze_business_quality(metrics: list, financial_line_items: list) -> dict:
    """
    Surface business-quality observations without prescribing a verdict.
    Returns raw growth/profitability snapshots and explanatory notes that the LLM can interpret.
    """
    notes: list[str] = []
    observations: dict[str, float | int | None] = {}

    if not metrics or not financial_line_items:
        notes.append("Insufficient data to analyze business quality.")
        return {"metrics": observations, "notes": notes}

    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    if len(revenues) >= 2 and revenues[-1] and revenues[0]:
        try:
            growth_rate = (revenues[0] - revenues[-1]) / abs(revenues[-1])
        except ZeroDivisionError:
            growth_rate = None
        if growth_rate is not None:
            observations["revenue_growth_pct"] = growth_rate * 100
            notes.append(f"Revenue change over sample: {growth_rate*100:.1f}% (oldest {revenues[-1]:,.2f}, latest {revenues[0]:,.2f}).")
    else:
        notes.append("Not enough revenue data for multi-period trend.")

    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]

    if op_margin_vals:
        above_15 = sum(1 for m in op_margin_vals if m is not None and m > 0.15)
        ratio = above_15 / len(op_margin_vals)
        observations["operating_margin_above_15_ratio"] = ratio
        observations["operating_margin_latest_pct"] = op_margin_vals[0] * 100
        notes.append(f"Operating margin >15% in {above_15}/{len(op_margin_vals)} periods (latest {op_margin_vals[0]*100:.1f}%).")
    else:
        notes.append("No operating margin data across periods.")

    if fcf_vals:
        positive_fcf = sum(1 for f in fcf_vals if f is not None and f > 0)
        ratio = positive_fcf / len(fcf_vals)
        observations["positive_fcf_ratio"] = ratio
        observations["free_cash_flow_latest"] = fcf_vals[0]
        notes.append(f"Positive free cash flow in {positive_fcf}/{len(fcf_vals)} periods (latest {fcf_vals[0]:,.2f}).")
    else:
        notes.append("No free cash flow data across periods.")

    latest_metrics = metrics[0]
    roe = getattr(latest_metrics, "return_on_equity", None)
    if roe is not None:
        observations["return_on_equity_pct"] = roe * 100
        notes.append(f"Latest ROE: {roe*100:.1f}%.")
    else:
        notes.append("ROE data not available.")

    return {"metrics": observations, "notes": notes}


def analyze_financial_discipline(metrics: list, financial_line_items: list) -> dict:
    """
    Evaluate the company's balance sheet over multiple periods:
    - Debt ratio trends
    - Capital returns to shareholders over time (dividends, buybacks)
    """
    notes: list[str] = []
    observations: dict[str, float | int | None] = {}

    if not metrics or not financial_line_items:
        notes.append("Insufficient data to analyze financial discipline.")
        return {"metrics": observations, "notes": notes}

    # 1. Multi-period debt ratio or debt_to_equity
    debt_to_equity_vals = [item.debt_to_equity for item in financial_line_items if item.debt_to_equity is not None]
    if debt_to_equity_vals:
        below_one_count = sum(1 for d in debt_to_equity_vals if d < 1.0)
        observations["debt_to_equity_median"] = float(sorted(debt_to_equity_vals)[len(debt_to_equity_vals) // 2])
        observations["debt_to_equity_below_1_ratio"] = below_one_count / len(debt_to_equity_vals)
        notes.append(f"Debt-to-equity observations: {below_one_count}/{len(debt_to_equity_vals)} below 1.0.")
    else:
        # Fallback to total_liabilities / total_assets
        liab_to_assets = []
        for item in financial_line_items:
            if item.total_liabilities and item.total_assets and item.total_assets > 0:
                liab_to_assets.append(item.total_liabilities / item.total_assets)

        if liab_to_assets:
            below_50pct_count = sum(1 for ratio in liab_to_assets if ratio < 0.5)
            observations["liabilities_to_assets_median"] = float(sorted(liab_to_assets)[len(liab_to_assets) // 2])
            observations["liabilities_to_assets_below_50pct_ratio"] = below_50pct_count / len(liab_to_assets)
            notes.append(f"Liabilities-to-assets <50% in {below_50pct_count}/{len(liab_to_assets)} periods.")
        else:
            notes.append("No consistent leverage ratio data available.")

    # 2. Capital allocation approach (dividends + share counts)
    dividends_list = [item.dividends_and_other_cash_distributions for item in financial_line_items if item.dividends_and_other_cash_distributions is not None]
    if dividends_list:
        paying_dividends_count = sum(1 for d in dividends_list if d < 0)
        observations["dividend_payment_ratio"] = paying_dividends_count / len(dividends_list)
        notes.append(f"Dividend payments observed in {paying_dividends_count}/{len(dividends_list)} periods (negative cash outflows).")
    else:
        notes.append("No dividend data found across periods.")

    # Check for decreasing share count (simple approach)
    shares = [item.outstanding_shares for item in financial_line_items if item.outstanding_shares is not None]
    if len(shares) >= 2:
        # For buybacks, the newest count should be less than the oldest count
        if shares[-1] != 0:
            share_change_pct = (shares[0] - shares[-1]) / shares[-1] * 100
        else:
            share_change_pct = None
        observations["share_count_change_pct"] = share_change_pct
        notes.append(f"Share count change over sample: start {shares[-1]:,.0f}, end {shares[0]:,.0f}.")
    else:
        notes.append("No multi-period share count data to assess buybacks.")

    return {"metrics": observations, "notes": notes}


def analyze_activism_potential(financial_line_items: list) -> dict:
    """
    Bill Ackman often engages in activism if a company has a decent brand or moat
    but is underperforming operationally.

    We'll do a simplified approach:
    - Look for positive revenue trends but subpar margins
    - That may indicate 'activism upside' if operational improvements could unlock value.
    """
    if not financial_line_items:
        return {"metrics": {}, "notes": ["Insufficient data for activism potential."]}

    # Check revenue growth vs. operating margin
    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    op_margins = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]

    if len(revenues) < 2 or not op_margins:
        return {
            "metrics": {},
            "notes": ["Not enough data to assess activism potential (need multi-year revenue + margins)."],
        }

    initial, final = revenues[-1], revenues[0]
    revenue_growth = (final - initial) / abs(initial) if initial else None
    avg_margin = sum(op_margins) / len(op_margins)

    metrics = {
        "revenue_growth_pct": revenue_growth * 100 if revenue_growth is not None else None,
        "average_operating_margin_pct": avg_margin * 100,
    }
    activism_leverage = revenue_growth is not None and revenue_growth > 0.15 and avg_margin < 0.10
    metrics["activism_candidate"] = activism_leverage

    if activism_leverage:
        notes = [
            f"Revenue growth ~{revenue_growth*100:.1f}% with average margin {avg_margin*100:.1f}% — operational upside may be accessible via activism."
        ]
    else:
        notes = ["No clear sign of activism leverage (either growth is muted or margins already acceptable)."]

    return {"metrics": metrics, "notes": notes}


def analyze_valuation(financial_line_items: list, market_cap: float) -> dict:
    """
    Ackman invests in companies trading at a discount to intrinsic value.
    Uses a simplified DCF with FCF as a proxy, plus margin of safety analysis.
    """
    if not financial_line_items or market_cap is None:
        return {
            "metrics": {"market_cap": market_cap},
            "notes": ["Insufficient data to perform valuation."],
        }

    # Since financial_line_items are in descending order (newest first),
    # the most recent period is the first element
    latest = financial_line_items[0]
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0

    if fcf <= 0:
        return {
            "metrics": {
                "market_cap": market_cap,
                "free_cash_flow": fcf,
                "intrinsic_value": None,
                "margin_of_safety_pct": None,
            },
            "notes": [f"No positive FCF for valuation; latest free cash flow = {fcf}."],
        }

    # Basic DCF assumptions
    growth_rate = 0.06
    discount_rate = 0.10
    terminal_multiple = 15
    projection_years = 5

    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv

    # Terminal Value
    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)

    intrinsic_value = present_value + terminal_value
    margin_of_safety = (intrinsic_value - market_cap) / market_cap

    metrics = {
        "market_cap": market_cap,
        "free_cash_flow": fcf,
        "discount_rate": discount_rate,
        "growth_rate": growth_rate,
        "terminal_multiple": terminal_multiple,
        "intrinsic_value": intrinsic_value,
        "margin_of_safety_pct": margin_of_safety * 100,
    }
    notes = [
        f"Intrinsic value estimate ~{intrinsic_value:,.2f} vs. market cap ~{market_cap:,.2f}.",
        f"Margin of safety: {margin_of_safety*100:.2f}%.",
    ]

    return {"metrics": metrics, "notes": notes}


def generate_ackman_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str,
) -> BillAckmanSignal:
    """
    Generates investment decisions in the style of Bill Ackman.
    Includes more explicit references to brand strength, activism potential,
    catalysts, and management changes in the system prompt.
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Bill Ackman AI agent, making investment decisions using his principles:

            1. Seek high-quality businesses with durable competitive advantages (moats), often in well-known consumer or service brands.
            2. Prioritize consistent free cash flow and long-duration growth potential.
            3. Demand disciplined capital allocation (reasonable leverage, buybacks, dividends, reinvestment).
            4. Valuation matters: pursue intrinsic value with a measurable margin of safety.
            5. Consider activism where management or operational improvements can unlock substantial upside.
            6. Concentrate capital in a few high-conviction ideas, but only where activism has teeth.

            Guidance for this portfolio context:
            - Compare the ticker against the other names in the batch and call out where capital should migrate if a peer offers better upside.
            - Explicitly score activism feasibility; mega-cap franchises (>$300B market cap) face steep governance hurdles, so conviction should fade unless there is exceptional mispricing and visible catalysts.
            - When quality is strong but valuation is stretched or activism leverage is weak, lean neutral/bearish and articulate the overvaluation risk.
            - Quantify free cash flow trends, leverage, and margin trajectories; cite numbers instead of platitudes.
            - Highlight catalysts (operational, balance-sheet, governance) that a Pershing Square-style campaign could accelerate.
            - Keep the tone analytical, direct, and willing to critique entrenched management.

            Return your final recommendation (signal: bullish, neutral, or bearish) with a 0-100 confidence and a thorough reasoning section.
            """,
            ),
            (
                "human",
                """Based on the following analysis, create an Ackman-style investment signal.

            Analysis Data for {ticker}:
            {analysis_data}

            Return your output in strictly valid JSON:
            {{
              "signal": "bullish" | "bearish" | "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    dump_dir = os.getenv("PROMPT_DUMP_DIR")
    if dump_dir:
        try:
            base = Path(dump_dir)
            base.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dump_path = base / f"{agent_id}_{ticker}_{timestamp}.txt"
            with dump_path.open("w", encoding="utf-8") as handle:
                handle.write(f"agent: {agent_id}\n")
                handle.write(f"ticker: {ticker}\n")
                handle.write("prompt:\n")
                handle.write(prompt.to_string())  # type: ignore[attr-defined]
        except Exception:
            pass

    def create_default_bill_ackman_signal():
        return BillAckmanSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=BillAckmanSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_bill_ackman_signal,
    )


async def generate_ackman_output_async(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str,
) -> BillAckmanSignal:
    """Async counterpart to generate_ackman_output."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Bill Ackman AI agent, making investment decisions using his principles:

            1. Seek high-quality businesses with durable competitive advantages (moats), often in well-known consumer or service brands.
            2. Prioritize consistent free cash flow and long-duration growth potential.
            3. Demand disciplined capital allocation (reasonable leverage, buybacks, dividends, reinvestment).
            4. Valuation matters: pursue intrinsic value with a measurable margin of safety.
            5. Consider activism where management or operational improvements can unlock substantial upside.
            6. Concentrate capital in a few high-conviction ideas, but only where activism has teeth.

            Guidance for this portfolio context:
            - Compare the ticker against the other names in the batch and call out where capital should migrate if a peer offers better upside.
            - Explicitly score activism feasibility; mega-cap franchises (>$300B market cap) face steep governance hurdles, so conviction should fade unless there is exceptional mispricing and visible catalysts.
            - When quality is strong but valuation is stretched or activism leverage is weak, lean neutral/bearish and articulate the overvaluation risk.
            - Quantify free cash flow trends, leverage, and margin trajectories; cite numbers instead of platitudes.
            - Highlight catalysts (operational, balance-sheet, governance) that a Pershing Square-style campaign could accelerate.
            - Keep the tone analytical, direct, and willing to critique entrenched management.

            Return your final recommendation (signal: bullish, neutral, or bearish) with a 0-100 confidence and a thorough reasoning section.
            """,
            ),
            (
                "human",
                """Based on the following analysis, create an Ackman-style investment signal.

            Analysis Data for {ticker}:
            {analysis_data}

            Return your output in strictly valid JSON:
            {{
              "signal": "bullish" | "bearish" | "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    dump_dir = os.getenv("PROMPT_DUMP_DIR")
    if dump_dir:
        try:
            base = Path(dump_dir)
            base.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dump_path = base / f"{agent_id}_{ticker}_{timestamp}.txt"
            with dump_path.open("w", encoding="utf-8") as handle:
                handle.write(f"agent: {agent_id}\n")
                handle.write(f"ticker: {ticker}\n")
                handle.write("prompt:\n")
                handle.write(prompt.to_string())  # type: ignore[attr-defined]
        except Exception:
            pass

    def create_default_bill_ackman_signal():
        return BillAckmanSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return await async_call_llm(
        prompt=prompt,
        pydantic_model=BillAckmanSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_bill_ackman_signal,
    )
