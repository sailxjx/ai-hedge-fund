import json
from typing import Any

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


def _format_number(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "-"
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{float_value:.{decimals}f}"


def _format_percent(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "-"
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{float_value * 100:.{decimals}f}%"


def _build_summary_table(analysis_data: dict[str, dict[str, Any]]) -> str:
    headers = ["Ticker", "Rev Growth", "Gross Margin", "MoS", "R&D Intensity", "FCF Consistency"]
    col_widths = [len(h) for h in headers]
    rows: list[list[str]] = []

    for ticker, payload in analysis_data.items():
        disruptive = payload.get("disruptive_analysis", {})
        innovation = payload.get("innovation_analysis", {})
        valuation = payload.get("valuation_analysis", {})
        context = payload.get("context", {})

        rev_growth = disruptive.get("metrics", {}).get("latest_revenue_growth")
        gross_margin = disruptive.get("metrics", {}).get("gross_margin_latest")
        mos = valuation.get("metrics", {}).get("margin_of_safety")
        rd_intensity = disruptive.get("metrics", {}).get("rd_intensity")
        fcf_ratio = innovation.get("metrics", {}).get("positive_fcf_ratio")

        row = [
            ticker,
            _format_percent(rev_growth),
            _format_percent(gross_margin),
            _format_percent(mos, decimals=0),
            _format_percent(rd_intensity),
            _format_percent(fcf_ratio),
        ]
        col_widths = [max(col_widths[idx], len(value)) for idx, value in enumerate(row)]
        rows.append(row)

    def fmt(row_values: list[str]) -> str:
        return " | ".join(row_values[idx].ljust(col_widths[idx]) for idx in range(len(headers)))

    header_line = fmt(headers)
    divider = "-+-".join("-" * width for width in col_widths)
    body = [fmt(row) for row in rows]
    return "\n".join([header_line, divider, *body])


def _build_portfolio_snapshot(portfolio: dict[str, Any] | None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    if not isinstance(portfolio, dict):
        return snapshot

    for key in ("cash", "gross_exposure", "net_exposure", "leverage"):
        value = portfolio.get(key)
        if value is not None:
            try:
                snapshot[key] = float(value)
            except (TypeError, ValueError):
                continue

    positions = portfolio.get("positions") or {}
    if isinstance(positions, dict):
        total_long = sum(int((pos or {}).get("long", 0) or 0) for pos in positions.values())
        total_short = sum(int((pos or {}).get("short", 0) or 0) for pos in positions.values())
        snapshot["positions_count"] = len(positions)
        snapshot["total_long_shares"] = total_long
        snapshot["total_short_shares"] = total_short

    return snapshot


class CathieWoodSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def cathie_wood_agent(state: AgentState, agent_id: str = "cathie_wood_agent"):
    """
    Analyzes stocks using Cathie Wood's investing principles and LLM reasoning.
    1. Prioritizes companies with breakthrough technologies or business models
    2. Focuses on industries with rapid adoption curves and massive TAM (Total Addressable Market).
    3. Invests mostly in AI, robotics, genomic sequencing, fintech, and blockchain.
    4. Willing to endure short-term volatility for long-term gains.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    cw_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        # Request multiple periods of data (annual or TTM) for a more robust view.
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "gross_margin",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                "research_and_development",
                "capital_expenditure",
                "operating_expense",
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing disruptive potential")
        disruptive_analysis = analyze_disruptive_potential(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing innovation-driven growth")
        innovation_analysis = analyze_innovation_growth(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating valuation & high-growth scenario")
        valuation_analysis = analyze_cathie_wood_valuation(financial_line_items, market_cap)

        analysis_data[ticker] = {
            "disruptive_analysis": disruptive_analysis,
            "innovation_analysis": innovation_analysis,
            "valuation_analysis": valuation_analysis,
            "context": {
                "market_cap": market_cap,
                "periods_analyzed": len(financial_line_items),
                "latest_financial_metrics": metrics[0].model_dump() if metrics else None,
            },
        }

        progress.update_status(agent_id, ticker, "Prepared analysis context")

    summary_table = _build_summary_table(analysis_data)
    portfolio_snapshot = _build_portfolio_snapshot(state["data"].get("portfolio"))

    analysis_dataset = {
        "per_ticker": analysis_data,
        "summary_table": summary_table,
        "portfolio_snapshot": portfolio_snapshot,
    }

    cw_analysis.clear()

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Generating Cathie Wood analysis")
        cw_output = generate_cathie_wood_output(
            ticker=ticker,
            analysis_dataset=analysis_dataset,
            state=state,
            agent_id=agent_id,
        )

        cw_analysis[ticker] = {"signal": cw_output.signal, "confidence": cw_output.confidence, "reasoning": cw_output.reasoning}

        progress.update_status(agent_id, ticker, "Done", analysis=cw_output.reasoning)

    message = HumanMessage(content=json.dumps(cw_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(cw_analysis, agent_id)

    state["data"]["analyst_signals"][agent_id] = cw_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


async def cathie_wood_agent_async(state: AgentState, agent_id: str = "cathie_wood_agent"):
    """
    Analyzes stocks using Cathie Wood's investing principles and LLM reasoning.
    1. Prioritizes companies with breakthrough technologies or business models
    2. Focuses on industries with rapid adoption curves and massive TAM (Total Addressable Market).
    3. Invests mostly in AI, robotics, genomic sequencing, fintech, and blockchain.
    4. Willing to endure short-term volatility for long-term gains.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    cw_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = await get_financial_metrics_async(ticker, end_date, period="annual", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        # Request multiple periods of data (annual or TTM) for a more robust view.
        financial_line_items = await search_line_items_async(
            ticker,
            [
                "revenue",
                "gross_margin",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                "research_and_development",
                "capital_expenditure",
                "operating_expense",
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = await get_market_cap_async(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing disruptive potential")
        disruptive_analysis = analyze_disruptive_potential(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing innovation-driven growth")
        innovation_analysis = analyze_innovation_growth(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating valuation & high-growth scenario")
        valuation_analysis = analyze_cathie_wood_valuation(financial_line_items, market_cap)

        analysis_data[ticker] = {
            "disruptive_analysis": disruptive_analysis,
            "innovation_analysis": innovation_analysis,
            "valuation_analysis": valuation_analysis,
            "context": {
                "market_cap": market_cap,
                "periods_analyzed": len(financial_line_items),
                "latest_financial_metrics": metrics[0].model_dump() if metrics else None,
            },
        }

        progress.update_status(agent_id, ticker, "Prepared analysis context")

    summary_table = _build_summary_table(analysis_data)
    portfolio_snapshot = _build_portfolio_snapshot(state["data"].get("portfolio"))

    analysis_dataset = {
        "per_ticker": analysis_data,
        "summary_table": summary_table,
        "portfolio_snapshot": portfolio_snapshot,
    }

    cw_analysis.clear()

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Generating Cathie Wood analysis")
        cw_output = await generate_cathie_wood_output_async(
            ticker=ticker,
            analysis_dataset=analysis_dataset,
            state=state,
            agent_id=agent_id,
        )

        cw_analysis[ticker] = {"signal": cw_output.signal, "confidence": cw_output.confidence, "reasoning": cw_output.reasoning}

        progress.update_status(agent_id, ticker, "Done", analysis=cw_output.reasoning)

    message = HumanMessage(content=json.dumps(cw_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(cw_analysis, agent_id)

    await update_analyst_signals_async(state, agent_id, cw_analysis)

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def analyze_disruptive_potential(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze whether the company has disruptive products, technology, or business model.
    Evaluates multiple dimensions of disruptive potential:
    1. Revenue Growth Acceleration - indicates market adoption
    2. R&D Intensity - shows innovation investment
    3. Gross Margin Trends - suggests pricing power and scalability
    4. Operating Leverage - demonstrates business model efficiency
    5. Market Share Dynamics - indicates competitive position
    """
    notes: list[str] = []
    metrics_payload: dict[str, float | None] = {
        "latest_revenue_growth": None,
        "revenue_growth_acceleration": None,
        "gross_margin_latest": None,
        "gross_margin_trend": None,
        "operating_leverage_gap": None,
        "rd_intensity": None,
    }
    flags: dict[str, Any] = {
        "revenue_accelerating": None,
        "gross_margin_expanding": None,
        "operating_leverage_positive": None,
        "rd_intensity_high": None,
    }

    if not metrics or not financial_line_items:
        notes.append("Insufficient data to analyze disruptive potential.")
        return {"metrics": metrics_payload, "flags": flags, "notes": notes}

    # 1. Revenue Growth Analysis - Check for accelerating growth
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    if len(revenues) >= 3:  # Need at least 3 periods to check acceleration
        growth_rates = []
        for i in range(len(revenues) - 1):
            if revenues[i] and revenues[i + 1]:
                growth_rate = (revenues[i] - revenues[i + 1]) / abs(revenues[i + 1]) if revenues[i + 1] != 0 else 0
                growth_rates.append(growth_rate)

        # Check if growth is accelerating (first growth rate higher than last, since they're in reverse order)
        if len(growth_rates) >= 2:
            acceleration = growth_rates[0] - growth_rates[-1]
            metrics_payload["revenue_growth_acceleration"] = float(acceleration)
            if acceleration > 0:
                flags["revenue_accelerating"] = True
                notes.append(f"Revenue growth accelerating: {growth_rates[0]*100:.1f}% vs {growth_rates[-1]*100:.1f}%.")
            else:
                flags["revenue_accelerating"] = False

        # Check absolute growth rate (most recent growth rate is at index 0)
        latest_growth = growth_rates[0] if growth_rates else 0
        metrics_payload["latest_revenue_growth"] = float(latest_growth)
        if latest_growth:
            notes.append(f"Latest revenue growth: {latest_growth*100:.1f}%.")
    else:
        notes.append("Insufficient revenue data for growth analysis.")

    # 2. Gross Margin Analysis - Check for expanding margins
    gross_margins = [item.gross_margin for item in financial_line_items if hasattr(item, "gross_margin") and item.gross_margin is not None]
    if len(gross_margins) >= 2:
        margin_trend = gross_margins[0] - gross_margins[-1]
        metrics_payload["gross_margin_trend"] = float(margin_trend)
        flags["gross_margin_expanding"] = margin_trend > 0
        if margin_trend > 0:
            notes.append(f"Gross margin expanding by {margin_trend*100:.1f}%.")

        # Check absolute margin level (most recent margin is at index 0)
        metrics_payload["gross_margin_latest"] = float(gross_margins[0])
        notes.append(f"Latest gross margin: {gross_margins[0]*100:.1f}%.")
    else:
        notes.append("Insufficient gross margin data.")

    # 3. Operating Leverage Analysis
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    operating_expenses = [item.operating_expense for item in financial_line_items if hasattr(item, "operating_expense") and item.operating_expense]

    if len(revenues) >= 2 and len(operating_expenses) >= 2:
        rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
        opex_growth = (operating_expenses[0] - operating_expenses[-1]) / abs(operating_expenses[-1])

        leverage_gap = float(rev_growth - opex_growth)
        metrics_payload["operating_leverage_gap"] = leverage_gap
        if leverage_gap > 0:
            flags["operating_leverage_positive"] = True
            notes.append("Positive operating leverage: revenue growth outpaces operating expense growth.")
        else:
            flags["operating_leverage_positive"] = False
    else:
        notes.append("Insufficient data for operating leverage analysis.")

    # 4. R&D Investment Analysis
    rd_expenses = [item.research_and_development for item in financial_line_items if hasattr(item, "research_and_development") and item.research_and_development is not None]
    if rd_expenses and revenues:
        rd_intensity = rd_expenses[0] / revenues[0]
        metrics_payload["rd_intensity"] = float(rd_intensity)
        flags["rd_intensity_high"] = rd_intensity >= 0.15
        notes.append(f"R&D intensity: {rd_intensity*100:.1f}% of revenue.")
    else:
        notes.append("No R&D data available.")

    return {"metrics": metrics_payload, "flags": flags, "notes": notes}


def analyze_innovation_growth(metrics: list, financial_line_items: list) -> dict:
    """
    Evaluate the company's commitment to innovation and potential for exponential growth.
    Analyzes multiple dimensions:
    1. R&D Investment Trends - measures commitment to innovation
    2. Free Cash Flow Generation - indicates ability to fund innovation
    3. Operating Efficiency - shows scalability of innovation
    4. Capital Allocation - reveals innovation-focused management
    5. Growth Reinvestment - demonstrates commitment to future growth
    """
    notes: list[str] = []
    metrics_payload: dict[str, float | None] = {
        "rd_growth": None,
        "rd_intensity_change": None,
        "positive_fcf_ratio": None,
        "fcf_growth": None,
        "operating_margin_latest": None,
        "operating_margin_trend": None,
        "capex_intensity": None,
        "capex_growth": None,
        "dividend_payout_ratio": None,
    }
    flags: dict[str, Any] = {
        "rd_intensity_increasing": None,
        "consistent_positive_fcf": None,
        "operating_margin_improving": None,
        "reinvesting_growth": None,
    }

    if not metrics or not financial_line_items:
        notes.append("Insufficient data to analyze innovation-driven growth.")
        return {"metrics": metrics_payload, "flags": flags, "notes": notes}

    # 1. R&D Investment Trends
    rd_expenses = [item.research_and_development for item in financial_line_items if hasattr(item, "research_and_development") and item.research_and_development]
    revenues = [item.revenue for item in financial_line_items if item.revenue]

    if rd_expenses and revenues and len(rd_expenses) >= 2:
        rd_growth = (rd_expenses[0] - rd_expenses[-1]) / abs(rd_expenses[-1]) if rd_expenses[-1] != 0 else 0
        metrics_payload["rd_growth"] = float(rd_growth)
        notes.append(f"R&D investment growth: {rd_growth*100:.1f}%.")

        # Check R&D intensity trend (corrected for reverse chronological order)
        rd_intensity_start = rd_expenses[-1] / revenues[-1]
        rd_intensity_end = rd_expenses[0] / revenues[0]
        intensity_change = rd_intensity_end - rd_intensity_start
        metrics_payload["rd_intensity_change"] = float(intensity_change)
        flags["rd_intensity_increasing"] = intensity_change > 0
        if intensity_change > 0:
            notes.append(f"R&D intensity rising: {rd_intensity_end*100:.1f}% vs {rd_intensity_start*100:.1f}%.")
    else:
        notes.append("Insufficient R&D data for trend analysis.")

    # 2. Free Cash Flow Analysis
    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow]
    if fcf_vals and len(fcf_vals) >= 2:
        fcf_growth = (fcf_vals[0] - fcf_vals[-1]) / abs(fcf_vals[-1])
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)

        metrics_payload["fcf_growth"] = float(fcf_growth)
        metrics_payload["positive_fcf_ratio"] = positive_fcf_count / len(fcf_vals)
        flags["consistent_positive_fcf"] = positive_fcf_count == len(fcf_vals)
        notes.append(f"Positive FCF ratio: {positive_fcf_count}/{len(fcf_vals)} periods; growth {fcf_growth*100:.1f}%.")
    else:
        notes.append("Insufficient FCF data for analysis.")

    # 3. Operating Efficiency Analysis
    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin]
    if op_margin_vals and len(op_margin_vals) >= 2:
        margin_trend = op_margin_vals[0] - op_margin_vals[-1]
        metrics_payload["operating_margin_latest"] = float(op_margin_vals[0])
        metrics_payload["operating_margin_trend"] = float(margin_trend)
        flags["operating_margin_improving"] = margin_trend > 0
        notes.append(f"Operating margin latest {op_margin_vals[0]*100:.1f}% (trend {margin_trend*100:.1f}%).")
    else:
        notes.append("Insufficient operating margin data.")

    # 4. Capital Allocation Analysis
    capex = [item.capital_expenditure for item in financial_line_items if hasattr(item, "capital_expenditure") and item.capital_expenditure]
    if capex and revenues and len(capex) >= 2:
        capex_intensity = abs(capex[0]) / revenues[0]
        capex_growth = (abs(capex[0]) - abs(capex[-1])) / abs(capex[-1]) if capex[-1] != 0 else 0

        metrics_payload["capex_intensity"] = float(capex_intensity)
        metrics_payload["capex_growth"] = float(capex_growth)
        notes.append(f"CAPEX intensity {capex_intensity*100:.1f}% with growth {capex_growth*100:.1f}%.")
    else:
        notes.append("Insufficient CAPEX data.")

    # 5. Growth Reinvestment Analysis
    dividends = [item.dividends_and_other_cash_distributions for item in financial_line_items if hasattr(item, "dividends_and_other_cash_distributions") and item.dividends_and_other_cash_distributions]
    if dividends and fcf_vals:
        latest_payout_ratio = dividends[0] / fcf_vals[0] if fcf_vals[0] != 0 else 1
        metrics_payload["dividend_payout_ratio"] = float(latest_payout_ratio)
        flags["reinvesting_growth"] = latest_payout_ratio < 0.4
        notes.append(f"Dividend payout ratio {latest_payout_ratio*100:.1f}%.")
    else:
        notes.append("Insufficient dividend data.")

    return {"metrics": metrics_payload, "flags": flags, "notes": notes}


def analyze_cathie_wood_valuation(financial_line_items: list, market_cap: float) -> dict:
    """
    Cathie Wood often focuses on long-term exponential growth potential. We can do
    a simplified approach looking for a large total addressable market (TAM) and the
    company's ability to capture a sizable portion.
    """
    notes: list[str] = []
    metrics_payload: dict[str, float | None] = {
        "intrinsic_value": None,
        "margin_of_safety": None,
        "growth_rate_assumption": 0.20,
        "discount_rate": 0.15,
        "terminal_multiple": 25.0,
        "projection_years": 5,
        "market_cap": float(market_cap) if market_cap is not None else None,
        "latest_free_cash_flow": None,
    }

    if not financial_line_items or market_cap is None:
        notes.append("Insufficient data for valuation.")
        return {"metrics": metrics_payload, "notes": notes}

    latest = financial_line_items[0]
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0
    metrics_payload["latest_free_cash_flow"] = float(fcf)

    if fcf <= 0:
        notes.append(f"No positive free cash flow; valuation requires scenario adjustments (FCF={fcf}).")
        return {"metrics": metrics_payload, "notes": notes}

    # Instead of a standard DCF, let's assume a higher growth rate for an innovative company.
    # Example values:
    growth_rate = metrics_payload["growth_rate_assumption"]
    discount_rate = metrics_payload["discount_rate"]
    terminal_multiple = metrics_payload["terminal_multiple"]
    projection_years = int(metrics_payload["projection_years"])

    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv

    # Terminal Value
    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value

    margin_of_safety = (intrinsic_value - market_cap) / market_cap

    metrics_payload["intrinsic_value"] = float(intrinsic_value)
    metrics_payload["margin_of_safety"] = float(margin_of_safety)

    notes.append(f"Intrinsic value estimate ${intrinsic_value:,.2f} vs market cap ${market_cap:,.2f}.")
    notes.append(f"Margin of safety {margin_of_safety:.2%} using growth {growth_rate:.0%} and discount {discount_rate:.0%}.")

    return {"metrics": metrics_payload, "notes": notes}


def generate_cathie_wood_output(
    ticker: str,
    analysis_dataset: dict[str, Any],
    state: AgentState,
    agent_id: str = "cathie_wood_agent",
) -> CathieWoodSignal:
    """
    Generates investment decisions in the style of Cathie Wood.
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Cathie Wood AI agent, making investment decisions using her principles:

            1. Seek companies leveraging disruptive innovation.
            2. Emphasize exponential growth potential, large TAM.
            3. Focus on technology, healthcare, or other future-facing sectors.
            4. Consider multi-year time horizons for potential breakthroughs.
            5. Accept higher volatility in pursuit of high returns.
            6. Evaluate management's vision and ability to invest in R&D.
            7. Use the provided summary_table to compare momentum across the batch and note current portfolio exposure before recommending moves.

            Rules:
            - Identify disruptive or breakthrough technology.
            - Evaluate strong potential for multi-year revenue growth.
            - Check if the company can scale effectively in a large market.
            - Use a growth-biased valuation approach.
            - Provide a data-driven recommendation (bullish, bearish, or neutral).
            
            When providing your reasoning, be thorough and specific by:
            1. Identifying the specific disruptive technologies/innovations the company is leveraging
            2. Highlighting growth metrics that indicate exponential potential (revenue acceleration, expanding TAM)
            3. Discussing the long-term vision and transformative potential over 5+ year horizons
            4. Explaining how the company might disrupt traditional industries or create new markets
            5. Addressing R&D investment and innovation pipeline that could drive future growth
            6. Using Cathie Wood's optimistic, future-focused, and conviction-driven voice
            
            For example, if bullish: "The company's AI-driven platform is transforming the $500B healthcare analytics market, with evidence of platform adoption accelerating from 40% to 65% YoY. Their R&D investments of 22% of revenue are creating a technological moat that positions them to capture a significant share of this expanding market. The current valuation doesn't reflect the exponential growth trajectory we expect as..."
            For example, if bearish: "While operating in the genomics space, the company lacks truly disruptive technology and is merely incrementally improving existing techniques. R&D spending at only 8% of revenue signals insufficient investment in breakthrough innovation. With revenue growth slowing from 45% to 20% YoY, there's limited evidence of the exponential adoption curve we look for in transformative companies..."
            """,
            ),
            (
                "human",
                """Based on the following analysis, create a Cathie Wood-style investment signal.

            Analysis Dataset for {ticker} (includes per-ticker metrics, multi-name summary table, and portfolio snapshot):
            {analysis_dataset}

            Return the trading signal in this JSON format:
            {{
              "signal": "bullish/bearish/neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_dataset": json.dumps(analysis_dataset, indent=2), "ticker": ticker})

    def create_default_cathie_wood_signal():
        return CathieWoodSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=CathieWoodSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_cathie_wood_signal,
    )


async def generate_cathie_wood_output_async(
    ticker: str,
    analysis_dataset: dict[str, Any],
    state: AgentState,
    agent_id: str = "cathie_wood_agent",
) -> CathieWoodSignal:
    """Async variant of generate_cathie_wood_output."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Cathie Wood AI agent, making investment decisions using her principles:

            1. Seek companies leveraging disruptive innovation.
            2. Emphasize exponential growth potential, large TAM.
            3. Focus on technology, healthcare, or other future-facing sectors.
            4. Consider multi-year time horizons for potential breakthroughs.
            5. Accept higher volatility in pursuit of high returns.
            6. Evaluate management's vision and ability to invest in R&D.
            7. Use the provided summary_table to compare momentum across the batch and note current portfolio exposure before recommending moves.
            """,
            ),
            (
                "human",
                """Based on the following analysis, create a Cathie Wood-style investment signal.

            Analysis Dataset for {ticker} (includes per-ticker metrics, multi-name summary table, and portfolio snapshot):
            {analysis_dataset}

            Return the trading signal in this JSON format:
            {{
              "signal": "bullish/bearish/neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_dataset": json.dumps(analysis_dataset, indent=2), "ticker": ticker})

    def create_default_cathie_wood_signal():
        return CathieWoodSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return await async_call_llm(
        prompt=prompt,
        pydantic_model=CathieWoodSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_cathie_wood_signal,
    )


# source: https://ark-invest.com
