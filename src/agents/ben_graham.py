import json
import math

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


class BenGrahamSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def ben_graham_agent(state: AgentState, agent_id: str = "ben_graham_agent"):
    """
    Analyzes stocks using Benjamin Graham's classic value-investing principles:
    1. Earnings stability over multiple years.
    2. Solid financial strength (low debt, adequate liquidity).
    3. Discount to intrinsic value (e.g. Graham Number or net-net).
    4. Adequate margin of safety.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    analysis_data = {}
    graham_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(ticker, ["earnings_per_share", "revenue", "net_income", "book_value_per_share", "total_assets", "total_liabilities", "current_assets", "current_liabilities", "dividends_and_other_cash_distributions", "outstanding_shares"], end_date, period="annual", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        # Perform sub-analyses
        progress.update_status(agent_id, ticker, "Analyzing earnings stability")
        earnings_analysis = analyze_earnings_stability(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing financial strength")
        strength_analysis = analyze_financial_strength(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing Graham valuation")
        valuation_analysis = analyze_valuation_graham(financial_line_items, market_cap)

        analysis_data[ticker] = {
            "earnings_stability": earnings_analysis,
            "financial_strength": strength_analysis,
            "valuation": valuation_analysis,
            "context": {
                "market_cap": market_cap,
                "latest_financial_metrics": metrics[0].model_dump() if metrics else None,
                "periods_analyzed": len(financial_line_items),
            },
        }

        progress.update_status(agent_id, ticker, "Generating Ben Graham analysis")
        graham_output = generate_graham_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        graham_analysis[ticker] = {"signal": graham_output.signal, "confidence": graham_output.confidence, "reasoning": graham_output.reasoning}

        progress.update_status(agent_id, ticker, "Done", analysis=graham_output.reasoning)

    # Wrap results in a single message for the chain
    message = HumanMessage(content=json.dumps(graham_analysis), name=agent_id)

    # Optionally display reasoning
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(graham_analysis, "Ben Graham Agent")

    # Store signals in the overall state
    state["data"]["analyst_signals"][agent_id] = graham_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


async def ben_graham_agent_async(state: AgentState, agent_id: str = "ben_graham_agent"):
    """
    Analyzes stocks using Benjamin Graham's classic value-investing principles:
    1. Earnings stability over multiple years.
    2. Solid financial strength (low debt, adequate liquidity).
    3. Discount to intrinsic value (e.g. Graham Number or net-net).
    4. Adequate margin of safety.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    analysis_data = {}
    graham_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = await get_financial_metrics_async(ticker, end_date, period="annual", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = await search_line_items_async(ticker, ["earnings_per_share", "revenue", "net_income", "book_value_per_share", "total_assets", "total_liabilities", "current_assets", "current_liabilities", "dividends_and_other_cash_distributions", "outstanding_shares"], end_date, period="annual", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = await get_market_cap_async(ticker, end_date, api_key=api_key)

        # Perform sub-analyses
        progress.update_status(agent_id, ticker, "Analyzing earnings stability")
        earnings_analysis = analyze_earnings_stability(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing financial strength")
        strength_analysis = analyze_financial_strength(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing Graham valuation")
        valuation_analysis = analyze_valuation_graham(financial_line_items, market_cap)

        analysis_data[ticker] = {
            "earnings_stability": earnings_analysis,
            "financial_strength": strength_analysis,
            "valuation": valuation_analysis,
            "context": {
                "market_cap": market_cap,
                "latest_financial_metrics": metrics[0].model_dump() if metrics else None,
                "periods_analyzed": len(financial_line_items),
            },
        }

        progress.update_status(agent_id, ticker, "Generating Ben Graham analysis")
        graham_output = await generate_graham_output_async(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        graham_analysis[ticker] = {"signal": graham_output.signal, "confidence": graham_output.confidence, "reasoning": graham_output.reasoning}

        progress.update_status(agent_id, ticker, "Done", analysis=graham_output.reasoning)

    # Wrap results in a single message for the chain
    message = HumanMessage(content=json.dumps(graham_analysis), name=agent_id)

    # Optionally display reasoning
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(graham_analysis, "Ben Graham Agent")

    # Store signals in the overall state
    await update_analyst_signals_async(state, agent_id, graham_analysis)

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def analyze_earnings_stability(metrics: list, financial_line_items: list) -> dict:
    """
    Graham wants at least several years of consistently positive earnings (ideally 5+).
    We'll check:
    1. Number of years with positive EPS.
    2. Growth in EPS from first to last period.
    """
    notes: list[str] = []
    observations: dict[str, float | int | None] = {}

    if not metrics or not financial_line_items:
        notes.append("Insufficient data for earnings stability analysis.")
        return {"metrics": observations, "notes": notes}

    eps_vals = []
    for item in financial_line_items:
        if item.earnings_per_share is not None:
            eps_vals.append(item.earnings_per_share)

    if len(eps_vals) < 2:
        notes.append("Not enough multi-year EPS data.")
        observations["total_eps_periods"] = len(eps_vals)
        observations["positive_eps_years"] = sum(1 for e in eps_vals if e > 0)
        observations["latest_eps"] = eps_vals[0] if eps_vals else None
        observations["earliest_eps"] = eps_vals[-1] if eps_vals else None
        return {"metrics": observations, "notes": notes}

    # 1. Consistently positive EPS
    positive_eps_years = sum(1 for e in eps_vals if e > 0)
    total_eps_years = len(eps_vals)
    observations["total_eps_periods"] = total_eps_years
    observations["positive_eps_years"] = positive_eps_years
    observations["positive_eps_ratio"] = positive_eps_years / total_eps_years if total_eps_years else None

    if positive_eps_years == total_eps_years:
        notes.append("EPS was positive in all available periods.")
    elif positive_eps_years >= (total_eps_years * 0.8):
        notes.append("EPS was positive in most periods.")
    else:
        notes.append("EPS was negative in multiple periods.")

    # 2. EPS growth from earliest to latest
    latest_eps = eps_vals[0]
    earliest_eps = eps_vals[-1]
    observations["latest_eps"] = latest_eps
    observations["earliest_eps"] = earliest_eps

    if earliest_eps not in (None, 0):
        eps_growth = (latest_eps - earliest_eps) / abs(earliest_eps)
        observations["eps_growth_pct"] = eps_growth * 100
        if eps_growth > 0:
            notes.append(f"EPS grew from {earliest_eps:.2f} to {latest_eps:.2f}.")
        else:
            notes.append(f"EPS declined or was flat (earliest {earliest_eps:.2f} → latest {latest_eps:.2f}).")
    else:
        observations["eps_growth_pct"] = None
        notes.append("Unable to compute EPS growth (earliest EPS near zero).")

    return {"metrics": observations, "notes": notes}


def analyze_financial_strength(financial_line_items: list) -> dict:
    """
    Graham checks liquidity (current ratio >= 2), manageable debt,
    and dividend record (preferably some history of dividends).
    """
    notes: list[str] = []
    observations: dict[str, float | int | None] = {}

    if not financial_line_items:
        notes.append("No data for financial strength analysis.")
        return {"metrics": observations, "notes": notes}

    latest_item = financial_line_items[0]
    total_assets = latest_item.total_assets or 0
    total_liabilities = latest_item.total_liabilities or 0
    current_assets = latest_item.current_assets or 0
    current_liabilities = latest_item.current_liabilities or 0

    # 1. Current ratio
    if current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        observations["current_ratio"] = current_ratio
        if current_ratio >= 2.0:
            notes.append(f"Current ratio = {current_ratio:.2f} (>=2.0: strong).")
        elif current_ratio >= 1.5:
            notes.append(f"Current ratio = {current_ratio:.2f} (moderate liquidity).")
        else:
            notes.append(f"Current ratio = {current_ratio:.2f} (below Graham’s 2.0 threshold).")
    else:
        observations["current_ratio"] = None
        notes.append("Cannot compute current ratio (missing or zero current liabilities).")

    # 2. Debt vs. Assets
    if total_assets > 0:
        debt_ratio = total_liabilities / total_assets
        observations["debt_ratio"] = debt_ratio
        if debt_ratio < 0.5:
            notes.append(f"Debt ratio = {debt_ratio:.2f}, under 0.50 (conservative).")
        elif debt_ratio < 0.8:
            notes.append(f"Debt ratio = {debt_ratio:.2f}, somewhat elevated but manageable.")
        else:
            notes.append(f"Debt ratio = {debt_ratio:.2f}, high relative to Graham’s preference.")
    else:
        observations["debt_ratio"] = None
        notes.append("Cannot compute debt ratio (missing total assets).")

    # 3. Dividend track record
    div_periods = [item.dividends_and_other_cash_distributions for item in financial_line_items if item.dividends_and_other_cash_distributions is not None]
    if div_periods:
        # In many data feeds, dividend outflow is shown as a negative number
        # (money going out to shareholders). We'll consider any negative as 'paid a dividend'.
        div_paid_years = sum(1 for d in div_periods if d < 0)
        observations["dividend_years"] = div_paid_years
        observations["dividend_periods_observed"] = len(div_periods)
        observations["dividend_payment_ratio"] = div_paid_years / len(div_periods) if div_periods else None
        if div_paid_years > 0:
            if div_paid_years >= (len(div_periods) // 2 + 1):
                notes.append("Company paid dividends in the majority of reported years.")
            else:
                notes.append("Company paid dividends in some years but not consistently.")
        else:
            notes.append("Company did not pay dividends in the observed periods.")
    else:
        observations["dividend_years"] = 0
        observations["dividend_periods_observed"] = 0
        observations["dividend_payment_ratio"] = None
        notes.append("No dividend data available to assess payout consistency.")

    return {"metrics": observations, "notes": notes}


def analyze_valuation_graham(financial_line_items: list, market_cap: float) -> dict:
    """
    Core Graham approach to valuation:
    1. Net-Net Check: (Current Assets - Total Liabilities) vs. Market Cap
    2. Graham Number: sqrt(22.5 * EPS * Book Value per Share)
    3. Compare per-share price to Graham Number => margin of safety
    """
    if not financial_line_items or not market_cap or market_cap <= 0:
        return {
            "metrics": {
                "market_cap": market_cap,
            },
            "notes": ["Insufficient data to perform valuation."],
        }

    latest = financial_line_items[0]
    current_assets = latest.current_assets or 0
    total_liabilities = latest.total_liabilities or 0
    book_value_ps = latest.book_value_per_share or 0
    eps = latest.earnings_per_share or 0
    shares_outstanding = latest.outstanding_shares or 0

    details: list[str] = []
    metrics: dict[str, float | None] = {
        "market_cap": market_cap,
        "current_assets": current_assets,
        "total_liabilities": total_liabilities,
        "book_value_per_share": book_value_ps,
        "earnings_per_share": eps,
        "shares_outstanding": shares_outstanding,
    }

    # 1. Net-Net Check
    #   NCAV = Current Assets - Total Liabilities
    #   If NCAV > Market Cap => historically a strong buy signal
    net_current_asset_value = current_assets - total_liabilities
    metrics["net_current_asset_value"] = net_current_asset_value
    if net_current_asset_value > 0 and shares_outstanding > 0:
        net_current_asset_value_per_share = net_current_asset_value / shares_outstanding
        price_per_share = market_cap / shares_outstanding if shares_outstanding else 0
        metrics["ncav_per_share"] = net_current_asset_value_per_share
        metrics["price_per_share"] = price_per_share

        details.append(f"Net Current Asset Value = {net_current_asset_value:,.2f}")
        details.append(f"NCAV Per Share = {net_current_asset_value_per_share:,.2f}")
        details.append(f"Price Per Share = {price_per_share:,.2f}")

        if net_current_asset_value > market_cap:
            details.append("Net-Net: NCAV exceeds market cap (deep value signal).")
        elif price_per_share:
            ratio = net_current_asset_value_per_share / price_per_share
            metrics["ncav_to_price_ratio"] = ratio
            if ratio >= 0.67:
                details.append("NCAV per share is at least 2/3 of price per share (moderate net-net discount).")
            else:
                details.append("NCAV per share below 2/3 of price per share.")
    else:
        details.append("NCAV not exceeding market cap or insufficient data for net-net approach.")

    # 2. Graham Number
    #   GrahamNumber = sqrt(22.5 * EPS * BVPS).
    #   Compare the result to the current price_per_share
    #   If GrahamNumber >> price, indicates undervaluation
    graham_number = None
    if eps > 0 and book_value_ps > 0:
        graham_number = math.sqrt(22.5 * eps * book_value_ps)
        details.append(f"Graham Number = {graham_number:.2f}")
        metrics["graham_number"] = graham_number
    else:
        details.append("Unable to compute Graham Number (EPS or Book Value missing/<=0).")
        metrics["graham_number"] = None

    # 3. Margin of Safety relative to Graham Number
    if graham_number and shares_outstanding > 0:
        current_price = market_cap / shares_outstanding
        metrics["price_per_share"] = current_price
        if current_price > 0:
            margin_of_safety = (graham_number - current_price) / current_price
            metrics["graham_margin_of_safety_pct"] = margin_of_safety * 100
            details.append(f"Margin of Safety (Graham Number) = {margin_of_safety:.2%}")
            if margin_of_safety <= 0:
                details.append("Price near or above the Graham Number (little/no margin of safety).")
        else:
            details.append("Current price is zero or invalid; can't compute margin of safety.")
    else:
        metrics["graham_margin_of_safety_pct"] = None

    return {"metrics": metrics, "notes": details}


def generate_graham_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str,
) -> BenGrahamSignal:
    """
    Generates an investment decision in the style of Benjamin Graham:
    - Value emphasis, margin of safety, net-nets, conservative balance sheet, stable earnings.
    - Return the result in a JSON structure: { signal, confidence, reasoning }.
    """

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Benjamin Graham AI agent, making investment decisions using his principles:
            1. Insist on a margin of safety by buying below intrinsic value (e.g., using Graham Number, net-net).
            2. Emphasize the company's financial strength (low leverage, ample current assets).
            3. Prefer stable earnings over multiple years.
            4. Consider dividend record for extra safety.
            5. Avoid speculative or high-growth assumptions; focus on proven metrics.
            
            When providing your reasoning, be thorough and specific by:
            1. Explaining the key valuation metrics that influenced your decision the most (Graham Number, NCAV, P/E, etc.)
            2. Highlighting the specific financial strength indicators (current ratio, debt levels, etc.)
            3. Referencing the stability or instability of earnings over time
            4. Providing quantitative evidence with precise numbers
            5. Comparing current metrics to Graham's specific thresholds (e.g., "Current ratio of 2.5 exceeds Graham's minimum of 2.0")
            6. Using Benjamin Graham's conservative, analytical voice and style in your explanation
            
            For example, if bullish: "The stock trades at a 35% discount to net current asset value, providing an ample margin of safety. The current ratio of 2.5 and debt-to-equity of 0.3 indicate strong financial position..."
            For example, if bearish: "Despite consistent earnings, the current price of $50 exceeds our calculated Graham Number of $35, offering no margin of safety. Additionally, the current ratio of only 1.2 falls below Graham's preferred 2.0 threshold..."
                        
            Return a rational recommendation: bullish, bearish, or neutral, with a confidence level (0-100) and thorough reasoning.
            """,
            ),
            (
                "human",
                """Based on the following analysis, create a Graham-style investment signal:

            Analysis Data for {ticker}:
            {analysis_data}

            Return JSON exactly in this format:
            {{
              "signal": "bullish" or "bearish" or "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_ben_graham_signal():
        return BenGrahamSignal(signal="neutral", confidence=0.0, reasoning="Error in generating analysis; defaulting to neutral.")

    return call_llm(
        prompt=prompt,
        pydantic_model=BenGrahamSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_ben_graham_signal,
    )


async def generate_graham_output_async(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str,
) -> BenGrahamSignal:
    """Async variant of generate_graham_output leveraging async_call_llm."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Benjamin Graham AI agent, making investment decisions using his principles:
            1. Insist on a margin of safety by buying below intrinsic value (e.g., using Graham Number, net-net).
            2. Emphasize the company's financial strength (low leverage, ample current assets).
            3. Prefer stable earnings over multiple years.
            4. Consider dividend record for extra safety.
            5. Avoid speculative or high-growth assumptions; focus on proven metrics.
            
            When providing your reasoning, be thorough and specific by:
            1. Explaining the key valuation metrics that influenced your decision the most (Graham Number, NCAV, P/E, etc.)
            2. Highlighting the specific financial strength indicators (current ratio, debt levels, etc.)
            3. Referencing the stability or instability of earnings over time
            4. Providing quantitative evidence with precise numbers
            5. Comparing current metrics to Graham's specific thresholds (e.g., \"Current ratio of 2.5 exceeds Graham's minimum of 2.0\")
            6. Using Benjamin Graham's conservative, analytical voice and style in your explanation
            """,
            ),
            (
                "human",
                """Based on the following analysis, create a Graham-style investment signal:

            Analysis Data for {ticker}:
            {analysis_data}

            Return JSON exactly in this format:
            {{
              \"signal\": \"bullish\" or \"bearish\" or \"neutral\",
              \"confidence\": float (0-100),
              \"reasoning\": \"string\"
            }}
            """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_ben_graham_signal():
        return BenGrahamSignal(signal="neutral", confidence=0.0, reasoning="Error in generating analysis; defaulting to neutral.")

    return await async_call_llm(
        prompt=prompt,
        pydantic_model=BenGrahamSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_ben_graham_signal,
    )
