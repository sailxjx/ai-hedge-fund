import json
from typing import Any, Iterable

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import (
    get_company_news,
    get_company_news_async,
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


def _build_summary_table(per_ticker: dict[str, dict[str, Any]]) -> str:
    headers = ["Ticker", "Rev Growth", "EPS Growth", "PEG", "Debt/Equity", "Payout Ratio", "Insider Net"]
    col_widths = [len(h) for h in headers]
    rows: list[list[str]] = []

    for ticker, payload in per_ticker.items():
        growth_metrics = (payload.get("growth_analysis") or {}).get("metrics") or {}
        valuation_metrics = (payload.get("valuation_analysis") or {}).get("metrics") or {}
        fundamentals_metrics = (payload.get("fundamentals_analysis") or {}).get("metrics") or {}
        insider_metrics = (payload.get("insider_activity") or {}).get("metrics") or {}

        row = [
            ticker,
            _format_percent(growth_metrics.get("revenue_growth_cagr")),
            _format_percent(growth_metrics.get("eps_growth_cagr")),
            _format_number(valuation_metrics.get("peg_ratio")),
            _format_number(fundamentals_metrics.get("debt_to_equity")),
            _format_percent(fundamentals_metrics.get("dividend_payout_ratio")),
            _format_number(insider_metrics.get("net_buy_volume")),
        ]
        col_widths = [max(col_widths[idx], len(value)) for idx, value in enumerate(row)]
        rows.append(row)

    def fmt(values: list[str]) -> str:
        return " | ".join(values[idx].ljust(col_widths[idx]) for idx in range(len(headers)))

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


class PeterLynchSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def build_peter_lynch_fallback_signal(
    ticker: str,
    analysis_data: dict[str, Any],
) -> PeterLynchSignal:
    """Create a neutral Peter Lynch signal when the LLM response is unavailable."""

    growth = (analysis_data.get("growth_analysis") or {}).get("notes") or []
    valuation = (analysis_data.get("valuation_analysis") or {}).get("notes") or []
    fundamentals = (analysis_data.get("fundamentals_analysis") or {}).get("notes") or []
    sentiment = (analysis_data.get("sentiment_analysis") or {}).get("notes") or []
    insiders = (analysis_data.get("insider_activity") or {}).get("notes") or []

    def _summarize(lines: Iterable[str]) -> str:
        return "; ".join(list(lines)[:2]) if lines else "No observations captured."

    reasoning = (
        f"LLM output was unavailable for {ticker}; returning a neutral stance while summarizing observations. "
        f"Growth: {_summarize(growth)} | Valuation: {_summarize(valuation)} | Fundamentals: {_summarize(fundamentals)} | "
        f"Sentiment: {_summarize(sentiment)} | Insider Activity: {_summarize(insiders)}."
    )

    return PeterLynchSignal(
        signal="neutral",
        confidence=35.0,
        reasoning=reasoning,
    )


def peter_lynch_agent(state: AgentState, agent_id: str = "peter_lynch_agent"):
    """
    Analyzes stocks using Peter Lynch's investing principles:
      - Invest in what you know (clear, understandable businesses).
      - Growth at a Reasonable Price (GARP), emphasizing the PEG ratio.
      - Look for consistent revenue & EPS increases and manageable debt.
      - Be alert for potential "ten-baggers" (high-growth opportunities).
      - Avoid overly complex or highly leveraged businesses.
      - Use news sentiment and insider trades for secondary inputs.
      - If fundamentals strongly align with GARP, be more aggressive.

    The result is a bullish/bearish/neutral signal, along with a
    confidence (0–100) and a textual reasoning explanation.
    """

    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    lynch_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Gathering financial line items")
        # Relevant line items for Peter Lynch's approach
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "earnings_per_share",
                "net_income",
                "operating_income",
                "gross_margin",
                "operating_margin",
                "free_cash_flow",
                "capital_expenditure",
                "cash_and_equivalents",
                "total_debt",
                "shareholders_equity",
                "outstanding_shares",
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date, limit=50, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, limit=50, api_key=api_key)

        # Perform sub-analyses:
        progress.update_status(agent_id, ticker, "Analyzing growth")
        growth_analysis = analyze_lynch_growth(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing fundamentals")
        fundamentals_analysis = analyze_lynch_fundamentals(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing valuation (focus on PEG)")
        valuation_analysis = analyze_lynch_valuation(financial_line_items, market_cap)

        progress.update_status(agent_id, ticker, "Analyzing sentiment")
        sentiment_analysis = analyze_sentiment(company_news)

        progress.update_status(agent_id, ticker, "Analyzing insider activity")
        insider_activity = analyze_insider_activity(insider_trades)

        analysis_data[ticker] = {
            "growth_analysis": growth_analysis,
            "valuation_analysis": valuation_analysis,
            "fundamentals_analysis": fundamentals_analysis,
            "sentiment_analysis": sentiment_analysis,
            "insider_activity": insider_activity,
            "context": {
                "market_cap": market_cap,
                "financial_periods": len(financial_line_items),
                "net_income": financial_line_items[0].net_income if financial_line_items else None,
            },
        }

        progress.update_status(agent_id, ticker, "Prepared analysis context")

    summary_table = _build_summary_table(analysis_data)
    portfolio_snapshot = _build_portfolio_snapshot(state["data"].get("portfolio"))

    analysis_dataset = {
        "per_ticker": analysis_data,
        "summary_table": summary_table,
        "portfolio_snapshot": portfolio_snapshot,
        "as_of": end_date,
    }

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Generating Peter Lynch analysis")
        lynch_output = generate_lynch_output(
            ticker=ticker,
            analysis_dataset=analysis_dataset,
            state=state,
            agent_id=agent_id,
        )

        lynch_analysis[ticker] = {
            "signal": lynch_output.signal,
            "confidence": lynch_output.confidence,
            "reasoning": lynch_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=lynch_output.reasoning)

    # Wrap up results
    message = HumanMessage(content=json.dumps(lynch_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(lynch_analysis, "Peter Lynch Agent")

    # Save signals to state
    state["data"]["analyst_signals"][agent_id] = lynch_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


async def peter_lynch_agent_async(state: AgentState, agent_id: str = "peter_lynch_agent"):
    """Async Peter Lynch persona leveraging non-blocking data fetches."""

    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    lynch_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = await search_line_items_async(
            ticker,
            [
                "revenue",
                "earnings_per_share",
                "net_income",
                "operating_income",
                "gross_margin",
                "operating_margin",
                "free_cash_flow",
                "capital_expenditure",
                "cash_and_equivalents",
                "total_debt",
                "shareholders_equity",
                "outstanding_shares",
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = await get_market_cap_async(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching insider trades")
        insider_trades = await get_insider_trades_async(ticker, end_date, limit=50, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = await get_company_news_async(ticker, end_date, limit=50, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing growth")
        growth_analysis = analyze_lynch_growth(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing fundamentals")
        fundamentals_analysis = analyze_lynch_fundamentals(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing valuation (focus on PEG)")
        valuation_analysis = analyze_lynch_valuation(financial_line_items, market_cap)

        progress.update_status(agent_id, ticker, "Analyzing sentiment")
        sentiment_analysis = analyze_sentiment(company_news)

        progress.update_status(agent_id, ticker, "Analyzing insider activity")
        insider_activity = analyze_insider_activity(insider_trades)

        analysis_data[ticker] = {
            "growth_analysis": growth_analysis,
            "valuation_analysis": valuation_analysis,
            "fundamentals_analysis": fundamentals_analysis,
            "sentiment_analysis": sentiment_analysis,
            "insider_activity": insider_activity,
            "context": {
                "market_cap": market_cap,
                "financial_periods": len(financial_line_items),
                "net_income": financial_line_items[0].net_income if financial_line_items else None,
            },
        }

        progress.update_status(agent_id, ticker, "Prepared analysis context")

    summary_table = _build_summary_table(analysis_data)
    portfolio_snapshot = _build_portfolio_snapshot(state["data"].get("portfolio"))

    analysis_dataset = {
        "per_ticker": analysis_data,
        "summary_table": summary_table,
        "portfolio_snapshot": portfolio_snapshot,
        "as_of": end_date,
    }

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Generating Peter Lynch analysis")
        lynch_output = await generate_lynch_output_async(
            ticker=ticker,
            analysis_dataset=analysis_dataset,
            state=state,
            agent_id=agent_id,
        )

        lynch_analysis[ticker] = {
            "signal": lynch_output.signal,
            "confidence": lynch_output.confidence,
            "reasoning": lynch_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=lynch_output.reasoning)

    message = HumanMessage(content=json.dumps(lynch_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(lynch_analysis, "Peter Lynch Agent")

    await update_analyst_signals_async(state, agent_id, lynch_analysis)

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def analyze_lynch_growth(financial_line_items: list) -> dict:
    """
    Evaluate growth based on revenue and EPS trends:
      - Consistent revenue growth
      - Consistent EPS growth
    Peter Lynch liked companies with steady, understandable growth,
    often searching for potential 'ten-baggers' with a long runway.
    """
    if not financial_line_items or len(financial_line_items) < 2:
        return {
            "metrics": {},
            "flags": {},
            "notes": ["Insufficient financial data for growth analysis."],
        }

    notes: list[str] = []
    metrics: dict[str, float | None] = {
        "revenue_growth_cagr": None,
        "revenue_growth_acceleration": None,
        "eps_growth_cagr": None,
        "eps_growth_acceleration": None,
    }
    flags: dict[str, bool | None] = {
        "revenue_consistent": None,
        "eps_consistent": None,
    }

    # 1) Revenue Growth
    revenues = [fi.revenue for fi in financial_line_items if fi.revenue is not None]
    if len(revenues) >= 2 and revenues[-1]:
        latest_rev = revenues[0]
        older_rev = revenues[-1]
        rev_growth = (latest_rev - older_rev) / abs(older_rev)
        metrics["revenue_growth_cagr"] = float(rev_growth)
        notes.append(f"Revenue change over period: {rev_growth:.1%}.")

        growth_rates: list[float] = []
        for i in range(len(revenues) - 1):
            base = revenues[i + 1]
            if base:
                growth_rates.append((revenues[i] - base) / abs(base))

        if len(growth_rates) >= 2:
            acceleration = growth_rates[0] - growth_rates[-1]
            metrics["revenue_growth_acceleration"] = float(acceleration)
            flags["revenue_consistent"] = acceleration >= 0 and min(growth_rates) > 0
            if acceleration > 0:
                notes.append(
                    f"Revenue growth accelerating: latest {growth_rates[0]*100:.1f}% vs oldest {growth_rates[-1]*100:.1f}%."
                )
        else:
            flags["revenue_consistent"] = all(rate > 0 for rate in growth_rates)
    else:
        notes.append("Insufficient revenue data to compute growth CAGR.")

    # 2) EPS Growth
    eps_values = [fi.earnings_per_share for fi in financial_line_items if fi.earnings_per_share is not None]
    if len(eps_values) >= 2 and abs(eps_values[-1]) > 1e-9:
        latest_eps = eps_values[0]
        older_eps = eps_values[-1]
        eps_growth = (latest_eps - older_eps) / abs(older_eps)
        metrics["eps_growth_cagr"] = float(eps_growth)
        notes.append(f"EPS change over period: {eps_growth:.1%}.")

        eps_rates: list[float] = []
        for i in range(len(eps_values) - 1):
            base = eps_values[i + 1]
            if abs(base) > 1e-9:
                eps_rates.append((eps_values[i] - base) / abs(base))

        if len(eps_rates) >= 2:
            eps_acceleration = eps_rates[0] - eps_rates[-1]
            metrics["eps_growth_acceleration"] = float(eps_acceleration)
            flags["eps_consistent"] = eps_acceleration >= 0 and min(eps_rates) > 0
            if eps_acceleration > 0:
                notes.append(
                    f"EPS growth accelerating: latest {eps_rates[0]*100:.1f}% vs oldest {eps_rates[-1]*100:.1f}%."
                )
        else:
            flags["eps_consistent"] = all(rate > 0 for rate in eps_rates)
    else:
        notes.append("Insufficient EPS data to compute growth CAGR.")

    return {"metrics": metrics, "flags": flags, "notes": notes}


def analyze_lynch_fundamentals(financial_line_items: list) -> dict:
    """
    Evaluate basic fundamentals:
      - Debt/Equity
      - Operating margin (or gross margin)
      - Positive Free Cash Flow
    Lynch avoided heavily indebted or complicated businesses.
    """
    if not financial_line_items:
        return {
            "metrics": {},
            "flags": {},
            "notes": ["Insufficient fundamentals data."],
        }

    notes: list[str] = []
    metrics: dict[str, float | None] = {
        "debt_to_equity": None,
        "operating_margin_latest": None,
        "operating_margin_trend": None,
        "free_cash_flow_latest": None,
        "free_cash_flow_positive_streak": None,
        "current_ratio": None,
        "cash_to_debt": None,
        "dividend_payout_ratio": None,
    }
    flags: dict[str, bool | None] = {
        "leverage_low": None,
        "operating_margin_healthy": None,
        "free_cash_flow_positive": None,
    }

    debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]
    equity_values = [fi.shareholders_equity for fi in financial_line_items if fi.shareholders_equity is not None]
    if debt_values and equity_values:
        recent_debt = float(debt_values[0])
        recent_equity = float(equity_values[0]) if abs(equity_values[0]) > 1e-9 else 1e-9
        de_ratio = recent_debt / recent_equity
        metrics["debt_to_equity"] = de_ratio
        flags["leverage_low"] = de_ratio < 0.5
        notes.append(f"Debt-to-equity ratio: {de_ratio:.2f}.")
    else:
        notes.append("Debt/equity data unavailable.")

    om_values = [fi.operating_margin for fi in financial_line_items if fi.operating_margin is not None]
    if om_values:
        metrics["operating_margin_latest"] = float(om_values[0])
        if len(om_values) >= 2:
            metrics["operating_margin_trend"] = float(om_values[0] - om_values[-1])
        flags["operating_margin_healthy"] = om_values[0] is not None and om_values[0] > 0.12
        notes.append(f"Operating margin latest: {om_values[0]*100:.1f}%.")
    else:
        notes.append("Operating margin data unavailable.")

    fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]
    if fcf_values:
        metrics["free_cash_flow_latest"] = float(fcf_values[0])
        positive_periods = sum(1 for value in fcf_values if value and value > 0)
        metrics["free_cash_flow_positive_streak"] = float(positive_periods)
        flags["free_cash_flow_positive"] = fcf_values[0] > 0
        notes.append(
            f"Free cash flow latest: {fcf_values[0]:,.0f}; positive periods {positive_periods}/{len(fcf_values)}."
        )
    else:
        notes.append("Free cash flow data unavailable.")

    if (
        hasattr(financial_line_items[0], "current_assets")
        and financial_line_items[0].current_assets
        and hasattr(financial_line_items[0], "current_liabilities")
        and financial_line_items[0].current_liabilities
    ):
        metrics["current_ratio"] = float(
            financial_line_items[0].current_assets / max(financial_line_items[0].current_liabilities, 1e-9)
        )

    if (
        hasattr(financial_line_items[0], "cash_and_equivalents")
        and financial_line_items[0].cash_and_equivalents is not None
        and debt_values
        and debt_values[0]
    ):
        metrics["cash_to_debt"] = float(financial_line_items[0].cash_and_equivalents / max(debt_values[0], 1e-9))

    if (
        hasattr(financial_line_items[0], "dividends_and_other_cash_distributions")
        and financial_line_items[0].dividends_and_other_cash_distributions is not None
        and fcf_values
        and fcf_values[0]
    ):
        metrics["dividend_payout_ratio"] = float(
            financial_line_items[0].dividends_and_other_cash_distributions / max(fcf_values[0], 1e-9)
        )

    return {"metrics": metrics, "flags": flags, "notes": notes}


def analyze_lynch_valuation(financial_line_items: list, market_cap: float | None) -> dict:
    """
    Peter Lynch's approach to 'Growth at a Reasonable Price' (GARP):
      - Emphasize the PEG ratio: (P/E) / Growth Rate
      - Also consider a basic P/E if PEG is unavailable
    A PEG < 1 is very attractive; 1-2 is fair; >2 is expensive.
    """
    if not financial_line_items or market_cap is None or market_cap <= 0:
        return {
            "metrics": {},
            "flags": {},
            "notes": ["Insufficient data for valuation."],
        }

    notes: list[str] = []
    metrics: dict[str, float | None] = {
        "pe_ratio": None,
        "eps_growth_rate": None,
        "peg_ratio": None,
        "price_to_book": None,
        "price_to_sales": None,
    }
    flags: dict[str, bool | None] = {
        "peg_reasonable": None,
        "pe_reasonable": None,
    }

    net_incomes = [fi.net_income for fi in financial_line_items if fi.net_income is not None]
    if net_incomes and net_incomes[0] and net_incomes[0] > 0:
        pe_ratio = market_cap / net_incomes[0]
        metrics["pe_ratio"] = float(pe_ratio)
        flags["pe_reasonable"] = pe_ratio < 20
        notes.append(f"Approximate P/E: {pe_ratio:.2f}.")
    else:
        notes.append("Unable to compute P/E (net income missing or negative).")

    eps_values = [fi.earnings_per_share for fi in financial_line_items if fi.earnings_per_share is not None]
    eps_growth_rate = None
    if len(eps_values) >= 2 and eps_values[-1] not in (0, None):
        latest_eps = eps_values[0]
        older_eps = eps_values[-1]
        years = len(eps_values) - 1
        if latest_eps and older_eps:
            eps_growth_rate = (latest_eps / older_eps) ** (1 / years) - 1 if older_eps > 0 else None
    if eps_growth_rate is not None:
        metrics["eps_growth_rate"] = float(eps_growth_rate)
        notes.append(f"CAGR EPS growth ~ {eps_growth_rate:.1%}.")
    else:
        notes.append("EPS growth rate unavailable.")

    if metrics["pe_ratio"] and metrics["eps_growth_rate"] and metrics["eps_growth_rate"] > 0:
        peg_ratio = metrics["pe_ratio"] / (metrics["eps_growth_rate"] * 100)
        metrics["peg_ratio"] = float(peg_ratio)
        flags["peg_reasonable"] = peg_ratio < 1.5
        notes.append(f"PEG ratio approximately {peg_ratio:.2f}.")

    latest = financial_line_items[0]
    if getattr(latest, "book_value_per_share", None) and getattr(latest, "outstanding_shares", None):
        book_value = latest.book_value_per_share * latest.outstanding_shares
        if book_value:
            metrics["price_to_book"] = float(market_cap / book_value)

    if getattr(latest, "revenue", None):
        metrics["price_to_sales"] = float(market_cap / latest.revenue) if latest.revenue else None

    return {"metrics": metrics, "flags": flags, "notes": notes}


def analyze_sentiment(news_items: list) -> dict:
    """
    Basic news sentiment check. Negative headlines weigh on the final score.
    """
    if not news_items:
        return {
            "metrics": {"headline_count": 0, "negative_ratio": None},
            "notes": ["No recent news; treating sentiment as neutral."],
        }

    negative_keywords = {"lawsuit", "fraud", "negative", "downturn", "decline", "investigation", "recall"}
    negative_count = 0
    notes: list[str] = []

    for news in news_items:
        headline = (news.title or "").lower()
        if any(word in headline for word in negative_keywords):
            negative_count += 1

    ratio = negative_count / len(news_items)
    if ratio > 0.3:
        notes.append(f"High proportion of negative headlines: {negative_count}/{len(news_items)}.")
    elif ratio > 0:
        notes.append(f"Mixed headlines with some negative coverage ({negative_count}/{len(news_items)}).")
    else:
        notes.append("Headlines largely positive or neutral.")

    return {
        "metrics": {"headline_count": len(news_items), "negative_ratio": ratio},
        "notes": notes,
    }


def analyze_insider_activity(insider_trades: list) -> dict:
    """
    Simple insider-trade analysis:
      - If there's heavy insider buying, it's a positive sign.
      - If there's mostly selling, it's a negative sign.
      - Otherwise, neutral.
    """
    if not insider_trades:
        return {
            "metrics": {"buy_count": 0, "sell_count": 0, "net_buy_volume": 0.0},
            "notes": ["No insider trade data provided."],
        }

    buy_count = 0
    sell_count = 0
    net_volume = 0.0

    for trade in insider_trades:
        shares = getattr(trade, "transaction_shares", None)
        if shares is None:
            continue
        net_volume += shares
        if shares > 0:
            buy_count += 1
        elif shares < 0:
            sell_count += 1

    total_trades = buy_count + sell_count
    notes: list[str] = []
    if total_trades == 0:
        notes.append("No significant buy/sell transactions found.")
    else:
        buy_ratio = buy_count / total_trades
        notes.append(f"Insider trades: {buy_count} buys vs {sell_count} sells (buy ratio {buy_ratio:.2f}).")
        if net_volume != 0:
            notes.append(f"Net insider share volume: {net_volume:,.0f}.")

    return {
        "metrics": {
            "buy_count": buy_count,
            "sell_count": sell_count,
            "net_buy_volume": net_volume,
        },
        "notes": notes,
    }


def generate_lynch_output(
    ticker: str,
    analysis_dataset: dict[str, Any],
    state: AgentState,
    agent_id: str,
) -> PeterLynchSignal:
    """
    Generates a final JSON signal in Peter Lynch's voice & style.
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Peter Lynch AI agent. You make investment decisions based on Peter Lynch's well-known principles:
                
                1. Invest in What You Know: Emphasize understandable businesses, possibly discovered in everyday life.
                2. Growth at a Reasonable Price (GARP): Rely on the PEG ratio as a prime metric.
                3. Look for 'Ten-Baggers': Companies capable of growing earnings and share price substantially.
                4. Steady Growth: Prefer consistent revenue/earnings expansion, less concern about short-term noise.
                5. Avoid High Debt: Watch for dangerous leverage.
                6. Management & Story: A good 'story' behind the stock, but not overhyped or too complex.
                
                When you provide your reasoning, do it in Peter Lynch's voice:
                - Cite the PEG ratio
                - Mention 'ten-bagger' potential if applicable
                - Refer to personal or anecdotal observations (e.g., "If my kids love the product...")
                - Use practical, folksy language
                - Provide key positives and negatives
                - Conclude with a clear stance (bullish, bearish, or neutral)
                
                Return your final output strictly in JSON with the fields:
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": 0 to 100,
                  "reasoning": "string"
                }}
                """,
            ),
            (
                "human",
                """Based on the following dataset, produce your Peter Lynch–style investment signal.

Ticker: {ticker}
Summary Table (batch comparison):
{summary_table}

Portfolio Snapshot:
{portfolio_snapshot}

Per-Ticker Analysis:
{per_ticker}

Return only valid JSON with "signal", "confidence", and "reasoning".
""",
            ),
        ]
    )

    per_ticker = analysis_dataset["per_ticker"][ticker]
    prompt = template.invoke(
        {
            "ticker": ticker,
            "summary_table": analysis_dataset["summary_table"],
            "portfolio_snapshot": json.dumps(analysis_dataset.get("portfolio_snapshot", {}), indent=2),
            "per_ticker": json.dumps(per_ticker, indent=2),
        }
    )

    def create_default_signal():
        return build_peter_lynch_fallback_signal(ticker, per_ticker)

    return call_llm(
        prompt=prompt,
        pydantic_model=PeterLynchSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )


async def generate_lynch_output_async(
    ticker: str,
    analysis_dataset: dict[str, Any],
    state: AgentState,
    agent_id: str,
) -> PeterLynchSignal:
    """Async variant of generate_lynch_output."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Peter Lynch. Issue a growth-at-reasonable-price judgement grounded in fundamentals and everyday common sense.""",
            ),
            (
                "human",
                """Ticker: {ticker}

Summary Table:
{summary_table}

Portfolio Snapshot:
{portfolio_snapshot}

Per-Ticker Analysis:
{per_ticker}

Return JSON:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": float,
  "reasoning": "string"
}}""",
            ),
        ]
    )

    per_ticker = analysis_dataset["per_ticker"][ticker]
    prompt = template.invoke(
        {
            "ticker": ticker,
            "summary_table": analysis_dataset["summary_table"],
            "portfolio_snapshot": json.dumps(analysis_dataset.get("portfolio_snapshot", {}), indent=2),
            "per_ticker": json.dumps(per_ticker, indent=2),
        }
    )

    def create_default_signal():
        return build_peter_lynch_fallback_signal(ticker, per_ticker)

    return await async_call_llm(
        prompt=prompt,
        pydantic_model=PeterLynchSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )
