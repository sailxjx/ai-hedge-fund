from langchain_core.messages import HumanMessage
from src.agents.persona_utils import persona_from_observations
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import pandas as pd
import numpy as np
import json
from src.utils.api_key import get_api_key_from_state
from src.tools.api import get_insider_trades, get_company_news

PERSONA_NAME = "Echo"
PERSONA_ROLE = "a sentiment persona blending insider flows and news tone"
PERSONA_BACKSTORY = (
    "Echo tracked behavioural flows for a macro fund and now interprets insider trades and news sentiment before advising positions."
)
PERSONA_INSTRUCTIONS = (
    "Compare insider flow vs news tone; explain when they diverge.",
    "Highlight catalyst snippets from the news when leaning bullish or bearish.",
    "Speak in first person and cite the key datapoints driving the judgement.",
)
ALLOWED_SIGNALS = ["bullish", "bearish", "neutral"]

##### Sentiment Agent #####
def sentiment_analyst_agent(state: AgentState, agent_id: str = "sentiment_analyst_agent"):
    """Analyzes market sentiment and generates trading signals for multiple tickers."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    # Initialize sentiment analysis for each ticker
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching insider trades")

        # Get the insider trades
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Analyzing trading patterns")

        # Get the signals from the insider trades
        transaction_shares = pd.Series([t.transaction_shares for t in insider_trades]).dropna()
        insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

        progress.update_status(agent_id, ticker, "Fetching company news")

        # Get the company news
        company_news = get_company_news(ticker, end_date, limit=100, api_key=api_key)

        # Get the sentiment from the company news
        sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
        news_signals = np.where(sentiment == "negative", "bearish", 
                              np.where(sentiment == "positive", "bullish", "neutral")).tolist()
        
        progress.update_status(agent_id, ticker, "Combining signals")
        # Combine signals from both sources with weights
        insider_weight = 0.3
        news_weight = 0.7
        
        # Calculate weighted signal counts
        bullish_signals = (
            insider_signals.count("bullish") * insider_weight +
            news_signals.count("bullish") * news_weight
        )
        bearish_signals = (
            insider_signals.count("bearish") * insider_weight +
            news_signals.count("bearish") * news_weight
        )

        breakdown = {
            "insider_trades": {
                "total_trades": len(insider_signals),
                "bullish_trades": insider_signals.count("bullish"),
                "bearish_trades": insider_signals.count("bearish"),
                "weighted_bullish": round(insider_signals.count("bullish") * insider_weight, 4),
                "weighted_bearish": round(insider_signals.count("bearish") * insider_weight, 4),
                "weight": insider_weight,
            },
            "news_sentiment": {
                "total_articles": len(news_signals),
                "bullish_articles": news_signals.count("bullish"),
                "bearish_articles": news_signals.count("bearish"),
                "neutral_articles": news_signals.count("neutral"),
                "weighted_bullish": round(news_signals.count("bullish") * news_weight, 4),
                "weighted_bearish": round(news_signals.count("bearish") * news_weight, 4),
                "weight": news_weight,
            },
            "combined": {
                "total_weighted_bullish": round(bullish_signals, 4),
                "total_weighted_bearish": round(bearish_signals, 4),
                "total_weighted_signals": len(insider_signals) * insider_weight + len(news_signals) * news_weight,
            },
            "recent_news": [
                {"headline": n.title, "sentiment": n.sentiment, "source": n.source}
                for n in company_news[:5]
            ],
        }

        observations = {
            "ticker": ticker,
            "weights": {"insider": insider_weight, "news": news_weight},
            "breakdown": breakdown,
        }

        decision = persona_from_observations(
            state=state,
            agent_id=agent_id,
            persona_name=PERSONA_NAME,
            persona_role=PERSONA_ROLE,
            persona_backstory=PERSONA_BACKSTORY,
            allowed_signals=ALLOWED_SIGNALS,
            observations=observations,
            persona_instructions=PERSONA_INSTRUCTIONS,
            default_signal="neutral",
            default_confidence=50.0,
            default_reasoning="Defaulted to neutral after missing persona response.",
        )

        payload = {
            "signal": decision.signal,
            "confidence": int(max(0, min(round(decision.confidence), 100))),
            "reasoning": decision.reasoning,
            "constraints": decision.constraints or {},
            "breakdown": breakdown,
            "meta": {"observations": observations},
        }

        sentiment_analysis[ticker] = payload

        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(payload, indent=4))


    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name=agent_id,
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = sentiment_analysis

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": data,
    }
