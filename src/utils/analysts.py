"""Constants and utilities related to analysts configuration."""

import importlib
from typing import Callable

from src.agents import portfolio_manager
from src.agents.aswath_damodaran import aswath_damodaran_agent
from src.agents.ben_graham import ben_graham_agent
from src.agents.bill_ackman import bill_ackman_agent
from src.agents.breakout_cover_sentinel import breakout_cover_sentinel_agent
from src.agents.cathie_wood import cathie_wood_agent
from src.agents.charlie_munger import charlie_munger_agent
from src.agents.crash_short_allocator import crash_short_allocator_agent
from src.agents.downside_flow_sentinel import downside_flow_sentinel_agent
from src.agents.donald_trump import donald_trump_agent
from src.agents.event_catalyst import event_catalyst_agent
from src.agents.fundamentals import fundamentals_analyst_agent
from src.agents.growth_momentum import growth_momentum_agent
from src.agents.macro_volatility_sentinel import macro_volatility_sentinel_agent
from src.agents.michael_burry import michael_burry_agent
from src.agents.mohnish_pabrai import mohnish_pabrai_agent
from src.agents.momentum_guardian import momentum_guardian_agent
from src.agents.peter_lynch import peter_lynch_agent
from src.agents.phil_fisher import phil_fisher_agent
from src.agents.rakesh_jhunjhunwala import rakesh_jhunjhunwala_agent
from src.agents.range_recovery_sentinel import range_recovery_sentinel_agent
from src.agents.regime_meta import regime_meta_agent
from src.agents.sentiment import sentiment_analyst_agent
from src.agents.short_cover_classifier import short_cover_classifier_agent
from src.agents.short_squeeze_guardian import short_squeeze_guardian_agent
from src.agents.stanley_druckenmiller import stanley_druckenmiller_agent
from src.agents.stat_mean_reversion import stat_mean_reversion_agent
from src.agents.stop_loss_guardian import stop_loss_guardian_agent
from src.agents.technicals import technical_analyst_agent
from src.agents.trend_regime import trend_regime_agent
from src.agents.valuation import valuation_analyst_agent
from src.agents.warren_buffett import warren_buffett_agent
from src.utils.runtime import async_personas_enabled

# Define analyst configuration - single source of truth
ANALYST_CONFIG = {
    "aswath_damodaran": {
        "display_name": "Aswath Damodaran",
        "description": "The Dean of Valuation",
        "investing_style": "Focuses on intrinsic value and financial metrics to assess investment opportunities through rigorous valuation analysis.",
        "agent_func": aswath_damodaran_agent,
        "type": "analyst",
        "order": 0,
    },
    "ben_graham": {
        "display_name": "Ben Graham",
        "description": "The Father of Value Investing",
        "investing_style": "Emphasizes a margin of safety and invests in undervalued companies with strong fundamentals through systematic value analysis.",
        "agent_func": ben_graham_agent,
        "type": "analyst",
        "order": 1,
    },
    "bill_ackman": {
        "display_name": "Bill Ackman",
        "description": "The Activist Investor",
        "investing_style": "Seeks to influence management and unlock value through strategic activism and contrarian investment positions.",
        "agent_func": bill_ackman_agent,
        "type": "analyst",
        "order": 2,
    },
    "cathie_wood": {
        "display_name": "Cathie Wood",
        "description": "The Queen of Growth Investing",
        "investing_style": "Focuses on disruptive innovation and growth, investing in companies that are leading technological advancements and market disruption.",
        "agent_func": cathie_wood_agent,
        "type": "analyst",
        "order": 3,
    },
    "charlie_munger": {
        "display_name": "Charlie Munger",
        "description": "The Rational Thinker",
        "investing_style": "Advocates for value investing with a focus on quality businesses and long-term growth through rational decision-making.",
        "agent_func": charlie_munger_agent,
        "type": "analyst",
        "order": 4,
    },
    "michael_burry": {
        "display_name": "Michael Burry",
        "description": "The Big Short Contrarian",
        "investing_style": "Makes contrarian bets, often shorting overvalued markets and investing in undervalued assets through deep fundamental analysis.",
        "agent_func": michael_burry_agent,
        "type": "analyst",
        "order": 5,
    },
    "mohnish_pabrai": {
        "display_name": "Mohnish Pabrai",
        "description": "The Dhandho Investor",
        "investing_style": "Focuses on value investing and long-term growth through fundamental analysis and a margin of safety.",
        "agent_func": mohnish_pabrai_agent,
        "type": "analyst",
        "order": 6,
    },
    "peter_lynch": {
        "display_name": "Peter Lynch",
        "description": "The 10-Bagger Investor",
        "investing_style": "Invests in companies with understandable business models and strong growth potential using the 'buy what you know' strategy.",
        "agent_func": peter_lynch_agent,
        "type": "analyst",
        "order": 6,
    },
    "phil_fisher": {
        "display_name": "Phil Fisher",
        "description": "The Scuttlebutt Investor",
        "investing_style": "Emphasizes investing in companies with strong management and innovative products, focusing on long-term growth through scuttlebutt research.",
        "agent_func": phil_fisher_agent,
        "type": "analyst",
        "order": 7,
    },
    "rakesh_jhunjhunwala": {
        "display_name": "Rakesh Jhunjhunwala",
        "description": "The Big Bull Of India",
        "investing_style": "Leverages macroeconomic insights to invest in high-growth sectors, particularly within emerging markets and domestic opportunities.",
        "agent_func": rakesh_jhunjhunwala_agent,
        "type": "analyst",
        "order": 8,
    },
    "stanley_druckenmiller": {
        "display_name": "Stanley Druckenmiller",
        "description": "The Macro Investor",
        "investing_style": "Focuses on macroeconomic trends, making large bets on currencies, commodities, and interest rates through top-down analysis.",
        "agent_func": stanley_druckenmiller_agent,
        "type": "analyst",
        "order": 9,
    },
    "warren_buffett": {
        "display_name": "Warren Buffett",
        "description": "The Oracle of Omaha",
        "investing_style": "Seeks companies with strong fundamentals and competitive advantages through value investing and long-term ownership.",
        "agent_func": warren_buffett_agent,
        "type": "analyst",
        "order": 10,
    },
    "technical_analyst": {
        "display_name": "Technical Analyst",
        "description": "Chart Pattern Specialist",
        "investing_style": "Focuses on chart patterns and market trends to make investment decisions, often using technical indicators and price action analysis.",
        "agent_func": technical_analyst_agent,
        "type": "analyst",
        "order": 11,
    },
    "growth_momentum": {
        "display_name": "Growth Momentum Analyst",
        "description": "Medium-term trend surfer",
        "investing_style": "Quantifies rolling momentum and volatility compression to lean long in durable uptrends.",
        "agent_func": growth_momentum_agent,
        "type": "analyst",
        "order": 12,
    },
    "stat_mean_reversion": {
        "display_name": "Statistical Mean Reversion",
        "description": "Z-score based rebound detector",
        "investing_style": "Looks for extreme deviations from rolling means and quantifies reversion odds via normal tail probabilities.",
        "agent_func": stat_mean_reversion_agent,
        "type": "analyst",
        "order": 13,
    },
    "fundamentals_analyst": {
        "display_name": "Fundamentals Analyst",
        "description": "Financial Statement Specialist",
        "investing_style": "Delves into financial statements and economic indicators to assess the intrinsic value of companies through fundamental analysis.",
        "agent_func": fundamentals_analyst_agent,
        "type": "analyst",
        "order": 14,
    },
    "sentiment_analyst": {
        "display_name": "Sentiment Analyst",
        "description": "Market Sentiment Specialist",
        "investing_style": "Gauges market sentiment and investor behavior to predict market movements and identify opportunities through behavioral analysis.",
        "agent_func": sentiment_analyst_agent,
        "type": "analyst",
        "order": 15,
    },
    "valuation_analyst": {
        "display_name": "Valuation Analyst",
        "description": "Company Valuation Specialist",
        "investing_style": "Specializes in determining the fair value of companies, using various valuation models and financial metrics for investment decisions.",
        "agent_func": valuation_analyst_agent,
        "type": "analyst",
        "order": 16,
    },
    "momentum_guardian": {
        "display_name": "Momentum Guardian",
        "description": "Regime-aware trend gatekeeper",
        "investing_style": "Screens technical momentum to prevent fighting strong bullish trends when sizing short exposure.",
        "agent_func": momentum_guardian_agent,
        "type": "analyst",
        "order": 17,
    },
    "event_catalyst": {
        "display_name": "Event Catalyst Analyst",
        "description": "Event-driven catalyst specialist",
        "investing_style": "Surfaces imminent catalysts from news flow to tilt short-term positioning.",
        "agent_func": event_catalyst_agent,
        "type": "analyst",
        "order": 18,
    },
    "trend_regime": {
        "display_name": "Trend Regime Analyst",
        "description": "Deterministic price regime detector",
        "investing_style": "Quantifies medium-term trend strength and breakouts to bias the book toward prevailing momentum.",
        "agent_func": trend_regime_agent,
        "type": "analyst",
        "order": 19,
    },
    "short_squeeze_guardian": {
        "display_name": "Short Squeeze Guardian",
        "description": "Short-risk sentry",
        "investing_style": "Detects rapid upside squeezes using deterministic velocity, breadth, and volume metrics to cap short exposure.",
        "agent_func": short_squeeze_guardian_agent,
        "type": "analyst",
        "order": 20,
    },
    "short_cover_classifier": {
        "display_name": "Short-Cover Classifier",
        "description": "Data-driven squeeze exit sentinel",
        "investing_style": "Fits a logistic classifier on price/volume features to forecast upside squeezes and unwind stale shorts.",
        "agent_func": short_cover_classifier_agent,
        "type": "analyst",
        "order": 21,
    },
    "breakout_cover_sentinel": {
        "display_name": "Breakout Cover Sentinel",
        "description": "Breakout cover sentry",
        "investing_style": "Identifies durable upside breakouts and forces persistent shorts to cover while biasing the book long.",
        "agent_func": breakout_cover_sentinel_agent,
        "type": "analyst",
        "order": 22,
    },
    "range_recovery_sentinel": {
        "display_name": "Range Recovery Sentinel",
        "description": "Range drift sentry",
        "investing_style": "Detects upside drift inside compressed ranges to unwind stale shorts and cap fresh short exposure.",
        "agent_func": range_recovery_sentinel_agent,
        "type": "analyst",
        "order": 23,
    },
    "regime_meta": {
        "display_name": "Regime Meta-Model",
        "description": "Volatility and drawdown classifier guiding risk overrides",
        "investing_style": "Blends deterministic regime features with ML analyst probabilities to steer long-vs-short exposure nudges.",
        "agent_func": regime_meta_agent,
        "type": "analyst",
        "order": 25,
    },
    "macro_volatility_sentinel": {
        "display_name": "Macro Volatility Sentinel",
        "description": "Volatility surge sentry",
        "investing_style": "Monitors volatility acceleration and drawdowns to suppress long exposure overrides ahead of crash regimes.",
        "agent_func": macro_volatility_sentinel_agent,
        "type": "analyst",
        "order": 26,
    },
    "downside_flow_sentinel": {
        "display_name": "Downside Flow Sentinel",
        "description": "Crash-flow sentry",
        "investing_style": "Tracks persistent downside momentum and drawdowns to enforce defensive short tilts when crash flows emerge.",
        "agent_func": downside_flow_sentinel_agent,
        "type": "analyst",
        "order": 24,
    },
    "crash_short_allocator": {
        "display_name": "Crash Short Allocator",
        "description": "Crash short allocator",
        "investing_style": "Aggregates crash probability, volatility surge, and downside flow diagnostics into explicit short targets for risk manager overrides.",
        "agent_func": crash_short_allocator_agent,
        "type": "analyst",
        "order": 27,
    },
    "stop_loss_guardian": {
        "display_name": "Stop-Loss Guardian",
        "description": "Loss containment sentry",
        "investing_style": "Monitors live positions and enforces deterministic stop-loss trims once adverse moves breach volatility bands.",
        "agent_func": stop_loss_guardian_agent,
        "type": "analyst",
        "order": 28,
    },
    "donald_trump": {
        "display_name": "Donald Trump Analyst",
        "description": "Populist policy shock persona",
        "investing_style": "Trades tariffs, tax shifts, and media momentum with campaign-style conviction to exploit policy surprises.",
        "agent_func": donald_trump_agent,
        "type": "analyst",
        "order": 29,
    },
}


def _lookup_async_variant(agent_func):
    module = importlib.import_module(agent_func.__module__)
    async_name = f"{agent_func.__name__}_async"
    return getattr(module, async_name, None)


# Derive ANALYST_ORDER from ANALYST_CONFIG for backwards compatibility
ANALYST_ORDER = [(config["display_name"], key) for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])]


def get_analyst_nodes(async_enabled: bool | None = None):
    """Get the mapping of analyst keys to their (node_name, agent_func) tuples."""
    use_async = async_personas_enabled() if async_enabled is None else async_enabled
    analyst_nodes: dict[str, tuple[str, Callable]] = {}
    for key, config in ANALYST_CONFIG.items():
        func = config["agent_func"]
        if use_async:
            async_candidate = config.get("agent_func_async") or _lookup_async_variant(func)
            if async_candidate is not None:
                func = async_candidate
        analyst_nodes[key] = (f"{key}_agent", func)
    return analyst_nodes


def get_agents_list():
    """Get the list of agents for API responses."""
    return [{"key": key, "display_name": config["display_name"], "description": config["description"], "investing_style": config["investing_style"], "order": config["order"]} for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])]
