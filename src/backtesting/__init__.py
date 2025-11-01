"""Backtesting package: interfaces and shared types for refactoring.

This module defines the public contracts (Protocols/ABCs) for the
backtesting subsystem. Implementations can live elsewhere and be
introduced gradually without changing existing behavior.
"""

from .controller import AgentController
from .data_types import (
    ActionLiteral,
    AgentDecision,
    AgentDecisions,
    AgentOutput,
    AgentSignals,
    PerformanceMetrics,
    PortfolioSnapshot,
    PortfolioValuePoint,
    PositionState,
    PriceDataFrame,
    TickerRealizedGains,
)
from .engine import BacktestEngine
from .metrics import PerformanceMetricsCalculator
from .output import OutputBuilder
from .portfolio import Portfolio
from .trader import TradeExecutor
from .valuation import calculate_portfolio_value, compute_exposures

__all__ = [
    # Types
    "ActionLiteral",
    "AgentDecision",
    "AgentDecisions",
    "AgentOutput",
    "AgentSignals",
    "PerformanceMetrics",
    "PortfolioSnapshot",
    "PortfolioValuePoint",
    "PositionState",
    "PriceDataFrame",
    "TickerRealizedGains",
    # Interfaces
    "Portfolio",
    "TradeExecutor",
    "PerformanceMetricsCalculator",
    "AgentController",
    "BacktestEngine",
    "calculate_portfolio_value",
    "compute_exposures",
    "OutputBuilder",
]
