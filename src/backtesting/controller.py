from __future__ import annotations

import copy
import inspect
from typing import Any, Callable, Dict, Sequence

from .data_types import (
    Action,
    ActionLiteral,
    AgentDecisions,
    AgentOutput,
    PortfolioSnapshot,
)
from .portfolio import Portfolio


class AgentController:
    """Responsible for invoking the trading agent and normalizing outputs."""

    def _snapshot_portfolio(self, portfolio: Portfolio | PortfolioSnapshot) -> PortfolioSnapshot:
        if isinstance(portfolio, Portfolio):
            return portfolio.get_snapshot()
        return portfolio

    def _normalize_output(self, output: Dict[str, Any] | None, tickers: Sequence[str]) -> AgentOutput:
        decisions_in: Dict[str, Any] = dict(output.get("decisions", {})) if isinstance(output, dict) else {}
        analyst_signals_in: Dict[str, Any] = dict(output.get("analyst_signals", {})) if isinstance(output, dict) else {}

        normalized_decisions: AgentDecisions = {}
        for ticker in tickers:
            d = decisions_in.get(ticker, {})
            action = d.get("action", "hold")
            qty = d.get("quantity", 0)
            try:
                qty_val = float(qty)
            except Exception:
                qty_val = 0.0
            try:
                action = Action(action).value  # validate/coerce
            except Exception:
                action = Action.HOLD.value  # type: ignore[assignment]
            normalized_decisions[ticker] = {"action": action, "quantity": qty_val}  # type: ignore[assignment]

        normalized_output: AgentOutput = {
            "decisions": normalized_decisions,
            "analyst_signals": analyst_signals_in,
        }
        if isinstance(output, dict) and "risk_manager_state" in output:
            normalized_output["risk_manager_state"] = copy.deepcopy(output["risk_manager_state"])
        if isinstance(output, dict) and "timings" in output:
            normalized_output["timings"] = copy.deepcopy(output["timings"])
        return normalized_output

    def run_agent(
        self,
        agent: Callable[..., AgentOutput],
        *,
        tickers: Sequence[str],
        start_date: str,
        end_date: str,
        portfolio: Portfolio | PortfolioSnapshot,
        model_name: str,
        model_provider: str,
        selected_analysts: Sequence[str] | None,
        metadata_overrides: Dict[str, Any] | None = None,
    ) -> AgentOutput:
        portfolio_payload = self._snapshot_portfolio(portfolio)

        output = agent(
            tickers=list(tickers),
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio_payload,
            model_name=model_name,
            model_provider=model_provider,
            selected_analysts=list(selected_analysts) if selected_analysts is not None else None,
            metadata_overrides=metadata_overrides,
        )

        return self._normalize_output(output, tickers)

    async def run_agent_async(
        self,
        agent: Callable[..., AgentOutput],
        *,
        tickers: Sequence[str],
        start_date: str,
        end_date: str,
        portfolio: Portfolio | PortfolioSnapshot,
        model_name: str,
        model_provider: str,
        selected_analysts: Sequence[str] | None,
        metadata_overrides: Dict[str, Any] | None = None,
    ) -> AgentOutput:
        portfolio_payload = self._snapshot_portfolio(portfolio)

        maybe_coroutine = agent(
            tickers=list(tickers),
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio_payload,
            model_name=model_name,
            model_provider=model_provider,
            selected_analysts=list(selected_analysts) if selected_analysts is not None else None,
            metadata_overrides=metadata_overrides,
        )

        if inspect.isawaitable(maybe_coroutine):
            output = await maybe_coroutine
        else:
            output = maybe_coroutine

        return self._normalize_output(output, tickers)
