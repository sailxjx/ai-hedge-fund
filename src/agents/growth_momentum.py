from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Tuple

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.persona_utils import (
    async_persona_from_observations,
    persona_from_observations,
)
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices, get_prices_async, prices_to_df, prices_to_df_async
from src.utils.api_key import get_api_key_from_state
from src.utils.async_state import update_analyst_signals_async
from src.utils.calibration import apply_platt, get_platt_parameters
from src.utils.progress import progress


def _extend_start(start_date: str | None, buffer_days: int = 200) -> str | None:
    if not start_date:
        return None
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        return start_date
    return (start_dt - timedelta(days=buffer_days)).strftime("%Y-%m-%d")


def _prepare_features(closes: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    closes = closes.astype(float)
    returns = closes.pct_change()

    features = pd.DataFrame(
        {
            "ret_1": returns,
            "ret_5": closes.pct_change(5),
            "ret_21": closes.pct_change(21),
            "vol_5": returns.rolling(5).std(ddof=0),
            "vol_21": returns.rolling(21).std(ddof=0),
            "ema_gap": closes / closes.ewm(span=21, adjust=False).mean() - 1,
        }
    )

    future_returns = returns.shift(-1)

    dataset = features.join(future_returns.rename("future_ret")).dropna()

    labels = (dataset["future_ret"] > 0).astype(float)
    features = dataset.drop(columns=["future_ret"])

    return features, labels


def _logistic_regression(X: np.ndarray, y: np.ndarray, *, lr: float = 0.1, epochs: int = 400, l2: float = 0.01) -> np.ndarray:
    weights = np.zeros(X.shape[1])
    for _ in range(epochs):
        z = X @ weights
        preds = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        gradient = (X.T @ (preds - y)) / len(y) + l2 * weights
        weights -= lr * gradient
    return weights


def _expected_value(prob: float, up_ret: float, down_ret: float) -> float:
    return prob * up_ret + (1.0 - prob) * down_ret


PERSONA_NAME = "Nova"
PERSONA_ROLE = "a momentum surfer who translates logistic edges into positioning guidance"
PERSONA_BACKSTORY = "Nova rode quantitative momentum models for years and now narrates when the book should lean bullish or bearish."
PERSONA_INSTRUCTIONS = (
    "Base the signal entirely on the observed probabilities, expected returns, and drawdown context.",
    "Set constraints only when the observations justify them; otherwise leave them empty.",
    "Explain the judgement in first person, citing the observations that drove the decision.",
)
ALLOWED_SIGNALS = ["bullish", "bearish", "neutral"]


##### Growth Momentum Analyst #####
def growth_momentum_agent(state: AgentState, agent_id: str = "growth_momentum_agent"):
    """Fits a logistic classifier on rolling returns to produce data-driven momentum signals."""

    data = state.get("data", {})
    tickers = data.get("tickers", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    extended_start = _extend_start(start_date)

    signals: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching price history")

        prices = get_prices(
            ticker=ticker,
            start_date=extended_start or start_date,
            end_date=end_date,
            api_key=api_key,
        )

        if not prices:
            signals[ticker] = {
                "signal": "neutral",
                "confidence": 40,
                "reasoning": "Insufficient price data for logistic modelling.",
            }
            continue

        df = prices_to_df(prices)
        if df.empty or len(df) < 90:
            signals[ticker] = {
                "signal": "neutral",
                "confidence": 42,
                "reasoning": "Need at least 90 observations to fit the classifier.",
            }
            continue

        df = df.sort_index()
        features_df, labels = _prepare_features(df["close"])
        if len(features_df) < 40 or labels.nunique() < 2:
            signals[ticker] = {
                "signal": "neutral",
                "confidence": 45,
                "reasoning": "Momentum classifier could not be trained (insufficient variation).",
            }
            continue

        feature_matrix = features_df.to_numpy(dtype=float)
        # Append intercept column
        intercept = np.ones((feature_matrix.shape[0], 1))
        X = np.hstack([intercept, feature_matrix])
        y = labels.to_numpy(dtype=float)

        # Hold out the most recent row for inference
        train_X = X[:-1]
        train_y = y[:-1]
        latest_features = X[-1]

        weights = _logistic_regression(train_X, train_y)
        logits = float(latest_features @ weights)
        raw_prob_up = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

        calibration_params = get_platt_parameters("growth_momentum", ticker)
        if calibration_params:
            prob_up = apply_platt(logits, calibration_params)
        else:
            prob_up = raw_prob_up

        future_returns = features_df.index.to_series().map(df["close"].pct_change().shift(-1)).dropna()

        up_returns = future_returns[future_returns > 0]
        down_returns = future_returns[future_returns <= 0]

        avg_up = float(up_returns.mean()) if not up_returns.empty else 0.01
        avg_down = float(down_returns.mean()) if not down_returns.empty else -0.01
        expected = _expected_value(prob_up, avg_up, avg_down)
        std_future = float(future_returns.std(ddof=0)) if len(future_returns) > 1 else 0.01

        denom = std_future if std_future > 0 else 0.01
        tolerance = 0.1 * denom
        strength = expected / denom

        indicators = {
            "prob_up": prob_up,
            "raw_prob_up": raw_prob_up,
            "expected_return": expected,
            "avg_up": avg_up,
            "avg_down": avg_down,
            "std_future": std_future,
            "base_rate": float(train_y.mean()),
            "logit": logits,
            "calibration_applied": bool(calibration_params),
        }
        if calibration_params:
            indicators["calibration"] = calibration_params

        observations = {
            "ticker": ticker,
            "logistic": {
                "raw_prob_up": raw_prob_up,
                "calibrated_prob_up": prob_up,
                "expected_return": expected,
                "std_future": std_future,
                "tolerance": tolerance,
                "strength_ratio": strength,
            },
            "returns_context": {
                "avg_up": avg_up,
                "avg_down": avg_down,
                "future_return_std": std_future,
            },
            "weights": weights.tolist(),
            "indicators": indicators,
            "calibration": calibration_params,
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
            default_confidence=55.0,
            default_reasoning="Defaulted to neutral after observation-only fallback.",
        )
        confidence = int(np.clip(decision.confidence, 0, 100))
        constraints = decision.constraints or {}

        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": confidence,
            "reasoning": decision.reasoning,
            "constraints": constraints,
            "indicators": indicators,
            "model_weights": weights.tolist(),
            "meta": {
                "observations": observations,
            },
        }

        progress.update_status(agent_id, ticker, f"Prob(up) {prob_up:.2%}, expected {expected:.3%} → {payload['signal'].upper()}")

        signals[ticker] = payload

    message = HumanMessage(content=json.dumps(signals), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(signals, "Growth Momentum Logistic Analyst")

    state["data"].setdefault("analyst_signals", {})[agent_id] = signals
    progress.update_status(agent_id, None, "Done")


async def growth_momentum_agent_async(state: AgentState, agent_id: str = "growth_momentum_agent"):
    """Async growth momentum analyst leveraging persona_from_observations."""

    data = state.get("data", {})
    tickers = data.get("tickers", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    extended_start = _extend_start(start_date)
    signals: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching price history")
        prices = await get_prices_async(
            ticker=ticker,
            start_date=extended_start or start_date,
            end_date=end_date,
            api_key=api_key,
        )

        if not prices:
            signals[ticker] = {
                "signal": "neutral",
                "confidence": 40,
                "reasoning": "Insufficient price data for logistic modelling.",
            }
            continue

        df = await prices_to_df_async(prices)
        if df.empty or len(df) < 90:
            signals[ticker] = {
                "signal": "neutral",
                "confidence": 42,
                "reasoning": "Need at least 90 observations to fit the classifier.",
            }
            continue

        df = df.sort_index()
        features_df, labels = _prepare_features(df["close"])
        if len(features_df) < 40 or labels.nunique() < 2:
            signals[ticker] = {
                "signal": "neutral",
                "confidence": 45,
                "reasoning": "Momentum classifier could not be trained (insufficient variation).",
            }
            continue

        feature_matrix = features_df.to_numpy(dtype=float)
        intercept = np.ones((feature_matrix.shape[0], 1))
        X = np.hstack([intercept, feature_matrix])
        y = labels.to_numpy(dtype=float)

        train_X = X[:-1]
        train_y = y[:-1]
        latest_features = X[-1]

        weights = _logistic_regression(train_X, train_y)
        logits = float(latest_features @ weights)
        raw_prob_up = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

        calibration_params = get_platt_parameters("growth_momentum", ticker)
        if calibration_params:
            prob_up = apply_platt(logits, calibration_params)
        else:
            prob_up = raw_prob_up

        future_returns = features_df.index.to_series().map(df["close"].pct_change().shift(-1)).dropna()
        up_returns = future_returns[future_returns > 0]
        down_returns = future_returns[future_returns <= 0]

        avg_up = float(up_returns.mean()) if not up_returns.empty else 0.01
        avg_down = float(down_returns.mean()) if not down_returns.empty else -0.01
        expected = _expected_value(prob_up, avg_up, avg_down)
        std_future = float(future_returns.std(ddof=0)) if len(future_returns) > 1 else 0.01

        denom = std_future if std_future > 0 else 0.01
        tolerance = 0.1 * denom
        strength = expected / denom

        indicators = {
            "prob_up": prob_up,
            "raw_prob_up": raw_prob_up,
            "expected_return": expected,
            "avg_up": avg_up,
            "avg_down": avg_down,
            "std_future": std_future,
            "base_rate": float(train_y.mean()),
            "logit": logits,
            "calibration_applied": bool(calibration_params),
        }
        if calibration_params:
            indicators["calibration"] = calibration_params

        observations = {
            "ticker": ticker,
            "logistic": {
                "raw_prob_up": raw_prob_up,
                "calibrated_prob_up": prob_up,
                "expected_return": expected,
                "std_future": std_future,
                "tolerance": tolerance,
                "strength_ratio": strength,
            },
            "returns_context": {
                "avg_up": avg_up,
                "avg_down": avg_down,
                "future_return_std": std_future,
            },
            "weights": weights.tolist(),
            "indicators": indicators,
            "calibration": calibration_params,
        }

        decision = await async_persona_from_observations(
            state=state,
            agent_id=agent_id,
            persona_name=PERSONA_NAME,
            persona_role=PERSONA_ROLE,
            persona_backstory=PERSONA_BACKSTORY,
            allowed_signals=ALLOWED_SIGNALS,
            observations=observations,
            persona_instructions=PERSONA_INSTRUCTIONS,
            default_signal="neutral",
            default_confidence=55.0,
            default_reasoning="Defaulted to neutral after observation-only fallback.",
        )
        confidence = int(np.clip(decision.confidence, 0, 100))
        constraints = decision.constraints or {}

        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": confidence,
            "reasoning": decision.reasoning,
            "constraints": constraints,
            "indicators": indicators,
            "model_weights": weights.tolist(),
            "meta": {
                "observations": observations,
            },
        }

        progress.update_status(agent_id, ticker, f"Prob(up) {prob_up:.2%}, expected {expected:.3%} → {payload['signal'].upper()}")
        signals[ticker] = payload

    message = HumanMessage(content=json.dumps(signals), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(signals, "Growth Momentum Logistic Analyst")

    await update_analyst_signals_async(state, agent_id, signals)
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }
