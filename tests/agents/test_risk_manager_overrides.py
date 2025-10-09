from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from src.agents.risk_manager import risk_management_agent
from src.data.models import Price


def _build_price_series(days: int = 90) -> list[Price]:
    base = datetime(2024, 1, 1)
    prices: list[Price] = []
    for offset in range(days):
        level = 100.0 + offset * 0.5
        stamp = (base + timedelta(days=offset)).strftime("%Y-%m-%d")
        prices.append(
            Price(
                open=level,
                close=level,
                high=level,
                low=level,
                volume=1_000_000,
                time=stamp,
            )
        )
    return prices


def _prices_to_df_stub(prices: list[Price]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [p.close for p in prices],
            "high": [p.high for p in prices],
            "low": [p.low for p in prices],
            "volume": [p.volume for p in prices],
        }
    )


def test_crash_allocator_target_short_overrides_event_bias(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TSLA": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "event_catalyst_agent": {
                    "TSLA": {"signal": "bullish", "constraints": {"preferred_direction": "long"}}
                },
                "macro_volatility_sentinel_agent": {
                    "TSLA": {"signal": "crash_alert", "constraints": {}, "metrics": {"close": 150.0}, "score": 0.9}
                },
                "downside_flow_sentinel_agent": {
                    "TSLA": {"signal": "downside_trend", "constraints": {}},
                },
                "crash_short_allocator_agent": {
                    "TSLA": {
                        "signal": "crash_short",
                        "confidence": 88,
                        "constraints": {
                            "preferred_direction": "short",
                            "allow_short": True,
                            "target_short_shares": 60,
                            "max_additional_short_shares": 60,
                            "max_short_exposure_pct": 0.25,
                            "max_long_exposure_pct": 0.02,
                            "max_additional_long_shares": 0,
                        },
                    }
                },
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "breakout_cover_sentinel_agent": {"TSLA": {"constraints": {}}},
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {"probabilities": {"rally": 0.3, "crash": 0.6}}}},
                "momentum_guardian_agent": {"TSLA": {"constraints": {}}},
                "trend_regime_agent": {
                    "TSLA": {"signal": "bearish", "confidence": 72, "constraints": {}}
                },
                "growth_momentum_agent": {
                    "TSLA": {
                        "constraints": {},
                        "indicators": {"prob_up": 0.42, "base_rate": 0.5},
                    }
                },
                "stat_mean_reversion_agent": {
                    "TSLA": {
                        "constraints": {},
                        "indicators": {"prob_revert_up": 0.4},
                    }
                },
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}
    constraint_notes = payload.get("reasoning", {}).get("constraints", [])

    assert overrides.get("preferred_direction") == "short"
    target_short = overrides.get("target_short_shares")
    assert isinstance(target_short, int) and target_short > 0
    assert any("CrashAllocator" in note for note in constraint_notes)


def test_crash_allocator_overrides_long_vote_blocks(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 150_000.0,
                "positions": {"NVDA": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 150_000.0,
            },
            "tickers": ["NVDA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "event_catalyst_agent": {
                    "NVDA": {
                        "signal": "bullish",
                        "confidence": 72,
                        "constraints": {
                            "preferred_direction": "long",
                            "reduce_position_change": True,
                        },
                    }
                },
                "trend_regime_agent": {
                    "NVDA": {
                        "signal": "bullish",
                        "confidence": 85,
                        "constraints": {"preferred_direction": "long"},
                    }
                },
                "growth_momentum_agent": {
                    "NVDA": {
                        "signal": "acceleration",
                        "confidence": 80,
                        "constraints": {"preferred_direction": "long"},
                        "indicators": {"prob_up": 0.78, "base_rate": 0.5},
                    }
                },
                "momentum_guardian_agent": {
                    "NVDA": {
                        "signal": "bullish",
                        "confidence": 82,
                        "constraints": {"preferred_direction": "long"},
                    }
                },
                "regime_meta_agent": {
                    "NVDA": {
                        "signal": "crash",
                        "confidence": 76,
                        "constraints": {"preferred_direction": "short"},
                        "indicators": {"probabilities": {"rally": 0.24, "crash": 0.58}},
                    }
                },
                "crash_short_allocator_agent": {
                    "NVDA": {
                        "signal": "crash_short",
                        "confidence": 84,
                        "allocation_pct": 0.12,
                        "constraints": {
                            "preferred_direction": "short",
                            "allow_short": True,
                            "target_short_shares": 60,
                            "max_additional_short_shares": 60,
                            "max_short_exposure_pct": 0.3,
                        },
                    }
                },
                "downside_flow_sentinel_agent": {
                    "NVDA": {
                        "signal": "downside_trend",
                        "confidence": 68,
                        "constraints": {},
                    }
                },
                "macro_volatility_sentinel_agent": {
                    "NVDA": {"signal": "watch", "constraints": {}}
                },
                "short_squeeze_guardian_agent": {"NVDA": {"constraints": {}}},
                "short_cover_classifier_agent": {"NVDA": {"constraints": {}}},
                "breakout_cover_sentinel_agent": {"NVDA": {"constraints": {}}},
                "range_recovery_sentinel_agent": {"NVDA": {"constraints": {}}},
                "stat_mean_reversion_agent": {"NVDA": {"constraints": {}, "indicators": {}}},
                "stop_loss_guardian_agent": {"NVDA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["NVDA"]
    overrides = payload.get("overrides") or {}

    assert overrides.get("block_new_shorts") is not True
    max_additional_short = overrides.get("max_additional_short_shares")
    assert isinstance(max_additional_short, int) and max_additional_short > 0
    target_short = overrides.get("target_short_shares")
    assert isinstance(target_short, int) and target_short > 0

    notes = payload.get("reasoning", {}).get("constraints", [])
    assert any("Crash allocator override" in note for note in notes)


def test_crash_override_reopens_short_capacity_when_directional_caps_zero(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 125_000.0,
                "positions": {"NVDA": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 125_000.0,
            },
            "tickers": ["NVDA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "range_recovery_sentinel_agent": {
                    "NVDA": {
                        "signal": "breakdown",
                        "confidence": 68,
                        "constraints": {"max_short_exposure_pct": 0.0},
                    }
                },
                "momentum_guardian_agent": {
                    "NVDA": {
                        "signal": "bullish",
                        "confidence": 82,
                        "constraints": {
                            "preferred_direction": "long",
                            "max_short_exposure_pct": 0.0,
                        },
                    }
                },
                "regime_meta_agent": {
                    "NVDA": {
                        "signal": "crash",
                        "confidence": 79,
                        "constraints": {"preferred_direction": "short"},
                        "indicators": {"probabilities": {"rally": 0.2, "crash": 0.62}},
                    }
                },
                "downside_flow_sentinel_agent": {
                    "NVDA": {
                        "signal": "crash_flow",
                        "confidence": 74,
                        "constraints": {},
                    }
                },
                "macro_volatility_sentinel_agent": {
                    "NVDA": {
                        "signal": "crash_alert",
                        "confidence": 80,
                        "constraints": {"reduce_position_change": True},
                        "metrics": {"close": 155.0},
                    }
                },
                "crash_short_allocator_agent": {
                    "NVDA": {
                        "signal": "crash_short",
                        "confidence": 88,
                        "allocation_pct": 0.18,
                        "constraints": {
                            "preferred_direction": "short",
                            "allow_short": True,
                            "target_short_shares": 80,
                            "max_additional_short_shares": 80,
                            "max_short_exposure_pct": 0.24,
                        },
                    }
                },
                "short_squeeze_guardian_agent": {"NVDA": {"constraints": {}}},
                "short_cover_classifier_agent": {"NVDA": {"constraints": {}}},
                "breakout_cover_sentinel_agent": {"NVDA": {"constraints": {}}},
                "stat_mean_reversion_agent": {"NVDA": {"constraints": {}, "indicators": {}}},
                "stop_loss_guardian_agent": {"NVDA": {"constraints": {}}},
                "event_catalyst_agent": {"NVDA": {"constraints": {}}},
                "growth_momentum_agent": {
                    "NVDA": {
                        "constraints": {},
                        "indicators": {"prob_up": 0.41, "base_rate": 0.52},
                    }
                },
                "trend_regime_agent": {
                    "NVDA": {
                        "signal": "bullish",
                        "confidence": 73,
                        "constraints": {"preferred_direction": "long"},
                    }
                },
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["NVDA"]
    overrides = payload.get("overrides") or {}

    assert overrides.get("block_new_shorts") is not True
    assert overrides.get("max_additional_short_shares", 0) > 0
    assert overrides.get("target_short_shares", 0) > 0

    notes = payload.get("reasoning", {}).get("constraints", [])
    assert any("CrashAllocator" in note for note in notes)


def test_force_cover_clamps_target_short(monkeypatch):
    series = _build_price_series()

    monkeypatch.setattr(
        "src.agents.risk_manager.get_prices",
        lambda *_, **__: series,
    )

    existing_short = 17

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 103_500.0,
                "positions": {"NVDA": {"long": 0, "short": existing_short}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["NVDA"],
            "start_date": "2025-06-01",
            "end_date": "2025-07-17",
            "analyst_signals": {
                "crash_short_allocator_agent": {
                    "NVDA": {
                        "signal": "cover_short",
                        "confidence": 86,
                        "constraints": {
                            "preferred_direction": "neutral",
                            "allow_short": True,
                            "target_short_shares": 0,
                            "max_additional_short_shares": 0,
                            "max_short_exposure_pct": 0.0,
                            "block_new_shorts": True,
                            "force_cover_qty": existing_short,
                            "force_cover_reason": "Crash allocator releasing short exposure",
                            "reference_price": 170.0,
                        },
                        "indicators": {"allocation_pct": 0.0},
                    }
                },
                "short_squeeze_guardian_agent": {
                    "NVDA": {
                        "signal": "elevated_risk",
                        "confidence": 60,
                        "constraints": {
                            "block_new_shorts": True,
                            "target_short_shares": existing_short,
                        },
                    }
                },
                "trend_regime_agent": {
                    "NVDA": {
                        "signal": "bullish",
                        "confidence": 72,
                        "constraints": {
                            "preferred_direction": "long",
                            "target_short_shares": 0,
                        },
                    }
                },
                "regime_meta_agent": {
                    "NVDA": {
                        "signal": "rally",
                        "confidence": 68,
                        "constraints": {"preferred_direction": "long"},
                        "indicators": {"probabilities": {"rally": 0.62, "crash": 0.31}},
                    }
                },
                "event_catalyst_agent": {
                    "NVDA": {
                        "signal": "bullish",
                        "confidence": 58,
                        "constraints": {"preferred_direction": "long"},
                    }
                },
                "growth_momentum_agent": {
                    "NVDA": {
                        "constraints": {"preferred_direction": "long"},
                        "indicators": {"prob_up": 0.6, "base_rate": 0.55},
                    }
                },
                "stat_mean_reversion_agent": {
                    "NVDA": {
                        "constraints": {"preferred_direction": "short"},
                        "indicators": {"prob_revert_up": 0.45},
                    }
                },
                "macro_volatility_sentinel_agent": {
                    "NVDA": {"signal": "watch", "constraints": {}}
                },
                "downside_flow_sentinel_agent": {
                    "NVDA": {"signal": "neutral", "constraints": {}}
                },
                "short_cover_classifier_agent": {"NVDA": {"constraints": {}}},
                "breakout_cover_sentinel_agent": {"NVDA": {"constraints": {}}},
                "range_recovery_sentinel_agent": {"NVDA": {"constraints": {}}},
                "momentum_guardian_agent": {"NVDA": {"constraints": {}}},
                "stop_loss_guardian_agent": {"NVDA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["NVDA"]
    overrides = payload.get("overrides") or {}

    assert overrides.get("force_cover_qty") == existing_short
    assert overrides.get("target_short_shares") == 0


def test_crash_override_ignores_directional_zero_targets(monkeypatch):
    series = _build_price_series()

    monkeypatch.setattr(
        "src.agents.risk_manager.get_prices",
        lambda *_, **__: series,
    )

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 140_000.0,
                "positions": {"NVDA": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 140_000.0,
            },
            "tickers": ["NVDA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "trend_regime_agent": {
                    "NVDA": {
                        "signal": "bullish",
                        "confidence": 82,
                        "constraints": {
                            "preferred_direction": "long",
                            "target_short_shares": 0,
                            "max_short_exposure_pct": 0.0,
                        },
                    }
                },
                "regime_meta_agent": {
                    "NVDA": {
                        "signal": "crash",
                        "confidence": 80,
                        "constraints": {"preferred_direction": "short"},
                        "indicators": {"probabilities": {"rally": 0.22, "crash": 0.63}},
                    }
                },
                "crash_short_allocator_agent": {
                    "NVDA": {
                        "signal": "crash_short",
                        "confidence": 90,
                        "allocation_pct": 0.16,
                        "constraints": {
                            "preferred_direction": "short",
                            "allow_short": True,
                            "target_short_shares": 70,
                            "max_additional_short_shares": 70,
                            "max_short_exposure_pct": 0.25,
                        },
                    }
                },
                "downside_flow_sentinel_agent": {
                    "NVDA": {"signal": "crash_flow", "constraints": {}}
                },
                "macro_volatility_sentinel_agent": {
                    "NVDA": {"signal": "crash_alert", "constraints": {}}
                },
                "short_squeeze_guardian_agent": {"NVDA": {"constraints": {}}},
                "short_cover_classifier_agent": {"NVDA": {"constraints": {}}},
                "breakout_cover_sentinel_agent": {"NVDA": {"constraints": {}}},
                "range_recovery_sentinel_agent": {"NVDA": {"constraints": {}}},
                "momentum_guardian_agent": {"NVDA": {"constraints": {}}},
                "growth_momentum_agent": {"NVDA": {"constraints": {}}},
                "stat_mean_reversion_agent": {"NVDA": {"constraints": {}}},
                "stop_loss_guardian_agent": {"NVDA": {"constraints": {}}},
                "event_catalyst_agent": {"NVDA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["NVDA"]
    overrides = payload.get("overrides") or {}

    assert overrides.get("target_short_shares", 0) > 0
    assert overrides.get("block_new_shorts") is not True
    assert overrides.get("preferred_direction") == "short"

    notes = payload.get("reasoning", {}).get("constraints", [])
    assert any("Crash allocator override" in note for note in notes)
    assert any("CrashAllocator target" in note for note in notes)
    assert not any("Trend target 0" in note for note in notes)


def test_risk_manager_short_preference_blocks_long(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TSLA": {"long": 0, "short": 200}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "momentum_guardian_agent": {"TSLA": {"constraints": {"preferred_direction": "short"}}},
                "trend_regime_agent": {"TSLA": {"constraints": {}}},
                "growth_momentum_agent": {
                    "TSLA": {
                        "constraints": {"preferred_direction": "short"},
                        "indicators": {"prob_up": 0.82, "base_rate": 0.45},
                    }
                },
                "stat_mean_reversion_agent": {"TSLA": {"constraints": {}, "indicators": {"prob_revert_up": 0.6}}},
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    overrides = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"].get("overrides") or {}

    assert overrides.get("preferred_direction") == "short"
    assert "target_long_shares" not in overrides
    assert "force_buy_reason" not in overrides

    constraints = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]["reasoning"].get("constraints", [])
    assert not any("long" in note.lower() and "target" in note.lower() for note in constraints)


def test_risk_manager_applies_long_target_when_allowed(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TSLA": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "momentum_guardian_agent": {"TSLA": {"constraints": {}}},
                "trend_regime_agent": {"TSLA": {"constraints": {"preferred_direction": "long"}, "confidence": 70}},
                "growth_momentum_agent": {
                    "TSLA": {
                        "constraints": {"preferred_direction": "long"},
                        "indicators": {"prob_up": 0.74, "base_rate": 0.42},
                    }
                },
                "stat_mean_reversion_agent": {"TSLA": {"constraints": {}, "indicators": {"prob_revert_up": 0.7}}},
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {"probabilities": {"rally": 0.65, "crash": 0.2}}}},
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}

    assert overrides.get("preferred_direction") == "long"
    assert overrides.get("target_long_shares", 0) > 0
    assert overrides.get("force_buy_reason") == "Probabilistic long allocation"

    constraints = payload["reasoning"].get("constraints", [])
    assert any("allocating long exposure" in note.lower() for note in constraints)


def test_bearish_trend_suppresses_mean_reversion_longs(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TSLA": {"long": 0, "short": 100}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "momentum_guardian_agent": {"TSLA": {"constraints": {}}},
                "trend_regime_agent": {"TSLA": {"constraints": {}, "signal": "bearish", "confidence": 78}},
                "growth_momentum_agent": {"TSLA": {"constraints": {}}},
                "stat_mean_reversion_agent": {
                    "TSLA": {
                        "constraints": {
                            "preferred_direction": "long",
                            "max_short_exposure_pct": 0.05,
                        },
                        "indicators": {"prob_revert_up": 0.68},
                    }
                },
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}
    constraint_notes = payload.get("reasoning", {}).get("constraints", [])

    assert overrides.get("preferred_direction") != "long"
    assert "target_long_shares" not in overrides
    assert any("suppressed mean reversion long bias" in note.lower() for note in constraint_notes)
    assert not any("meanreversion cap" in note.lower() for note in constraint_notes)


def test_bullish_trend_blocks_new_shorts_even_with_mean_reversion_short(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TSLA": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "momentum_guardian_agent": {"TSLA": {"constraints": {}}},
                "trend_regime_agent": {"TSLA": {"constraints": {}, "signal": "bullish", "confidence": 70}},
                "growth_momentum_agent": {"TSLA": {"constraints": {}}},
                "stat_mean_reversion_agent": {
                    "TSLA": {
                        "constraints": {
                            "preferred_direction": "short",
                            "max_short_exposure_pct": 0.2,
                        },
                        "indicators": {"prob_revert_up": 0.35},
                    }
                },
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}
    constraint_notes = payload.get("reasoning", {}).get("constraints", [])

    assert overrides.get("block_new_shorts") is True
    assert overrides.get("max_additional_short_shares") == 0
    assert "target_short_shares" not in overrides or overrides.get("target_short_shares") == 0
    assert not overrides.get("preferred_direction") == "short"
    assert any("blocking new shorts" in note.lower() for note in constraint_notes)


def test_event_catalyst_preferred_direction_respected(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 150_000.0,
                "positions": {"TSLA": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 150_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "momentum_guardian_agent": {"TSLA": {"constraints": {}}},
                "trend_regime_agent": {"TSLA": {"constraints": {}}},
                "growth_momentum_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stat_mean_reversion_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
                "event_catalyst_agent": {
                    "TSLA": {
                        "signal": "bullish",
                        "confidence": 76,
                        "constraints": {
                            "preferred_direction": "long",
                            "max_short_exposure_pct": 0.05,
                        },
                    }
                },
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    notes = payload.get("reasoning", {}).get("constraints", [])

    assert overrides.get("preferred_direction") == "long"
    assert overrides.get("block_new_shorts") is True
    assert any("eventcatalyst" in note.lower() for note in notes)


def test_risk_limit_caps_crash_allocator_targets(monkeypatch):
    series = _build_price_series()
    last_price = float(series[-1].close)

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    def fake_vol_limit(_volatility: float) -> float:
        return 0.04  # enforce a 4% portfolio cap regardless of realised vol

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)
    monkeypatch.setattr(
        "src.agents.risk_manager.calculate_volatility_adjusted_limit",
        fake_vol_limit,
    )

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TSLA": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "crash_short_allocator_agent": {
                    "TSLA": {
                        "constraints": {
                            "preferred_direction": "short",
                            "allow_short": True,
                            "target_short_shares": 60,
                            "max_additional_short_shares": 60,
                            "max_short_exposure_pct": 0.5,
                        }
                    }
                },
                "trend_regime_agent": {"TSLA": {"constraints": {}}},
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "breakout_cover_sentinel_agent": {"TSLA": {"constraints": {}}},
                "momentum_guardian_agent": {"TSLA": {"constraints": {}}},
                "regime_meta_agent": {"TSLA": {"constraints": {}}},
                "growth_momentum_agent": {"TSLA": {"constraints": {}}},
                "stat_mean_reversion_agent": {"TSLA": {"constraints": {}}},
                "downside_flow_sentinel_agent": {"TSLA": {"constraints": {}}},
                "macro_volatility_sentinel_agent": {"TSLA": {"constraints": {}}},
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}

    expected_cap_shares = int((100_000.0 * 0.04) // last_price)
    assert overrides.get("target_short_shares") == expected_cap_shares
    assert overrides.get("max_additional_short_shares") == expected_cap_shares

    notes = payload.get("reasoning", {}).get("constraints", [])
    assert any(
        "risk limit caps" in note.lower() or "short target trimmed" in note.lower()
        for note in notes
    )


def test_range_recovery_constraints_drive_short_trim(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TSLA": {"long": 0, "short": 6}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "short_cover_classifier_agent": {"TSLA": {"constraints": {}}},
                "breakout_cover_sentinel_agent": {"TSLA": {"constraints": {}}},
                "range_recovery_sentinel_agent": {
                    "TSLA": {
                        "signal": "drift_recovery",
                        "constraints": {
                            "block_new_shorts": True,
                            "allow_short": False,
                            "target_short_shares": 2,
                            "force_cover_qty": 4,
                            "force_cover_reason": "Range recovery sentinel trimming stale shorts",
                        },
                    }
                },
                "momentum_guardian_agent": {"TSLA": {"constraints": {}}},
                "trend_regime_agent": {"TSLA": {"constraints": {}}},
                "growth_momentum_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stat_mean_reversion_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "event_catalyst_agent": {"TSLA": {"constraints": {}}},
                "macro_volatility_sentinel_agent": {"TSLA": {"constraints": {}}},
                "downside_flow_sentinel_agent": {"TSLA": {"constraints": {}}},
                "crash_short_allocator_agent": {"TSLA": {"constraints": {}}},
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}
    notes = payload.get("reasoning", {}).get("constraints", [])

    assert overrides.get("force_cover_qty") == 4
    assert overrides.get("force_cover_reason") == "Range recovery sentinel trimming stale shorts"
    assert overrides.get("target_short_shares") == 2
    assert overrides.get("block_new_shorts") is True
    assert overrides.get("max_additional_short_shares") == 0
    assert any("rangerecovery blocks new shorts" in note.lower() for note in notes)
    assert any("rangerecovery target 2 shorts" in note.lower() for note in notes)


def test_max_short_cap_preserves_long_limit(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TSLA": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "event_catalyst_agent": {
                    "TSLA": {
                        "signal": "bullish",
                        "confidence": 70,
                        "constraints": {
                            "preferred_direction": "long",
                            "max_short_exposure_pct": 0.0,
                        },
                    }
                },
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "momentum_guardian_agent": {"TSLA": {"constraints": {}}},
                "trend_regime_agent": {
                    "TSLA": {"constraints": {"preferred_direction": "long"}, "confidence": 72}
                },
                "growth_momentum_agent": {
                    "TSLA": {
                        "constraints": {},
                        "indicators": {"prob_up": 0.62, "base_rate": 0.5},
                    }
                },
                "stat_mean_reversion_agent": {
                    "TSLA": {"constraints": {}, "indicators": {"prob_revert_up": 0.4}}
                },
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "downside_flow_sentinel_agent": {"TSLA": {"constraints": {}}},
                "crash_short_allocator_agent": {"TSLA": {"constraints": {}}},
                "macro_volatility_sentinel_agent": {"TSLA": {"constraints": {}}},
                "short_cover_classifier_agent": {"TSLA": {"constraints": {}}},
                "breakout_cover_sentinel_agent": {"TSLA": {"constraints": {}}},
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}
    reasoning = payload.get("reasoning", {})
    notes = reasoning.get("constraints", [])

    assert reasoning.get("position_limit") > 0
    assert overrides.get("target_short_shares") == 0
    assert overrides.get("max_additional_short_shares") == 0
    assert overrides.get("block_new_shorts") is True
    assert any("eventcatalyst" in note.lower() and "short cap" in note.lower() for note in notes)


def test_event_catalyst_limits_long_allocation_and_shorts(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 300_000.0,
                "positions": {"TSLA": {"long": 100, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 300_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "momentum_guardian_agent": {
                    "TSLA": {"constraints": {"target_long_shares": 180}, "confidence": 60}
                },
                "trend_regime_agent": {"TSLA": {"constraints": {}}},
                "growth_momentum_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stat_mean_reversion_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
                "event_catalyst_agent": {
                    "TSLA": {
                        "signal": "neutral",
                        "confidence": 55,
                        "constraints": {
                            "reduce_position_change": True,
                            "max_long_add": 20,
                        },
                    }
                },
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}
    notes = payload.get("reasoning", {}).get("constraints", [])

    assert overrides.get("target_long_shares") == 120
    assert overrides.get("max_additional_short_shares") == 0
    assert overrides.get("block_new_shorts") is True
    assert any("eventcatalyst" in note.lower() for note in notes)


def test_volatility_sentinel_blocks_long_adds(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 200_000.0,
                "positions": {"TSLA": {"long": 90, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 200_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "momentum_guardian_agent": {
                    "TSLA": {"constraints": {"target_long_shares": 180}, "confidence": 70}
                },
                "trend_regime_agent": {"TSLA": {"constraints": {}}},
                "growth_momentum_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stat_mean_reversion_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "macro_volatility_sentinel_agent": {
                    "TSLA": {
                        "constraints": {
                            "max_additional_long_shares": 0,
                            "max_long_exposure_pct": 0.08,
                            "reduce_position_change": True,
                        },
                        "confidence": 80,
                    }
                },
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}

    assert overrides.get("target_long_shares", 90) <= 90
    notes = payload.get("reasoning", {}).get("constraints", [])
    assert any("volatilitysentinel" in note.lower() for note in notes)


def test_volatility_sentinel_trims_existing_long(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 120_000.0,
                "positions": {"TSLA": {"long": 120, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 120_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "momentum_guardian_agent": {"TSLA": {"constraints": {}}},
                "trend_regime_agent": {"TSLA": {"constraints": {}}},
                "growth_momentum_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stat_mean_reversion_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "macro_volatility_sentinel_agent": {
                    "TSLA": {
                        "constraints": {
                            "max_long_shares": 60,
                            "preferred_direction": "short",
                        },
                        "confidence": 85,
                    }
                },
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}

    assert overrides.get("target_long_shares") == 60
    assert overrides.get("force_buy_reason") is None


def test_volatility_sentinel_caps_position_limit(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)
    monkeypatch.setattr("src.agents.risk_manager.calculate_volatility_adjusted_limit", lambda _: 0.30)
    monkeypatch.setattr("src.agents.risk_manager.calculate_correlation_multiplier", lambda _: 1.0)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 250_000.0,
                "positions": {"TSLA": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 250_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "momentum_guardian_agent": {"TSLA": {"constraints": {}}},
                "trend_regime_agent": {"TSLA": {"constraints": {}}},
                "growth_momentum_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stat_mean_reversion_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "macro_volatility_sentinel_agent": {
                    "TSLA": {
                        "constraints": {
                            "max_long_exposure_pct": 0.05,
                        },
                        "confidence": 70,
                    }
                },
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    reasoning = payload.get("reasoning", {})

    expected_cap = 250_000.0 * 0.05
    assert abs(reasoning.get("position_limit") - expected_cap) < 1e-6
    notes = reasoning.get("constraints", [])
    assert any("long cap" in note.lower() for note in notes)


def test_breakout_cover_full_force_cover(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    breakout_constraints = {
        "preferred_direction": "long",
        "block_new_shorts": True,
        "allow_short": False,
        "target_short_shares": 0,
        "force_cover_qty": 20,
        "force_cover_reason": "Breakout cover sentinel forcing full unwind",
    }

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TSLA": {"long": 0, "short": 20}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "breakout_cover_sentinel_agent": {"TSLA": {"constraints": breakout_constraints, "confidence": 90}},
                "momentum_guardian_agent": {"TSLA": {"constraints": {}}},
                "trend_regime_agent": {"TSLA": {"constraints": {}, "confidence": 40}},
                "growth_momentum_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stat_mean_reversion_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "event_catalyst_agent": {"TSLA": {"constraints": {}}},
                "macro_volatility_sentinel_agent": {"TSLA": {"constraints": {}}},
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}
    notes = payload.get("reasoning", {}).get("constraints", [])

    assert overrides.get("preferred_direction") == "long"
    assert overrides.get("block_new_shorts") is True
    assert overrides.get("target_short_shares") == 0
    assert overrides.get("force_cover_qty") == 20
    assert overrides.get("force_cover_reason") == "Breakout cover sentinel forcing full unwind"
    assert any("BreakoutCover target 0 shorts" in note for note in notes)


def test_breakout_cover_trim_respects_partial_cover(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    breakout_constraints = {
        "preferred_direction": "long",
        "block_new_shorts": True,
        "allow_short": True,
        "target_short_shares": 6,
        "force_cover_qty": 14,
        "force_cover_reason": "Breakout cover sentinel trimming shorts",
        "max_short_exposure_pct": 0.02,
    }

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TSLA": {"long": 0, "short": 20}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "breakout_cover_sentinel_agent": {"TSLA": {"constraints": breakout_constraints, "confidence": 75}},
                "momentum_guardian_agent": {"TSLA": {"constraints": {}}},
                "trend_regime_agent": {"TSLA": {"constraints": {}, "confidence": 40}},
                "growth_momentum_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stat_mean_reversion_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "event_catalyst_agent": {"TSLA": {"constraints": {}}},
                "macro_volatility_sentinel_agent": {"TSLA": {"constraints": {}}},
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}
    notes = payload.get("reasoning", {}).get("constraints", [])

    assert overrides.get("preferred_direction") == "long"
    assert overrides.get("block_new_shorts") is True
    assert overrides.get("target_short_shares") == 6
    assert overrides.get("force_cover_qty") == 14
    assert overrides.get("force_cover_reason") == "Breakout cover sentinel trimming shorts"
    assert any("BreakoutCover target 6 shorts" in note for note in notes)


def test_short_preference_relaxes_when_shorts_blocked(monkeypatch):
    series = _build_price_series()

    monkeypatch.setattr(
        "src.agents.risk_manager.get_prices",
        lambda *_, **__: series,
    )

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {
                    "TSLA": {
                        "long": 0,
                        "short": 0,
                        "long_cost_basis": 0.0,
                        "short_cost_basis": 0.0,
                    }
                },
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-07-01",
            "end_date": "2024-07-22",
            "analyst_signals": {
                "downside_flow_sentinel_agent": {"TSLA": {"constraints": {}, "confidence": 55}},
                "crash_short_allocator_agent": {"TSLA": {"constraints": {}}},
                "event_catalyst_agent": {
                    "TSLA": {
                        "constraints": {"preferred_direction": "short", "max_long_add": 0},
                        "confidence": 70,
                    }
                },
                "macro_volatility_sentinel_agent": {"TSLA": {"constraints": {}}},
                "short_squeeze_guardian_agent": {
                    "TSLA": {
                        "constraints": {
                            "block_new_shorts": True,
                            "max_additional_short_shares": 0,
                            "target_short_shares": 6,
                        }
                    }
                },
                "short_cover_classifier_agent": {"TSLA": {"constraints": {}}},
                "breakout_cover_sentinel_agent": {"TSLA": {"constraints": {}}},
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "momentum_guardian_agent": {"TSLA": {"constraints": {}}},
                "trend_regime_agent": {"TSLA": {"constraints": {}, "confidence": 50}},
                "growth_momentum_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stat_mean_reversion_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}
    notes = payload.get("reasoning", {}).get("constraints", [])

    assert overrides.get("block_new_shorts") is True
    assert overrides.get("target_short_shares") == 0
    assert overrides.get("preferred_direction") == "neutral"
    assert any("short target clamped" in note.lower() for note in notes)
    assert any("short bias relaxed to neutral" in note.lower() for note in notes)


def test_risk_manager_honors_short_cover_force_unwind(monkeypatch):
    series = _build_price_series()

    monkeypatch.setattr(
        "src.agents.risk_manager.get_prices",
        lambda *_, **__: series,
    )

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TSLA": {"long": 0, "short": 40}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_cover_classifier_agent": {
                    "TSLA": {
                        "signal": "squeeze_cover",
                        "constraints": {
                            "preferred_direction": "long",
                            "block_new_shorts": True,
                            "allow_short": False,
                            "target_short_shares": 0,
                            "force_cover_qty": 40,
                            "force_cover_reason": "Classifier expects upside squeeze; unwind remaining shorts",
                        },
                    }
                },
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "trend_regime_agent": {"TSLA": {"constraints": {}}},
                "growth_momentum_agent": {
                    "TSLA": {
                        "constraints": {},
                        "indicators": {"prob_up": 0.55, "base_rate": 0.52},
                    }
                },
                "stat_mean_reversion_agent": {"TSLA": {"constraints": {}, "indicators": {"prob_revert_up": 0.55}}},
                "regime_meta_agent": {"TSLA": {"constraints": {}, "indicators": {}}},
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    overrides = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"].get("overrides") or {}

    assert overrides.get("force_cover_qty") == 40
    assert overrides.get("preferred_direction") == "long"
    assert overrides.get("target_short_shares") == 0
    assert overrides.get("block_new_shorts") is True

def test_crash_mode_relaxes_short_squeeze_after_streak(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    base_state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 200_000.0,
                "positions": {"NVDA": {"long": 0, "short": 0}},
            },
            "tickers": ["NVDA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_squeeze_guardian_agent": {
                    "NVDA": {
                        "constraints": {
                            "block_new_shorts": True,
                            "max_additional_short_shares": 0,
                            "max_short_exposure_pct": 0.0,
                            "allow_short": False,
                        }
                    }
                },
                "crash_short_allocator_agent": {
                    "NVDA": {
                        "signal": "crash_short",
                        "confidence": 92,
                        "allocation_pct": 0.18,
                        "constraints": {
                            "preferred_direction": "short",
                            "allow_short": True,
                            "target_short_shares": 60,
                            "max_additional_short_shares": 60,
                            "max_short_exposure_pct": 0.25,
                            "raw_target_short_shares": 60,
                        },
                        "indicators": {
                            "raw_target_short_shares": 60,
                        },
                    }
                },
                "regime_meta_agent": {
                    "NVDA": {
                        "constraints": {},
                        "indicators": {"probabilities": {"rally": 0.3, "crash": 0.6}},
                    }
                },
                "short_cover_classifier_agent": {"NVDA": {"constraints": {}}},
                "breakout_cover_sentinel_agent": {"NVDA": {"constraints": {}}},
                "range_recovery_sentinel_agent": {"NVDA": {"constraints": {}}},
                "momentum_guardian_agent": {"NVDA": {"constraints": {}}},
                "trend_regime_agent": {"NVDA": {"constraints": {}}},
                "growth_momentum_agent": {"NVDA": {"constraints": {}}},
                "stat_mean_reversion_agent": {"NVDA": {"constraints": {}}},
                "event_catalyst_agent": {"NVDA": {"constraints": {}}},
                "macro_volatility_sentinel_agent": {"NVDA": {"constraints": {}}},
                "downside_flow_sentinel_agent": {"NVDA": {"constraints": {}}},
                "stop_loss_guardian_agent": {"NVDA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    first_result = risk_management_agent(base_state)
    first_payload = first_result["data"]["analyst_signals"]["risk_management_agent"]["NVDA"]
    first_overrides = first_payload.get("overrides") or {}
    first_constraints = first_payload.get("reasoning", {}).get("constraints", [])

    assert first_overrides.get("crash_allocator_raw_target_shares") == 60
    assert first_overrides.get("block_new_shorts") is True
    assert any("Crash allocator raw target" in note for note in first_constraints)

    second_state = {
        "messages": first_result["messages"],
        "data": first_result["data"],
        "metadata": base_state["metadata"],
    }

    second_result = risk_management_agent(second_state)
    second_payload = second_result["data"]["analyst_signals"]["risk_management_agent"]["NVDA"]
    second_overrides = second_payload.get("overrides") or {}
    second_constraints = second_payload.get("reasoning", {}).get("constraints", [])
    crash_state = second_result["data"].get("risk_manager_state", {}).get("crash_mode", {}).get("NVDA", {})

    assert crash_state.get("active") is True
    assert crash_state.get("streak") >= 2
    assert second_overrides.get("crash_allocator_raw_target_shares") == 60
    assert not second_overrides.get("block_new_shorts")
    assert any("Crash probability streak activated crash-mode override" in note for note in second_constraints)
    assert any("Crash mode relaxes ShortSqueeze guardrails" in note for note in second_constraints)


def test_short_target_never_exceeds_short_capacity(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TSLA": {"long": 5, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "event_catalyst_agent": {
                    "TSLA": {
                        "signal": "neutral",
                        "constraints": {"preferred_direction": "short"},
                    }
                },
                "macro_volatility_sentinel_agent": {
                    "TSLA": {"signal": "watch", "constraints": {}}},
                "downside_flow_sentinel_agent": {
                    "TSLA": {"signal": "idle", "constraints": {}}},
                "crash_short_allocator_agent": {
                    "TSLA": {
                        "signal": "crash_short",
                        "confidence": 88,
                        "constraints": {
                            "preferred_direction": "short",
                            "target_short_shares": 34,
                            "max_additional_short_shares": 34,
                            "max_short_exposure_pct": 0.35,
                        },
                    }
                },
                "short_squeeze_guardian_agent": {
                    "TSLA": {"constraints": {"block_new_shorts": False}}
                },
                "short_cover_classifier_agent": {"TSLA": {"constraints": {}}},
                "breakout_cover_sentinel_agent": {"TSLA": {"constraints": {}}},
                "range_recovery_sentinel_agent": {
                    "TSLA": {"constraints": {"max_additional_short_shares": 6}}
                },
                "regime_meta_agent": {
                    "TSLA": {
                        "constraints": {},
                        "indicators": {"probabilities": {"rally": 0.55, "crash": 0.22}},
                    }
                },
                "momentum_guardian_agent": {
                    "TSLA": {
                        "signal": "bullish_momentum",
                        "confidence": 92,
                        "constraints": {"allow_short": False, "max_short_exposure_pct": 0.0},
                    }
                },
                "trend_regime_agent": {
                    "TSLA": {
                        "signal": "bullish",
                        "confidence": 82,
                        "constraints": {
                            "preferred_direction": "long",
                            "max_short_exposure_pct": 0.02,
                        },
                    }
                },
                "growth_momentum_agent": {
                    "TSLA": {
                        "signal": "neutral",
                        "confidence": 55,
                        "constraints": {},
                        "indicators": {"prob_up": 0.51, "base_rate": 0.5},
                    }
                },
                "stat_mean_reversion_agent": {
                    "TSLA": {"constraints": {}, "indicators": {}}},
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    overrides = (
        result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
        .get("overrides")
        or {}
    )

    existing_short = state["data"]["portfolio"]["positions"]["TSLA"].get("short", 0)
    target = overrides.get("target_short_shares")
    max_additional = overrides.get("max_additional_short_shares")

    if isinstance(target, int) and isinstance(max_additional, int):
        assert target <= existing_short + max_additional

def test_bearish_consensus_relaxes_directional_blocks(monkeypatch):
    series = _build_price_series()

    monkeypatch.setattr(
        "src.agents.risk_manager.get_prices",
        lambda *_, **__: series,
    )

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TSLA": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2025-08-01",
            "end_date": "2025-09-12",
            "analyst_signals": {
                "aswath_damodaran_agent": {
                    "TSLA": {"signal": "bearish", "confidence": 88}
                },
                "ben_graham_agent": {
                    "TSLA": {"signal": "bearish", "confidence": 84}
                },
                "michael_burry_agent": {
                    "TSLA": {"signal": "bearish", "confidence": 82}
                },
                "charlie_munger_agent": {
                    "TSLA": {"signal": "bearish", "confidence": 80}
                },
                "technical_analyst_agent": {
                    "TSLA": {"signal": "bearish", "confidence": 72}
                },
                "stat_mean_reversion_agent": {
                    "TSLA": {"signal": "bearish", "confidence": 70, "constraints": {}}
                },
                "momentum_guardian_agent": {
                    "TSLA": {
                        "signal": "bullish_momentum",
                        "confidence": 92,
                        "constraints": {
                            "allow_short": False,
                            "block_new_shorts": True,
                            "max_short_exposure_pct": 0.0,
                            "preferred_direction": "long",
                        },
                    }
                },
                "trend_regime_agent": {
                    "TSLA": {
                        "signal": "bullish",
                        "confidence": 80,
                        "constraints": {
                            "preferred_direction": "long",
                            "allow_short": False,
                            "block_new_shorts": True,
                            "target_short_shares": 0,
                        },
                    }
                },
                "range_recovery_sentinel_agent": {
                    "TSLA": {
                        "signal": "range_monitor",
                        "confidence": 63,
                        "constraints": {
                            "allow_short": False,
                            "block_new_shorts": True,
                            "target_short_shares": 0,
                            "max_short_exposure_pct": 0.04,
                        },
                    }
                },
                "event_catalyst_agent": {
                    "TSLA": {
                        "signal": "neutral",
                        "confidence": 60,
                        "constraints": {
                            "preferred_direction": "long",
                            "allow_short": False,
                            "block_new_shorts": True,
                            "max_short_exposure_pct": 0.0,
                        },
                    }
                },
                "short_squeeze_guardian_agent": {
                    "TSLA": {
                        "signal": "calm",
                        "confidence": 40,
                        "constraints": {"max_short_exposure_pct": 0.08},
                    }
                },
                "short_cover_classifier_agent": {"TSLA": {"signal": "neutral", "constraints": {}}},
                "breakout_cover_sentinel_agent": {"TSLA": {"signal": "monitor", "constraints": {}}},
                "macro_volatility_sentinel_agent": {"TSLA": {"signal": "watch", "constraints": {}}},
                "downside_flow_sentinel_agent": {"TSLA": {"signal": "stable", "constraints": {}}},
                "crash_short_allocator_agent": {
                    "TSLA": {
                        "signal": "monitor",
                        "confidence": 50,
                        "constraints": {},
                        "indicators": {},
                    }
                },
                "stop_loss_guardian_agent": {"TSLA": {"signal": "inactive", "constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}

    assert overrides.get("block_new_shorts") is not True
    assert overrides.get("max_additional_short_shares", 0) > 0
    assert (overrides.get("preferred_direction") or "").lower() != "long"

    notes = payload.get("reasoning", {}).get("constraints", [])
    assert any("Bearish consensus" in note for note in notes)


def test_short_cover_near_trigger_preserves_short_block(monkeypatch):
    series = _build_price_series()

    monkeypatch.setattr(
        "src.agents.risk_manager.get_prices",
        lambda *_, **__: series,
    )

    short_cover_metrics = {
        "probability": 0.24,
        "soft_threshold": 0.25,
        "improvement": 0.0018,
        "improvement_threshold": 0.0015,
        "soft_trigger": False,
        "squeeze_trigger": False,
    }

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 125_000.0,
                "positions": {"TSLA": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 125_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2025-09-01",
            "end_date": "2025-09-10",
            "analyst_signals": {
                "aswath_damodaran_agent": {
                    "TSLA": {"signal": "bearish", "confidence": 88}
                },
                "ben_graham_agent": {
                    "TSLA": {"signal": "bearish", "confidence": 84}
                },
                "michael_burry_agent": {
                    "TSLA": {"signal": "bearish", "confidence": 82}
                },
                "charlie_munger_agent": {
                    "TSLA": {"signal": "bearish", "confidence": 78}
                },
                "technical_analyst_agent": {
                    "TSLA": {"signal": "bearish", "confidence": 72}
                },
                "short_cover_classifier_agent": {
                    "TSLA": {
                        "signal": "bias_long",
                        "confidence": 62,
                        "constraints": {
                            "preferred_direction": "long",
                            "allow_short": False,
                            "block_new_shorts": True,
                            "target_short_shares": 0,
                        },
                        "metrics": short_cover_metrics,
                    }
                },
                "crash_short_allocator_agent": {
                    "TSLA": {
                        "signal": "crash_short",
                        "confidence": 70,
                        "constraints": {
                            "preferred_direction": "short",
                            "allow_short": True,
                            "target_short_shares": 40,
                            "max_additional_short_shares": 40,
                        },
                    }
                },
                "momentum_guardian_agent": {
                    "TSLA": {
                        "signal": "bullish_momentum",
                        "confidence": 90,
                        "constraints": {
                            "allow_short": False,
                            "block_new_shorts": True,
                            "preferred_direction": "long",
                        },
                    }
                },
                "trend_regime_agent": {
                    "TSLA": {
                        "signal": "bullish",
                        "confidence": 80,
                        "constraints": {
                            "allow_short": False,
                            "block_new_shorts": True,
                            "target_short_shares": 0,
                        },
                    }
                },
                "regime_meta_agent": {
                    "TSLA": {
                        "signal": "neutral",
                        "confidence": 55,
                        "constraints": {},
                        "indicators": {"probabilities": {"crash": 0.42, "rally": 0.36}},
                    }
                },
                "short_squeeze_guardian_agent": {"TSLA": {"signal": "calm", "constraints": {}}},
                "macro_volatility_sentinel_agent": {"TSLA": {"signal": "watch", "constraints": {}}},
                "downside_flow_sentinel_agent": {"TSLA": {"signal": "stable", "constraints": {}}},
                "stat_mean_reversion_agent": {"TSLA": {"signal": "bearish", "confidence": 68, "constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}
    notes = payload.get("reasoning", {}).get("constraints", [])

    assert overrides.get("block_new_shorts") is True
    assert overrides.get("max_additional_short_shares") == 0
    assert overrides.get("target_short_shares") == 0
    assert any("ShortCover" in note for note in notes)


def test_bearish_fundamental_stack_relaxes_bullish_trend_short_block(monkeypatch):
    series = _build_price_series()

    def fake_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
        return series

    monkeypatch.setattr("src.agents.risk_manager.get_prices", fake_get_prices)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 250_000.0,
                "positions": {"TSLA": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 250_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "ben_graham_agent": {"TSLA": {"signal": "bearish", "confidence": 92}},
                "charlie_munger_agent": {"TSLA": {"signal": "bearish", "confidence": 88}},
                "aswath_damodaran_agent": {"TSLA": {"signal": "bearish", "confidence": 90}},
                "michael_burry_agent": {"TSLA": {"signal": "bearish", "confidence": 94}},
                "valuation_analyst_agent": {"TSLA": {"signal": "bearish", "confidence": 86}},
                "event_catalyst_agent": {
                    "TSLA": {
                        "signal": "bearish catalyst",
                        "confidence": 72,
                        "constraints": {"preferred_direction": "short"},
                    }
                },
                "downside_flow_sentinel_agent": {
                    "TSLA": {
                        "signal": "downside_trend",
                        "confidence": 74,
                        "constraints": {},
                    }
                },
                "trend_regime_agent": {
                    "TSLA": {
                        "signal": "bullish",
                        "confidence": 70,
                        "constraints": {"preferred_direction": "long"},
                    }
                },
                "momentum_guardian_agent": {
                    "TSLA": {"signal": "bullish", "confidence": 68, "constraints": {}}
                },
                "growth_momentum_agent": {
                    "TSLA": {
                        "signal": "neutral",
                        "confidence": 55,
                        "constraints": {},
                        "indicators": {"prob_up": 0.52, "base_rate": 0.5},
                    }
                },
                "stat_mean_reversion_agent": {
                    "TSLA": {
                        "signal": "bearish",
                        "confidence": 40,
                        "constraints": {},
                        "indicators": {"prob_revert_up": 0.35},
                    }
                },
                "regime_meta_agent": {
                    "TSLA": {
                        "signal": "neutral",
                        "confidence": 58,
                        "constraints": {},
                        "indicators": {"probabilities": {"rally": 0.45, "crash": 0.28}},
                    }
                },
                "macro_volatility_sentinel_agent": {"TSLA": {"signal": "watch", "constraints": {}}},
                "short_squeeze_guardian_agent": {"TSLA": {"signal": "normal", "constraints": {}}},
                "short_cover_classifier_agent": {"TSLA": {"constraints": {}}},
                "breakout_cover_sentinel_agent": {"TSLA": {"constraints": {}}},
                "range_recovery_sentinel_agent": {"TSLA": {"constraints": {}}},
                "crash_short_allocator_agent": {
                    "TSLA": {"signal": "neutral", "confidence": 35, "constraints": {}}
                },
                "stop_loss_guardian_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}
    constraint_notes = payload.get("reasoning", {}).get("constraints", [])

    assert not overrides.get("block_new_shorts")
    max_additional_short = overrides.get("max_additional_short_shares")
    assert max_additional_short is None or max_additional_short != 0
    assert (
        "Bearish fundamentals + catalyst stack override bullish trend short block"
        in constraint_notes
    )


def test_short_cover_caution_unblock_reblocked_by_trend(monkeypatch):
    series = _build_price_series()

    monkeypatch.setattr("src.agents.risk_manager.get_prices", lambda *_, **__: series)
    monkeypatch.setattr("src.agents.risk_manager.prices_to_df", _prices_to_df_stub)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TSLA": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_cover_classifier_agent": {
                    "TSLA": {
                        "signal": "bias_long",
                        "constraints": {
                            "preferred_direction": "long",
                            "block_new_shorts": False,
                            "allow_short": True,
                        },
                        "metrics": {
                            "new_short_unblock_reason": "boost_guidance",
                            "precision_trend_strength": 0.35,
                            "precision_boost_trim_streak": 3,
                            "probability": 0.235,
                            "threshold": 0.23,
                            "soft_threshold": 0.228,
                            "improvement": 0.012,
                            "improvement_threshold": 0.013,
                            "improvement_threshold_effective": 0.011,
                        },
                    }
                },
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "momentum_guardian_agent": {"TSLA": {"constraints": {}}},
                "trend_regime_agent": {"TSLA": {"constraints": {}}},
                "crash_short_allocator_agent": {"TSLA": {"constraints": {}}},
                "macro_volatility_sentinel_agent": {"TSLA": {"constraints": {}}},
                "downside_flow_sentinel_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}
    notes = payload.get("reasoning", {}).get("constraints", [])

    assert overrides.get("block_new_shorts") is True
    assert any("Short-cover caution unblock suppressed" in note for note in notes)


def test_short_cover_caution_unblock_allows_when_trend_soft(monkeypatch):
    series = _build_price_series()

    monkeypatch.setattr("src.agents.risk_manager.get_prices", lambda *_, **__: series)
    monkeypatch.setattr("src.agents.risk_manager.prices_to_df", _prices_to_df_stub)

    state = {
        "messages": [],
        "data": {
            "portfolio": {
                "cash": 100_000.0,
                "positions": {"TSLA": {"long": 0, "short": 0}},
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
            },
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {
                "short_cover_classifier_agent": {
                    "TSLA": {
                        "signal": "bias_long",
                        "constraints": {
                            "preferred_direction": "long",
                            "block_new_shorts": False,
                            "allow_short": True,
                        },
                        "metrics": {
                            "new_short_unblock_reason": "boost_guidance",
                            "precision_trend_strength": 0.12,
                            "precision_boost_trim_streak": 0,
                            "probability": 0.225,
                            "threshold": 0.23,
                            "soft_threshold": 0.227,
                            "improvement": 0.009,
                            "improvement_threshold": 0.01,
                            "improvement_threshold_effective": 0.009,
                        },
                    }
                },
                "short_squeeze_guardian_agent": {"TSLA": {"constraints": {}}},
                "momentum_guardian_agent": {"TSLA": {"constraints": {}}},
                "trend_regime_agent": {"TSLA": {"constraints": {}}},
                "crash_short_allocator_agent": {"TSLA": {"constraints": {}}},
                "macro_volatility_sentinel_agent": {"TSLA": {"constraints": {}}},
                "downside_flow_sentinel_agent": {"TSLA": {"constraints": {}}},
            },
        },
        "metadata": {"show_reasoning": False},
    }

    result = risk_management_agent(state)
    payload = result["data"]["analyst_signals"]["risk_management_agent"]["TSLA"]
    overrides = payload.get("overrides") or {}
    notes = payload.get("reasoning", {}).get("constraints", [])

    # No caution note implies block (if any) came from other consensus logic.
    assert all("Short-cover caution unblock suppressed" not in note for note in notes)
