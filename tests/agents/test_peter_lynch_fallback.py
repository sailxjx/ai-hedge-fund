from src.agents.peter_lynch import (
    PeterLynchSignal,
    build_peter_lynch_fallback_signal,
)


def test_peter_lynch_fallback_bullish_signal_confidence():
    analysis_data = {
        "signal": "bullish",
        "score": 8.4,
        "max_score": 10.0,
        "growth_analysis": {"score": 9.0, "details": "Strong revenue and EPS expansion"},
        "valuation_analysis": {"score": 7.5, "details": "PEG ratio at 0.9"},
        "fundamentals_analysis": {"score": 6.0, "details": "Moderate debt with solid margins"},
        "sentiment_analysis": {"score": 7.0, "details": "Minimal negative headlines"},
        "insider_activity": {"score": 8.0, "details": "Heavy insider buying"},
    }

    result: PeterLynchSignal = build_peter_lynch_fallback_signal("TSLA", analysis_data)

    assert result.signal == "bullish"
    assert result.confidence >= 70
    assert "Deterministic fallback" in result.reasoning
    assert "Composite score" in result.reasoning


def test_peter_lynch_fallback_handles_sparse_data():
    analysis_data = {
        "signal": "neutral",
        "score": 5.0,
        "max_score": 10.0,
        "growth_analysis": {},
    }

    result: PeterLynchSignal = build_peter_lynch_fallback_signal("MSFT", analysis_data)

    assert result.signal == "neutral"
    assert 35 <= result.confidence <= 65
    assert "No component data" in result.reasoning or "Drivers:" in result.reasoning
