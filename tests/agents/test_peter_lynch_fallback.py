from src.agents.peter_lynch import build_peter_lynch_fallback_signal, PeterLynchSignal


def test_peter_lynch_fallback_neutral_with_context():
    analysis_data = {
        "growth_analysis": {"notes": ["Revenue change over period: 24.0%."]},
        "valuation_analysis": {"notes": ["PEG ratio approximately 0.9."]},
        "fundamentals_analysis": {"notes": ["Debt-to-equity ratio: 0.30."]},
        "sentiment_analysis": {"notes": ["Headlines largely positive or neutral."]},
        "insider_activity": {"notes": ["Insider trades: 4 buys vs 1 sells (buy ratio 0.80)."]},
    }

    result: PeterLynchSignal = build_peter_lynch_fallback_signal("TSLA", analysis_data)

    assert result.signal == "neutral"
    assert result.confidence == 35.0
    assert "LLM output was unavailable" in result.reasoning
    assert "Growth:" in result.reasoning
    assert "Insider Activity:" in result.reasoning


def test_peter_lynch_fallback_handles_sparse_data():
    analysis_data = {
        "growth_analysis": {},
    }

    result: PeterLynchSignal = build_peter_lynch_fallback_signal("MSFT", analysis_data)

    assert result.signal == "neutral"
    assert result.confidence == 35.0
    assert "No observations captured" in result.reasoning
