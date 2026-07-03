import json
from datetime import date

import pytest

from agentic_options_reporter.models.schemas import (
    IndicatorSnapshot,
    Recommendation,
    RiskAssessment,
    ScoredCandidate,
    SupportResistanceLevel,
    TrendAssessment,
    VolumeAssessment,
)
from agentic_options_reporter.thesis import investment_thesis, options_strategy, quant_interpreter, risk_challenger
from agentic_options_reporter.thesis.parsing import ThesisGenerationError

from conftest import FakeLlmClient


def _indicators() -> IndicatorSnapshot:
    return IndicatorSnapshot(
        sma_20=100, sma_50=98, sma_200=None, ema_12=101, ema_26=99, adx_14=30,
        rsi_14=55, macd=1.2, macd_signal=1.0, macd_histogram=0.2, stoch_k=60,
        stoch_d=58, bb_upper=110, bb_middle=100, bb_lower=90, atr_14=2.5,
        obv=1_000_000, volume_sma_20=900_000,
    )


def _trend() -> TrendAssessment:
    return TrendAssessment(direction="bullish", strength="strong", adx=30)


def _volume() -> VolumeAssessment:
    return VolumeAssessment(relative_volume=1.5, flags=["high_volume"])


def _candidate() -> ScoredCandidate:
    return ScoredCandidate(
        contract_symbol="TESTC00100000", option_type="call", strike=100.0, expiration=date(2026, 1, 16),
        delta=0.55, gamma=0.02, theta=-0.05, vega=0.1, rho=0.02,
        max_loss=250.0, max_gain=None, breakeven=102.5, reward_risk_ratio=None,
        probability_of_profit=0.6, score=78.5,
        score_breakdown={"trend_alignment": 1.0, "liquidity": 0.8},
    )


def _levels() -> list[SupportResistanceLevel]:
    return [SupportResistanceLevel(price=95.0, level_type="support", touches=3, last_touch_index=10)]


def _recommendation() -> Recommendation:
    return Recommendation(action="BUY", contract_symbol="TESTC00100000", confidence=0.78, rationale="top pick")


def test_quant_interpreter_passes_through_scores_not_llm_authored():
    llm = FakeLlmClient(
        {"quantitative markets analyst": json.dumps({"narrative": "Strong.", "key_factors": ["trend"]})}
    )
    candidate = _candidate()

    result = quant_interpreter.run(llm, _indicators(), _trend(), _volume(), candidate)

    assert result.narrative == "Strong."
    assert result.key_factors == ["trend"]
    # These must come from the candidate, never from the LLM response.
    assert result.score_breakdown == candidate.score_breakdown
    assert result.overall_score == candidate.score


def test_quant_interpreter_ignores_llm_attempt_to_smuggle_scores():
    """Even if the model tries to include score fields, they must be ignored."""
    llm = FakeLlmClient(
        {
            "quantitative markets analyst": json.dumps(
                {"narrative": "Strong.", "key_factors": ["trend"], "overall_score": 999.0}
            )
        }
    )
    candidate = _candidate()

    result = quant_interpreter.run(llm, _indicators(), _trend(), _volume(), candidate)

    assert result.overall_score == candidate.score
    assert result.overall_score != 999.0


def test_quant_interpreter_raises_on_malformed_response():
    llm = FakeLlmClient({"quantitative markets analyst": "not json"})
    with pytest.raises(ThesisGenerationError):
        quant_interpreter.run(llm, _indicators(), _trend(), _volume(), _candidate())


def test_risk_challenger_parses_response():
    llm = FakeLlmClient(
        {
            "skeptical risk manager": json.dumps(
                {"risk_level": "medium", "concerns": ["high IV"], "position_sizing_note": "Size at 2%."}
            )
        }
    )
    result = risk_challenger.run(llm, _candidate(), _trend(), _levels())
    assert result.risk_level == "medium"
    assert result.concerns == ["high IV"]


def test_risk_challenger_raises_on_invalid_risk_level():
    llm = FakeLlmClient(
        {
            "skeptical risk manager": json.dumps(
                {"risk_level": "extreme", "concerns": [], "position_sizing_note": ""}
            )
        }
    )
    with pytest.raises(ThesisGenerationError):
        risk_challenger.run(llm, _candidate(), _trend(), _levels())


def test_options_strategy_parses_response():
    llm = FakeLlmClient(
        {"options strategist": json.dumps({"strategy": "Bull Call Spread", "rationale": "Defined risk."})}
    )
    risk = RiskAssessment(risk_level="medium", concerns=["high IV"], position_sizing_note="Size at 2%.")
    result = options_strategy.run(llm, _trend(), _candidate(), risk)
    assert result.strategy == "Bull Call Spread"


def test_investment_thesis_with_risk_and_strategy():
    llm = FakeLlmClient(
        {
            "portfolio manager": json.dumps(
                {"thesis": "Bullish with defined risk.", "consensus": "bullish"}
            )
        }
    )
    from agentic_options_reporter.models.schemas import QuantInterpretation, StrategySuggestion

    quant = QuantInterpretation(
        narrative="Strong.", key_factors=["trend"], score_breakdown={"x": 1.0}, overall_score=78.5
    )
    risk = RiskAssessment(risk_level="medium", concerns=["high IV"], position_sizing_note="Size at 2%.")
    strategy = StrategySuggestion(strategy="Bull Call Spread", rationale="Defined risk.")

    result = investment_thesis.run(llm, quant, risk, strategy, _recommendation(), _trend(), _volume())
    assert result.consensus == "bullish"
    assert "defined risk" in result.thesis.lower()


def test_investment_thesis_handles_missing_risk_and_strategy():
    llm = FakeLlmClient(
        {"portfolio manager": json.dumps({"thesis": "No position recommended.", "consensus": "neutral"})}
    )
    from agentic_options_reporter.models.schemas import QuantInterpretation

    quant = QuantInterpretation(narrative="no candidates", key_factors=[], score_breakdown={}, overall_score=0.0)
    recommendation = Recommendation(action="AVOID", contract_symbol=None, confidence=0.0, rationale="no candidates")

    result = investment_thesis.run(llm, quant, None, None, recommendation, _trend(), _volume())
    assert result.consensus == "neutral"
