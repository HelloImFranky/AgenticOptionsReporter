import json
from datetime import date, datetime, timezone

from agentic_options_reporter.models.schemas import (
    AnalysisResult,
    IndicatorSnapshot,
    Recommendation,
    ScoredCandidate,
    SupportResistanceLevel,
    TrendAssessment,
    VolumeAssessment,
)
from agentic_options_reporter.thesis.orchestrator import run_thesis_pipeline

from conftest import FakeLlmClient

_ALL_RESPONSES = {
    "quantitative markets analyst": json.dumps({"narrative": "Strong setup.", "key_factors": ["trend", "liquidity"]}),
    "skeptical risk manager": json.dumps(
        {"risk_level": "medium", "concerns": ["high IV"], "position_sizing_note": "Size at 2%."}
    ),
    "options strategist": json.dumps({"strategy": "Bull Call Spread", "rationale": "Defined risk given IV concern."}),
    "portfolio manager": json.dumps(
        {"thesis": "Bullish setup with defined-risk structure recommended.", "consensus": "bullish"}
    ),
}


def _candidate() -> ScoredCandidate:
    return ScoredCandidate(
        contract_symbol="TESTC00100000", option_type="call", strike=100.0, expiration=date(2026, 1, 16),
        delta=0.55, gamma=0.02, theta=-0.05, vega=0.1, rho=0.02,
        max_loss=250.0, max_gain=None, breakeven=102.5, reward_risk_ratio=None,
        probability_of_profit=0.6, score=78.5,
        score_breakdown={"trend_alignment": 1.0, "liquidity": 0.8},
    )


def _analysis_result(candidates, recommendation) -> AnalysisResult:
    return AnalysisResult(
        symbol="TEST",
        run_id=1,
        generated_at=datetime.now(timezone.utc),
        indicators=IndicatorSnapshot(
            sma_20=100, sma_50=98, sma_200=None, ema_12=101, ema_26=99, adx_14=30,
            rsi_14=55, macd=1.2, macd_signal=1.0, macd_histogram=0.2, stoch_k=60,
            stoch_d=58, bb_upper=110, bb_middle=100, bb_lower=90, atr_14=2.5,
            obv=1_000_000, volume_sma_20=900_000,
        ),
        trend=TrendAssessment(direction="bullish", strength="strong", adx=30),
        volume=VolumeAssessment(relative_volume=1.5, flags=["high_volume"]),
        support_resistance=[SupportResistanceLevel(price=95.0, level_type="support", touches=3, last_touch_index=10)],
        candidates=candidates,
        recommendation=recommendation,
    )


def test_full_pipeline_runs_all_four_agents():
    llm = FakeLlmClient(_ALL_RESPONSES)
    candidate = _candidate()
    recommendation = Recommendation(
        action="BUY", contract_symbol=candidate.contract_symbol, confidence=0.78, rationale="top pick"
    )
    result = _analysis_result([candidate], recommendation)

    thesis = run_thesis_pipeline(result, llm)

    assert len(llm.calls) == 4
    assert thesis.run_id == 1
    assert thesis.quant_interpretation.narrative == "Strong setup."
    assert thesis.quant_interpretation.overall_score == candidate.score
    assert thesis.quant_interpretation.score_breakdown == candidate.score_breakdown
    assert thesis.risk_assessment.risk_level == "medium"
    assert thesis.strategy_suggestion.strategy == "Bull Call Spread"
    assert thesis.investment_thesis.consensus == "bullish"


def test_no_candidate_short_circuit_skips_risk_and_strategy():
    llm = FakeLlmClient(
        {"portfolio manager": json.dumps({"thesis": "No position recommended.", "consensus": "neutral"})}
    )
    recommendation = Recommendation(action="AVOID", contract_symbol=None, confidence=0.0, rationale="no candidates")
    result = _analysis_result([], recommendation)

    thesis = run_thesis_pipeline(result, llm)

    # Only investment_thesis should have been called.
    assert len(llm.calls) == 1
    assert thesis.risk_assessment is None
    assert thesis.strategy_suggestion is None
    assert thesis.quant_interpretation.narrative == "no candidates"
    assert thesis.quant_interpretation.score_breakdown == {}
    assert thesis.quant_interpretation.overall_score == 0.0
    assert thesis.investment_thesis.consensus == "neutral"


def test_recommendation_contract_not_in_candidates_is_treated_as_no_candidate():
    """Defensive: if the recommendation references a contract missing from
    candidates, the pipeline should short-circuit rather than crash."""
    llm = FakeLlmClient(
        {"portfolio manager": json.dumps({"thesis": "No position recommended.", "consensus": "neutral"})}
    )
    recommendation = Recommendation(
        action="BUY", contract_symbol="DOES_NOT_EXIST", confidence=0.5, rationale="stale reference"
    )
    result = _analysis_result([_candidate()], recommendation)

    thesis = run_thesis_pipeline(result, llm)

    assert thesis.risk_assessment is None
    assert thesis.strategy_suggestion is None
