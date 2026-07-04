import json
from datetime import date, datetime, timezone

import pytest

from agentic_options_reporter.models.schemas import (
    AnalystEstimates,
    CompanyProfile,
    FinancialRatios,
    FinancialStatementSummary,
    IndicatorSnapshot,
    MacroObservation,
    NewsArticle,
    Recommendation,
    RiskAssessment,
    ScoredCandidate,
    SupportResistanceLevel,
    TrendAssessment,
    VolumeAssessment,
)
from agentic_options_reporter.thesis import (
    financial_research,
    investment_thesis,
    macro_research,
    news_research,
    options_strategy,
    quant_interpreter,
    risk_challenger,
)
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

    result = investment_thesis.run(
        llm, quant, None, None, None, risk, strategy, _recommendation(), _trend(), _volume()
    )
    assert result.consensus == "bullish"
    assert "defined risk" in result.thesis.lower()


def test_investment_thesis_synthesizes_all_research_findings():
    from agentic_options_reporter.models.schemas import (
        FinancialResearchFinding,
        MacroResearchFinding,
        NewsResearchFinding,
        QuantInterpretation,
        StrategySuggestion,
    )

    llm = FakeLlmClient(
        {"portfolio manager": json.dumps({"thesis": "Bullish across the board.", "consensus": "bullish"})}
    )
    quant = QuantInterpretation(
        narrative="Strong.", key_factors=["trend"], score_breakdown={"x": 1.0}, overall_score=78.5
    )
    financial = FinancialResearchFinding(
        company_health="strong", growth="accelerating", profitability="high",
        cash_flow="positive", analyst_consensus="Buy", narrative="Fundamentals solid.",
    )
    news = NewsResearchFinding(
        sentiment="bullish", summary="Positive coverage.", catalysts=["earnings beat"], risks=[]
    )
    macro = MacroResearchFinding(regime="risk_on", outlook="Favorable.", summary="Rates steady.")
    risk = RiskAssessment(risk_level="medium", concerns=["high IV"], position_sizing_note="Size at 2%.")
    strategy = StrategySuggestion(strategy="Bull Call Spread", rationale="Defined risk.")

    result = investment_thesis.run(
        llm, quant, financial, news, macro, risk, strategy, _recommendation(), _trend(), _volume()
    )
    assert result.consensus == "bullish"

    # The prompt sent to the LLM must actually carry every finding through.
    _, user_prompt = llm.calls[-1]
    assert "health=strong" in user_prompt
    assert "sentiment=bullish" in user_prompt
    assert "regime=risk_on" in user_prompt


def test_investment_thesis_handles_missing_risk_and_strategy():
    llm = FakeLlmClient(
        {"portfolio manager": json.dumps({"thesis": "No position recommended.", "consensus": "neutral"})}
    )
    from agentic_options_reporter.models.schemas import QuantInterpretation

    quant = QuantInterpretation(narrative="no candidates", key_factors=[], score_breakdown={}, overall_score=0.0)
    recommendation = Recommendation(action="AVOID", contract_symbol=None, confidence=0.0, rationale="no candidates")

    result = investment_thesis.run(
        llm, quant, None, None, None, None, None, recommendation, _trend(), _volume()
    )
    assert result.consensus == "neutral"


def _profile() -> CompanyProfile:
    return CompanyProfile(
        ticker="TEST", name="Test Corp", sector="Technology", industry="Software",
        market_cap=1_000_000_000, description="Makes software.",
    )


def _statements() -> FinancialStatementSummary:
    return FinancialStatementSummary(
        ticker="TEST", period="2025", revenue=500_000_000, net_income=80_000_000,
        operating_cash_flow=100_000_000, free_cash_flow=70_000_000,
    )


def _ratios() -> FinancialRatios:
    return FinancialRatios(
        ticker="TEST", pe_ratio=25.0, pb_ratio=8.0, debt_to_equity=0.5, current_ratio=1.8,
        return_on_equity=0.3, gross_margin=0.6, net_margin=0.16,
    )


def _estimates() -> AnalystEstimates:
    return AnalystEstimates(
        ticker="TEST", consensus_rating="Buy", price_target_mean=120.0,
        price_target_high=140.0, price_target_low=100.0, num_analysts=15,
    )


def test_financial_research_passes_through_analyst_consensus_not_llm_authored():
    llm = FakeLlmClient(
        {
            "financial research analyst": json.dumps(
                {
                    "company_health": "strong",
                    "growth": "accelerating",
                    "profitability": "high",
                    "cash_flow": "positive",
                    "narrative": "Fundamentals look solid.",
                }
            )
        }
    )
    result = financial_research.run(llm, _profile(), _statements(), _ratios(), _estimates())
    assert result.analyst_consensus == "Buy"
    assert result.company_health == "strong"
    assert result.narrative == "Fundamentals look solid."


def test_financial_research_ignores_llm_attempt_to_smuggle_consensus():
    llm = FakeLlmClient(
        {
            "financial research analyst": json.dumps(
                {
                    "company_health": "strong",
                    "growth": "accelerating",
                    "profitability": "high",
                    "cash_flow": "positive",
                    "narrative": "Fundamentals look solid.",
                    "analyst_consensus": "Strong Sell",
                }
            )
        }
    )
    result = financial_research.run(llm, _profile(), _statements(), _ratios(), _estimates())
    assert result.analyst_consensus == "Buy"
    assert result.analyst_consensus != "Strong Sell"


def test_financial_research_raises_on_invalid_health():
    llm = FakeLlmClient(
        {
            "financial research analyst": json.dumps(
                {
                    "company_health": "excellent",
                    "growth": "accelerating",
                    "profitability": "high",
                    "cash_flow": "positive",
                    "narrative": "x",
                }
            )
        }
    )
    with pytest.raises(ThesisGenerationError):
        financial_research.run(llm, _profile(), _statements(), _ratios(), _estimates())


def _articles() -> list[NewsArticle]:
    return [
        NewsArticle(
            headline="Company beats earnings", source="Reuters", url="https://example.com/a",
            published_at=datetime(2026, 6, 1, tzinfo=timezone.utc), summary="Solid quarter.",
        )
    ]


def test_news_research_parses_response():
    llm = FakeLlmClient(
        {
            "news research analyst": json.dumps(
                {
                    "sentiment": "bullish",
                    "summary": "Positive earnings momentum.",
                    "catalysts": ["earnings beat"],
                    "risks": ["supply chain"],
                }
            )
        }
    )
    result = news_research.run(llm, _articles())
    assert result.sentiment == "bullish"
    assert result.catalysts == ["earnings beat"]
    assert result.risks == ["supply chain"]


def test_news_research_handles_no_articles():
    llm = FakeLlmClient(
        {
            "news research analyst": json.dumps(
                {"sentiment": "neutral", "summary": "No notable news.", "catalysts": [], "risks": []}
            )
        }
    )
    result = news_research.run(llm, [])
    assert result.sentiment == "neutral"
    assert result.catalysts == []


def test_news_research_raises_on_invalid_sentiment():
    llm = FakeLlmClient(
        {
            "news research analyst": json.dumps(
                {"sentiment": "euphoric", "summary": "x", "catalysts": [], "risks": []}
            )
        }
    )
    with pytest.raises(ThesisGenerationError):
        news_research.run(llm, _articles())


def _observations() -> list[MacroObservation]:
    return [
        MacroObservation(
            metric_id="policy_rate", label="Federal funds rate", value=5.25,
            unit="percent", as_of=date(2026, 6, 1), source="FRED",
        ),
        MacroObservation(
            metric_id="cpi", label="Consumer Price Index", value=310.0, unit="index",
            as_of=date(2026, 6, 1), source="BLS", yoy_change_pct=3.3,
        ),
        MacroObservation(
            metric_id="gdp", label="Gross domestic product (nominal)", value=23000.0,
            unit="usd", as_of=date(2026, 4, 1), source="BEA", yoy_change_pct=2.1,
        ),
    ]


def test_macro_research_parses_response():
    llm = FakeLlmClient(
        {
            "macroeconomic analyst": json.dumps(
                {
                    "regime": "risk_on",
                    "outlook": "Conditions favor risk assets near-term.",
                    "summary": "Rates steady, inflation cooling, growth resilient.",
                }
            )
        }
    )
    result = macro_research.run(llm, _observations())
    assert result.regime == "risk_on"
    assert "risk assets" in result.outlook.lower()


def test_macro_research_raises_on_invalid_regime():
    llm = FakeLlmClient(
        {
            "macroeconomic analyst": json.dumps(
                {"regime": "goldilocks", "outlook": "x", "summary": "x"}
            )
        }
    )
    with pytest.raises(ThesisGenerationError):
        macro_research.run(llm, _observations())
