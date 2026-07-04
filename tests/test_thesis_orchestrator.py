import json
from datetime import date, datetime, timezone

from agentic_options_reporter.data.financial_provider import FinancialProvider
from agentic_options_reporter.data.macro_provider import MacroProvider
from agentic_options_reporter.data.news import NewsProvider, ProviderHealth
from agentic_options_reporter.models.schemas import (
    AnalysisResult,
    AnalystEstimates,
    CompanyProfile,
    CpiSnapshot,
    FinancialRatios,
    FinancialStatementSummary,
    GdpSnapshot,
    IndicatorSnapshot,
    InterestRates,
    NewsArticle,
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

_ALL_RESPONSES_WITH_RESEARCH = {
    **_ALL_RESPONSES,
    "financial research analyst": json.dumps(
        {
            "company_health": "strong",
            "growth": "accelerating",
            "profitability": "high",
            "cash_flow": "positive",
            "narrative": "Fundamentals solid.",
        }
    ),
    "news research analyst": json.dumps(
        {"sentiment": "bullish", "summary": "Positive coverage.", "catalysts": ["earnings beat"], "risks": []}
    ),
    "macroeconomic analyst": json.dumps(
        {"regime": "risk_on", "outlook": "Favorable.", "summary": "Rates steady."}
    ),
}


class FakeFinancialProvider(FinancialProvider):
    def get_company_profile(self, ticker: str) -> CompanyProfile:
        return CompanyProfile(
            ticker=ticker, name="Test Corp", sector="Technology", industry="Software",
            market_cap=1_000_000_000, description="Makes software.",
        )

    def get_financial_statements(self, ticker: str) -> FinancialStatementSummary:
        return FinancialStatementSummary(
            ticker=ticker, period="2025", revenue=500_000_000, net_income=80_000_000,
            operating_cash_flow=100_000_000, free_cash_flow=70_000_000,
        )

    def get_ratios(self, ticker: str) -> FinancialRatios:
        return FinancialRatios(
            ticker=ticker, pe_ratio=25.0, pb_ratio=8.0, debt_to_equity=0.5, current_ratio=1.8,
            return_on_equity=0.3, gross_margin=0.6, net_margin=0.16,
        )

    def get_analyst_estimates(self, ticker: str) -> AnalystEstimates:
        return AnalystEstimates(
            ticker=ticker, consensus_rating="Buy", price_target_mean=120.0,
            price_target_high=140.0, price_target_low=100.0, num_analysts=15,
        )


class FakeNewsProvider(NewsProvider):
    async def search(self, query, start_date=None, end_date=None, language="en", limit=20):
        return [
            NewsArticle(
                headline="Company beats earnings", source="Reuters", url="https://example.com/a",
                published_at=datetime(2026, 6, 1, tzinfo=timezone.utc), summary="Solid quarter.",
            )
        ]

    async def top_headlines(self, category=None, limit=20):
        return []

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider="fake", healthy=True, checked_at=datetime.now(timezone.utc)
        )


class FakeMacroProvider(MacroProvider):
    def get_interest_rates(self) -> InterestRates:
        return InterestRates(fed_funds_rate=5.25, ten_year_yield=4.3, two_year_yield=4.1, as_of=date(2026, 6, 1))

    def get_cpi(self) -> CpiSnapshot:
        return CpiSnapshot(value=310.0, yoy_change_pct=3.3, as_of=date(2026, 6, 1))

    def get_gdp(self) -> GdpSnapshot:
        return GdpSnapshot(value=23000.0, yoy_growth_pct=2.1, as_of=date(2026, 4, 1))

    def get_macro_calendar(self) -> list:
        return []


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
    # No providers were passed in, so research findings must be absent.
    assert thesis.financial_research is None
    assert thesis.news_research is None
    assert thesis.macro_research is None


def test_pipeline_runs_research_agents_when_providers_configured():
    llm = FakeLlmClient(_ALL_RESPONSES_WITH_RESEARCH)
    candidate = _candidate()
    recommendation = Recommendation(
        action="BUY", contract_symbol=candidate.contract_symbol, confidence=0.78, rationale="top pick"
    )
    result = _analysis_result([candidate], recommendation)

    thesis = run_thesis_pipeline(
        result,
        llm,
        financial_provider=FakeFinancialProvider(),
        news_provider=FakeNewsProvider(),
        macro_provider=FakeMacroProvider(),
    )

    assert len(llm.calls) == 7
    assert thesis.financial_research.company_health == "strong"
    assert thesis.financial_research.analyst_consensus == "Buy"
    assert thesis.news_research.sentiment == "bullish"
    assert thesis.macro_research.regime == "risk_on"


def test_pipeline_runs_only_configured_research_agents():
    llm = FakeLlmClient(_ALL_RESPONSES_WITH_RESEARCH)
    candidate = _candidate()
    recommendation = Recommendation(
        action="BUY", contract_symbol=candidate.contract_symbol, confidence=0.78, rationale="top pick"
    )
    result = _analysis_result([candidate], recommendation)

    thesis = run_thesis_pipeline(result, llm, news_provider=FakeNewsProvider())

    assert thesis.financial_research is None
    assert thesis.news_research is not None
    assert thesis.macro_research is None


def test_pipeline_records_no_warnings_when_research_succeeds():
    llm = FakeLlmClient(_ALL_RESPONSES_WITH_RESEARCH)
    candidate = _candidate()
    recommendation = Recommendation(
        action="BUY", contract_symbol=candidate.contract_symbol, confidence=0.78, rationale="top pick"
    )
    result = _analysis_result([candidate], recommendation)

    thesis = run_thesis_pipeline(result, llm, news_provider=FakeNewsProvider())

    assert thesis.pipeline_warnings == []


class RateLimitedNewsProvider(NewsProvider):
    """Simulates a provider that IS configured but 429s at call time."""

    async def search(self, query, start_date=None, end_date=None, language="en", limit=20):
        from agentic_options_reporter.data.news import NewsProviderRateLimited

        raise NewsProviderRateLimited("Finnhub rate limited: 429 Too Many Requests")

    async def top_headlines(self, category=None, limit=20):
        raise NotImplementedError

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider="rate-limited", healthy=False, checked_at=datetime.now(timezone.utc)
        )


def test_provider_failure_mid_run_records_warning_instead_of_crashing():
    """A configured provider failing during the run (e.g. rate limited)
    must not throw away the rest of the pipeline: the finding is null,
    the failure lands in pipeline_warnings, and the thesis still
    synthesizes over what's present."""
    llm = FakeLlmClient(_ALL_RESPONSES_WITH_RESEARCH)
    candidate = _candidate()
    recommendation = Recommendation(
        action="BUY", contract_symbol=candidate.contract_symbol, confidence=0.78, rationale="top pick"
    )
    result = _analysis_result([candidate], recommendation)

    thesis = run_thesis_pipeline(
        result,
        llm,
        financial_provider=FakeFinancialProvider(),
        news_provider=RateLimitedNewsProvider(),
        macro_provider=FakeMacroProvider(),
    )

    # The failed agent's finding is null; the others still completed.
    assert thesis.news_research is None
    assert thesis.financial_research is not None
    assert thesis.macro_research is not None
    assert thesis.investment_thesis.consensus == "bullish"

    assert len(thesis.pipeline_warnings) == 1
    assert thesis.pipeline_warnings[0].startswith("news_research:")
    assert "rate limited" in thesis.pipeline_warnings[0].lower()


def test_research_agents_run_even_without_candidate():
    """Research findings are ticker/market-wide, not contract-specific, so
    they should still run when there's no candidate to size (unlike
    risk/strategy, which are legitimately skipped in that case)."""
    llm = FakeLlmClient(
        {
            "portfolio manager": json.dumps({"thesis": "No position recommended.", "consensus": "neutral"}),
            **{
                k: v
                for k, v in _ALL_RESPONSES_WITH_RESEARCH.items()
                if k in ("financial research analyst", "news research analyst", "macroeconomic analyst")
            },
        }
    )
    recommendation = Recommendation(action="AVOID", contract_symbol=None, confidence=0.0, rationale="no candidates")
    result = _analysis_result([], recommendation)

    thesis = run_thesis_pipeline(
        result,
        llm,
        financial_provider=FakeFinancialProvider(),
        news_provider=FakeNewsProvider(),
        macro_provider=FakeMacroProvider(),
    )

    assert thesis.risk_assessment is None
    assert thesis.strategy_suggestion is None
    assert thesis.financial_research is not None
    assert thesis.news_research is not None
    assert thesis.macro_research is not None


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
