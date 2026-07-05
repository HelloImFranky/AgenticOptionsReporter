import json
from datetime import date, datetime, timezone

from agentic_options_reporter.data.financial import FinancialProvider, ProviderHealth as FinancialProviderHealth
from agentic_options_reporter.data.macro import MacroProvider
from agentic_options_reporter.data.news import NewsProvider, ProviderHealth
from agentic_options_reporter.data.sec_provider import SECProvider
from agentic_options_reporter.models.schemas import (
    AnalysisResult,
    AnalystEstimates,
    CompanyProfile,
    FinancialRatios,
    FinancialStatementSummary,
    IndicatorSnapshot,
    MacroObservation,
    NewsArticle,
    Recommendation,
    ScoredCandidate,
    SecFiling,
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
    "catalyst analyst": json.dumps(
        {
            "catalysts": [
                {"title": "Q2 earnings beat", "category": "earnings", "horizon": "recent",
                 "direction": "bullish", "detail": "Beat consensus."}
            ],
            "summary": "Earnings just beat.",
            "net_bias": "bullish",
        }
    ),
}

# The research-only response keys, for tests that configure providers but
# skip the candidate-specific risk/strategy/quant agents.
_RESEARCH_KEYS = (
    "financial research analyst",
    "news research analyst",
    "macroeconomic analyst",
    "catalyst analyst",
)


class FakeFinancialProvider(FinancialProvider):
    _DATASETS = frozenset({"profile", "statements", "ratios", "analyst_estimates"})

    @property
    def supported_datasets(self) -> frozenset[str]:
        return self._DATASETS

    async def get_company_profile(self, ticker: str) -> CompanyProfile:
        return CompanyProfile(
            ticker=ticker, name="Test Corp", sector="Technology", industry="Software",
            market_cap=1_000_000_000, description="Makes software.",
        )

    async def get_financial_statements(self, ticker: str) -> FinancialStatementSummary:
        return FinancialStatementSummary(
            ticker=ticker, period="2025", revenue=500_000_000, net_income=80_000_000,
            operating_cash_flow=100_000_000, free_cash_flow=70_000_000,
        )

    async def get_ratios(self, ticker: str) -> FinancialRatios:
        return FinancialRatios(
            ticker=ticker, pe_ratio=25.0, pb_ratio=8.0, debt_to_equity=0.5, current_ratio=1.8,
            return_on_equity=0.3, gross_margin=0.6, net_margin=0.16,
        )

    async def get_analyst_estimates(self, ticker: str) -> AnalystEstimates:
        return AnalystEstimates(
            ticker=ticker, consensus_rating="Buy", price_target_mean=120.0,
            price_target_high=140.0, price_target_low=100.0, num_analysts=15,
        )

    async def health(self) -> FinancialProviderHealth:
        return FinancialProviderHealth(
            provider="fake", healthy=True, checked_at=datetime.now(timezone.utc)
        )


class FakeNewsProvider(NewsProvider):
    @property
    def capabilities(self):
        return frozenset({"company_news", "top_headlines"})

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
    _VALUES = {
        "policy_rate": (5.25, "percent"),
        "cpi": (310.0, "index"),
        "gdp": (23000.0, "usd"),
    }

    @property
    def supported_metrics(self) -> frozenset[str]:
        return frozenset(self._VALUES)

    async def fetch(self, metric_id: str) -> MacroObservation:
        value, unit = self._VALUES[metric_id]
        return MacroObservation(
            metric_id=metric_id, label=metric_id, value=value, unit=unit,
            as_of=date(2026, 6, 1), source="fake",
        )

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider="fake", healthy=True, checked_at=datetime.now(timezone.utc)
        )


class FakeSecProvider(SECProvider):
    async def get_recent_filings(self, ticker, limit=10):
        return [
            SecFiling(
                ticker=ticker.upper(), form_type="8-K", filed_at=date(2026, 6, 2),
                url="https://sec.gov/a", accession_number="0000-26-01",
            )
        ]

    async def get_10k(self, ticker):
        return None

    async def get_10q(self, ticker):
        return None

    async def get_8k(self, ticker):
        return None

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider="fake-sec", healthy=True, checked_at=datetime.now(timezone.utc)
        )


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
    assert thesis.catalyst_research is None


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
        sec_provider=FakeSecProvider(),
    )

    # quant, risk, strategy, financial, news, macro, catalyst, thesis
    assert len(llm.calls) == 8
    assert thesis.financial_research.company_health == "strong"
    assert thesis.financial_research.analyst_consensus == "Buy"
    assert thesis.news_research.sentiment == "bullish"
    assert thesis.macro_research.regime == "risk_on"
    assert thesis.catalyst_research is not None
    assert thesis.catalyst_research.net_bias == "bullish"
    assert thesis.catalyst_research.catalysts[0].category == "earnings"


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


class PartialFinancialProvider(FakeFinancialProvider):
    """A Finnhub-style provider: no statements. get_financial_statements
    must never be called (the router filters it out)."""

    _DATASETS = frozenset({"profile", "ratios", "analyst_estimates"})

    async def get_financial_statements(self, ticker):
        raise AssertionError("router must not call an unadvertised dataset")


def test_financial_research_runs_with_partial_dataset_coverage():
    """Finnhub-only financial: profile/ratios/estimates present, statements
    absent — the agent still produces a finding over what's available."""
    llm = FakeLlmClient(_ALL_RESPONSES_WITH_RESEARCH)
    candidate = _candidate()
    recommendation = Recommendation(
        action="BUY", contract_symbol=candidate.contract_symbol, confidence=0.78, rationale="top pick"
    )
    result = _analysis_result([candidate], recommendation)

    thesis = run_thesis_pipeline(result, llm, financial_provider=PartialFinancialProvider())

    assert thesis.financial_research is not None
    assert thesis.financial_research.company_health == "strong"
    assert thesis.financial_research.analyst_consensus == "Buy"
    assert thesis.pipeline_warnings == []
    # The prompt must reflect the missing statements section, not fabricate it.
    _, user_prompt = next(
        (c for c in reversed(llm.calls) if "financial research analyst" in c[0]), (None, "")
    )
    assert "Financial statements: not available." in user_prompt


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

    @property
    def capabilities(self):
        return frozenset({"company_news", "top_headlines"})

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
    # Catalyst still runs over the streams that succeeded (macro), even
    # though its news stream hit the same rate limit.
    assert thesis.catalyst_research is not None
    assert thesis.investment_thesis.consensus == "bullish"

    # Both agents that depend on the news provider report the failure: the
    # news_research step and the catalyst_research news stream.
    news_warnings = [w for w in thesis.pipeline_warnings if w.startswith("news_research:")]
    catalyst_warnings = [w for w in thesis.pipeline_warnings if w.startswith("catalyst_research:")]
    assert len(news_warnings) == 1
    assert "rate limited" in news_warnings[0].lower()
    assert len(catalyst_warnings) == 1
    assert "news" in catalyst_warnings[0].lower()


def test_unusable_research_agent_response_records_warning_instead_of_502():
    """An LLM response an agent can't parse/validate (even after lenient
    coercion) must not 502 the whole thesis: the affected research finding
    stays null, a warning is recorded, and the rest completes."""
    responses = {**_ALL_RESPONSES_WITH_RESEARCH, "catalyst analyst": "sorry, I cannot help with that"}
    llm = FakeLlmClient(responses)
    candidate = _candidate()
    recommendation = Recommendation(
        action="BUY", contract_symbol=candidate.contract_symbol, confidence=0.78, rationale="top pick"
    )
    result = _analysis_result([candidate], recommendation)

    thesis = run_thesis_pipeline(
        result,
        llm,
        news_provider=FakeNewsProvider(),
        macro_provider=FakeMacroProvider(),
        sec_provider=FakeSecProvider(),
    )

    assert thesis.catalyst_research is None            # the unparseable agent is dropped
    assert thesis.news_research is not None            # the others still ran
    assert thesis.investment_thesis is not None        # synthesis still completed
    catalyst_warnings = [w for w in thesis.pipeline_warnings if w.startswith("catalyst_research:")]
    assert len(catalyst_warnings) == 1
    assert "unusable model response" in catalyst_warnings[0].lower()


def test_research_agents_run_even_without_candidate():
    """Research findings are ticker/market-wide, not contract-specific, so
    they should still run when there's no candidate to size (unlike
    risk/strategy, which are legitimately skipped in that case)."""
    llm = FakeLlmClient(
        {
            "portfolio manager": json.dumps({"thesis": "No position recommended.", "consensus": "neutral"}),
            **{k: v for k, v in _ALL_RESPONSES_WITH_RESEARCH.items() if k in _RESEARCH_KEYS},
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
        sec_provider=FakeSecProvider(),
    )

    assert thesis.risk_assessment is None
    assert thesis.strategy_suggestion is None
    assert thesis.financial_research is not None
    assert thesis.news_research is not None
    assert thesis.macro_research is not None
    assert thesis.catalyst_research is not None


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
