from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agentic_options_reporter import main as main_module
from agentic_options_reporter.analysis.composite_score import compute_composite_score
from agentic_options_reporter.models.schemas import (
    AgentThesisResult,
    CatalystFinding,
    CatalystItem,
    DomainScore,
    FinancialResearchFinding,
    IndicatorSnapshot,
    InvestmentThesis,
    MacroResearchFinding,
    NewsResearchFinding,
    QuantInterpretation,
    Recommendation,
    RiskAssessment,
    ScoredCandidate,
    StrategySuggestion,
    SupportResistanceLevel,
    TrendAssessment,
    VolumeAssessment,
)
from agentic_options_reporter.persistence import make_session_factory, persist_analysis_run


def _domain_score(domain: str, score: float = 80.0) -> DomainScore:
    return DomainScore(
        domain=domain, score=score, confidence=90.0, evidence=[], factors=[],
        source="quant", generated_at=datetime(2026, 1, 1),
    )


def _quant_interpretation(narrative: str = "Strong trend.", score: float = 78.5) -> QuantInterpretation:
    domain_scores = {"technical": _domain_score("technical", score)}
    return QuantInterpretation(
        narrative=narrative,
        key_factors=["trend"],
        quant_trade_quality=compute_composite_score(domain_scores, source="quant", contract_symbol="TESTC00100000"),
        technical_domain_score=DomainScore(
            domain="technical", score=score, confidence=85.0, evidence=[], factors=[],
            source="agent", generated_at=datetime(2026, 1, 1),
        ),
    )


@pytest.fixture
def client(monkeypatch):
    session_factory = make_session_factory("sqlite:///:memory:")
    monkeypatch.setattr(main_module, "_session_factory", session_factory)
    return TestClient(main_module.app), session_factory


def _indicator_snapshot() -> IndicatorSnapshot:
    return IndicatorSnapshot(
        sma_20=100, sma_50=98, sma_200=None, ema_12=101, ema_26=99, adx_14=30,
        rsi_14=55, macd=1.2, macd_signal=1.0, macd_histogram=0.2, stoch_k=60,
        stoch_d=58, bb_upper=110, bb_middle=100, bb_lower=90, atr_14=2.5,
        obv=1_000_000, volume_sma_20=900_000,
    )


def test_health(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_run_not_found(client):
    test_client, _ = client
    response = test_client.get("/runs/999")
    assert response.status_code == 404


def _scored_candidate() -> ScoredCandidate:
    return ScoredCandidate(
        contract_symbol="TESTC00100000",
        option_type="call",
        strike=100.0,
        expiration=date(2026, 1, 16),
        delta=0.55,
        gamma=0.02,
        theta=-0.05,
        vega=0.1,
        rho=0.02,
        max_loss=250.0,
        max_gain=None,
        breakeven=102.5,
        reward_risk_ratio=None,
        probability_of_profit=0.6,
        score=78.5,
        domain_scores={"technical": _domain_score("technical", 91.0)},
    )


def _persist_full_run(session_factory, recommendation=None) -> int:
    recommendation = recommendation or Recommendation(
        action="BUY", contract_symbol="TESTC00100000", confidence=0.7, rationale="test"
    )
    trend = TrendAssessment(direction="bullish", strength="moderate", adx=25)
    volume = VolumeAssessment(relative_volume=1.2, flags=["normal_volume"])
    levels = [SupportResistanceLevel(price=95.0, level_type="support", touches=3, last_touch_index=10)]
    with session_factory() as session:
        return persist_analysis_run(
            session,
            "TEST",
            260,
            None,
            _indicator_snapshot(),
            trend,
            volume,
            levels,
            [_scored_candidate()],
            recommendation,
            None,
            "swing",
        )


def test_get_run_and_list_runs(client):
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)

    response = test_client.get(f"/runs/{run_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "TEST"
    assert body["recommendation"]["action"] == "BUY"

    response = test_client.get("/runs", params={"symbol": "TEST"})
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id


def test_get_run_returns_persisted_fundamentals(client):
    """A run analysed with fundamentals returns the same snapshot on replay
    (they're persisted as JSON, not response-only)."""
    from agentic_options_reporter.models.schemas import (
        CompanyMetrics,
        CompanyProfile,
        FundamentalsSnapshot,
    )

    test_client, session_factory = client
    recommendation = Recommendation(
        action="BUY", contract_symbol="TESTC00100000", confidence=0.7, rationale="test"
    )
    fundamentals = FundamentalsSnapshot(
        ticker="TEST",
        profile=CompanyProfile(ticker="TEST", name="Test Corp"),
        metrics=CompanyMetrics(ticker="TEST", pe_ratio=25.0, beta=1.1),
    )
    with session_factory() as session:
        run_id = persist_analysis_run(
            session, "TEST", 260, None, _indicator_snapshot(),
            TrendAssessment(direction="bullish", strength="moderate", adx=25),
            VolumeAssessment(relative_volume=1.2, flags=["normal_volume"]),
            [SupportResistanceLevel(price=95.0, level_type="support", touches=3, last_touch_index=10)],
            [_scored_candidate()], recommendation, None, "swing",
            fundamentals=fundamentals, data_warnings=["statements: rate limited"],
        )

    body = test_client.get(f"/runs/{run_id}").json()
    assert body["fundamentals"] is not None
    assert body["fundamentals"]["profile"]["name"] == "Test Corp"
    assert body["fundamentals"]["metrics"]["pe_ratio"] == 25.0
    assert body["data_warnings"] == ["statements: rate limited"]


def test_get_run_without_fundamentals_returns_null(client):
    """Runs persisted without fundamentals (e.g. legacy rows) replay cleanly
    with fundamentals=null and no warnings."""
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)  # helper omits fundamentals

    body = test_client.get(f"/runs/{run_id}").json()
    assert body["fundamentals"] is None
    assert body["data_warnings"] == []


def test_get_run_returns_real_trend_volume_and_levels(client):
    """Regression test: a replayed run must not fall back to placeholders."""
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)

    body = test_client.get(f"/runs/{run_id}").json()
    assert body["trend"]["direction"] == "bullish"
    assert body["trend"]["strength"] == "moderate"
    assert body["trend"]["adx"] == 25
    assert body["volume"]["relative_volume"] == 1.2
    assert body["volume"]["flags"] == ["normal_volume"]
    assert len(body["support_resistance"]) == 1
    assert body["support_resistance"][0]["price"] == 95.0


def test_analyze_uses_workflow(client, monkeypatch):
    test_client, session_factory = client

    canned = None

    def fake_run_analysis(**kwargs):
        nonlocal canned
        from agentic_options_reporter.models.schemas import AnalysisResult

        canned = AnalysisResult(
            symbol=kwargs["symbol"],
            run_id=1,
            generated_at=datetime.now(timezone.utc),
            indicators=_indicator_snapshot(),
            trend=TrendAssessment(direction="bullish", strength="moderate", adx=25),
            volume=VolumeAssessment(relative_volume=1.2, flags=["normal_volume"]),
            support_resistance=[],
            candidates=[],
            recommendation=Recommendation(
                action="HOLD", contract_symbol=None, confidence=0.4, rationale="test"
            ),
        )
        return canned

    monkeypatch.setattr(main_module, "run_analysis", fake_run_analysis)

    response = test_client.get("/analyze/TEST")
    assert response.status_code == 200
    assert response.json()["symbol"] == "TEST"


def _fake_thesis_result(run_id: int) -> AgentThesisResult:
    return AgentThesisResult(
        run_id=run_id,
        generated_at=datetime.now(timezone.utc),
        quant_interpretation=_quant_interpretation(),
        risk_assessment=RiskAssessment(
            risk_level="medium", concerns=["high IV"], position_sizing_note="Size at 2%.",
            domain_score=_domain_score("risk"),
        ),
        strategy_suggestion=StrategySuggestion(
            strategy="Bull Call Spread", rationale="Defined risk.", domain_score=_domain_score("liquidity")
        ),
        investment_thesis=InvestmentThesis(thesis="Bullish setup.", consensus="bullish"),
    )


def test_generate_thesis_not_found(client):
    test_client, _ = client
    response = test_client.post("/runs/999/thesis")
    assert response.status_code == 404


def test_generate_thesis_success(client):
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)
    fake_result = _fake_thesis_result(run_id)

    with patch("agentic_options_reporter.main.build_llm_client", return_value=MagicMock()), patch(
        "agentic_options_reporter.main.run_thesis_pipeline", return_value=fake_result
    ):
        response = test_client.post(f"/runs/{run_id}/thesis")

    assert response.status_code == 200
    body = response.json()
    assert body["investment_thesis"]["consensus"] == "bullish"
    assert body["risk_assessment"]["risk_level"] == "medium"
    assert body["strategy_suggestion"]["strategy"] == "Bull Call Spread"


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse a text/event-stream body into (event, data) tuples."""
    import json

    frames = []
    event = None
    data_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
        elif not line and event is not None and data_lines:
            frames.append((event, json.loads("\n".join(data_lines))))
            event, data_lines = None, []
    return frames


def test_generate_thesis_stream_emits_agent_events_then_result(client):
    from agentic_options_reporter.models.schemas import AgentEvent, AgentExchange

    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)
    fake_result = _fake_thesis_result(run_id)

    def fake_pipeline(analysis_result, llm_client, on_event=None, **_kwargs):
        on_event(AgentEvent(
            agent="quant_interpreter", phase="started",
            at=datetime.now(timezone.utc).replace(tzinfo=None),
        ))
        on_event(AgentEvent(
            agent="quant_interpreter", phase="completed",
            at=datetime.now(timezone.utc).replace(tzinfo=None),
            output={"narrative": "Strong."},
            exchange=AgentExchange(
                system_prompt="sys", user_prompt="usr", raw_response="{...}"
            ),
        ))
        return fake_result

    with patch("agentic_options_reporter.main.build_llm_client", return_value=MagicMock()), patch(
        "agentic_options_reporter.main.run_thesis_pipeline", side_effect=fake_pipeline
    ):
        response = test_client.post(f"/runs/{run_id}/thesis/stream")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _parse_sse(response.text)
    kinds = [event for event, _ in frames]
    assert kinds == ["agent", "agent", "result"]
    # The under-the-hood exchange rides along on the completed agent frame.
    completed = frames[1][1]
    assert completed["phase"] == "completed"
    assert completed["exchange"]["system_prompt"] == "sys"
    # The terminal result carries the full thesis and is persisted.
    assert frames[-1][1]["investment_thesis"]["consensus"] == "bullish"
    persisted = test_client.get(f"/runs/{run_id}/thesis")
    assert persisted.status_code == 200


def test_generate_thesis_stream_emits_error_frame_on_fatal_failure(client):
    from agentic_options_reporter.thesis.llm_client import LlmError

    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)

    def failing_pipeline(analysis_result, llm_client, on_event=None, **_kwargs):
        raise LlmError("provider exploded")

    with patch("agentic_options_reporter.main.build_llm_client", return_value=MagicMock()), patch(
        "agentic_options_reporter.main.run_thesis_pipeline", side_effect=failing_pipeline
    ):
        response = test_client.post(f"/runs/{run_id}/thesis/stream")

    assert response.status_code == 200
    frames = _parse_sse(response.text)
    assert frames[-1][0] == "error"
    assert "provider exploded" in frames[-1][1]["detail"]


def test_generate_thesis_stream_not_found_before_streaming(client):
    test_client, _ = client
    response = test_client.post("/runs/999/thesis/stream")
    assert response.status_code == 404


def test_generate_thesis_stream_llm_build_error_returns_502(client):
    from agentic_options_reporter.thesis.llm_client import LlmError

    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)

    with patch("agentic_options_reporter.main.build_llm_client", side_effect=LlmError("no key")):
        response = test_client.post(f"/runs/{run_id}/thesis/stream")

    assert response.status_code == 502


def test_generate_thesis_conflicts_without_regenerate(client):
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)
    fake_result = _fake_thesis_result(run_id)

    with patch("agentic_options_reporter.main.build_llm_client", return_value=MagicMock()), patch(
        "agentic_options_reporter.main.run_thesis_pipeline", return_value=fake_result
    ):
        first = test_client.post(f"/runs/{run_id}/thesis")
        second = test_client.post(f"/runs/{run_id}/thesis")

    assert first.status_code == 200
    assert second.status_code == 409


def test_generate_thesis_regenerate_replaces_existing(client):
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)
    fake_result = _fake_thesis_result(run_id)

    with patch("agentic_options_reporter.main.build_llm_client", return_value=MagicMock()), patch(
        "agentic_options_reporter.main.run_thesis_pipeline", return_value=fake_result
    ):
        test_client.post(f"/runs/{run_id}/thesis")
        response = test_client.post(f"/runs/{run_id}/thesis", json={"regenerate": True})

    assert response.status_code == 200


def test_generate_thesis_llm_error_returns_502(client):
    from agentic_options_reporter.thesis.llm_client import LlmError

    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)

    with patch("agentic_options_reporter.main.build_llm_client", side_effect=LlmError("no key")):
        response = test_client.post(f"/runs/{run_id}/thesis")

    assert response.status_code == 502


def test_generate_thesis_passes_custom_provider_and_api_key(client):
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)
    fake_result = _fake_thesis_result(run_id)

    captured = {}

    def fake_build_llm_client(provider, api_key=None, model=None, max_tokens=1024):
        captured["provider"] = provider
        captured["api_key"] = api_key
        captured["model"] = model
        return MagicMock()

    with patch(
        "agentic_options_reporter.main.build_llm_client", side_effect=fake_build_llm_client
    ), patch("agentic_options_reporter.main.run_thesis_pipeline", return_value=fake_result):
        response = test_client.post(
            f"/runs/{run_id}/thesis",
            json={"provider": "openai", "api_key": "sk-custom-123"},
        )

    assert response.status_code == 200
    assert captured["provider"] == "openai"
    assert captured["api_key"] == "sk-custom-123"
    # settings.llm_model is anthropic-specific; a non-anthropic provider
    # must not receive it as its model.
    assert captured["model"] is None


def test_generate_thesis_defaults_to_auto_provider(client):
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)
    fake_result = _fake_thesis_result(run_id)

    captured = {}

    def fake_build_llm_client(provider, api_key=None, model=None, max_tokens=1024):
        captured["provider"] = provider
        captured["model"] = model
        return MagicMock()

    with patch(
        "agentic_options_reporter.main.build_llm_client", side_effect=fake_build_llm_client
    ), patch("agentic_options_reporter.main.run_thesis_pipeline", return_value=fake_result):
        response = test_client.post(f"/runs/{run_id}/thesis")

    assert response.status_code == 200
    assert captured["provider"] == "auto"
    # settings.llm_model is anthropic-specific; "auto" must not receive it.
    assert captured["model"] is None


def test_generate_thesis_explicit_anthropic_provider_uses_settings_model(client):
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)
    fake_result = _fake_thesis_result(run_id)

    captured = {}

    def fake_build_llm_client(provider, api_key=None, model=None, max_tokens=1024):
        captured["provider"] = provider
        captured["model"] = model
        return MagicMock()

    with patch(
        "agentic_options_reporter.main.build_llm_client", side_effect=fake_build_llm_client
    ), patch("agentic_options_reporter.main.run_thesis_pipeline", return_value=fake_result):
        response = test_client.post(f"/runs/{run_id}/thesis", json={"provider": "anthropic"})

    assert response.status_code == 200
    assert captured["provider"] == "anthropic"
    assert captured["model"] == main_module.get_settings().llm_model


def test_generate_thesis_auto_provider_with_api_key_returns_422(client):
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)

    response = test_client.post(
        f"/runs/{run_id}/thesis", json={"provider": "auto", "api_key": "sk-custom-123"}
    )

    assert response.status_code == 422


def test_generate_thesis_unsupported_provider_returns_502(client):
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)

    response = test_client.post(f"/runs/{run_id}/thesis", json={"provider": "made-up-provider"})

    assert response.status_code == 502


def test_get_thesis_not_found_before_generation(client):
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)

    response = test_client.get(f"/runs/{run_id}/thesis")
    assert response.status_code == 404


def test_get_thesis_after_generation(client):
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)
    fake_result = _fake_thesis_result(run_id)

    with patch("agentic_options_reporter.main.build_llm_client", return_value=MagicMock()), patch(
        "agentic_options_reporter.main.run_thesis_pipeline", return_value=fake_result
    ):
        test_client.post(f"/runs/{run_id}/thesis")

    response = test_client.get(f"/runs/{run_id}/thesis")
    assert response.status_code == 200
    assert response.json()["quant_interpretation"]["narrative"] == "Strong trend."


def test_generate_thesis_no_candidate_short_circuit(client):
    test_client, session_factory = client
    recommendation = Recommendation(action="AVOID", contract_symbol=None, confidence=0.0, rationale="no candidates")
    run_id = _persist_full_run(session_factory, recommendation=recommendation)

    fake_result = AgentThesisResult(
        run_id=run_id,
        generated_at=datetime.now(timezone.utc),
        quant_interpretation=QuantInterpretation(
            narrative="no candidates",
            key_factors=[],
            quant_trade_quality=compute_composite_score({}, source="quant", contract_symbol=None),
            technical_domain_score=DomainScore(
                domain="technical", score=0.0, confidence=0.0, evidence=[], factors=[],
                source="agent", generated_at=datetime(2026, 1, 1),
            ),
        ),
        risk_assessment=None,
        strategy_suggestion=None,
        investment_thesis=InvestmentThesis(thesis="No position recommended.", consensus="neutral"),
    )

    with patch("agentic_options_reporter.main.build_llm_client", return_value=MagicMock()), patch(
        "agentic_options_reporter.main.run_thesis_pipeline", return_value=fake_result
    ):
        response = test_client.post(f"/runs/{run_id}/thesis")

    assert response.status_code == 200
    body = response.json()
    assert body["risk_assessment"] is None
    assert body["strategy_suggestion"] is None


def test_generate_thesis_persists_and_returns_research_findings(client):
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)
    fake_result = _fake_thesis_result(run_id)
    fake_result.financial_research = FinancialResearchFinding(
        company_health="strong", growth="accelerating", profitability="high",
        cash_flow="positive", analyst_consensus="Buy", narrative="Fundamentals solid.",
        domain_score=_domain_score("fundamental"),
    )
    fake_result.news_research = NewsResearchFinding(
        sentiment="bullish", summary="Positive coverage.", catalysts=["earnings beat"], risks=["supply chain"],
        domain_score=_domain_score("sentiment"),
    )
    fake_result.macro_research = MacroResearchFinding(
        regime="risk_on", outlook="Favorable.", summary="Rates steady.", domain_score=_domain_score("macro")
    )
    fake_result.catalyst_research = CatalystFinding(
        net_bias="bullish",
        summary="Earnings just beat; fresh 8-K.",
        catalysts=[
            CatalystItem(
                title="Q2 earnings beat", category="earnings", horizon="recent",
                direction="bullish", detail="Beat consensus.",
            ),
            CatalystItem(
                title="8-K filed", category="filing", horizon="recent",
                direction="uncertain", detail="Material event disclosed.",
            ),
        ],
    )

    with patch("agentic_options_reporter.main.build_llm_client", return_value=MagicMock()), patch(
        "agentic_options_reporter.main.run_thesis_pipeline", return_value=fake_result
    ):
        response = test_client.post(f"/runs/{run_id}/thesis")

    assert response.status_code == 200
    body = response.json()
    assert body["financial_research"]["analyst_consensus"] == "Buy"
    assert body["news_research"]["catalysts"] == ["earnings beat"]
    assert body["macro_research"]["regime"] == "risk_on"
    assert body["catalyst_research"]["net_bias"] == "bullish"
    assert len(body["catalyst_research"]["catalysts"]) == 2
    assert body["catalyst_research"]["catalysts"][0]["category"] == "earnings"

    # Round-trip through the GET endpoint (reads back from persistence).
    get_response = test_client.get(f"/runs/{run_id}/thesis")
    assert get_response.status_code == 200
    get_body = get_response.json()
    assert get_body["financial_research"]["company_health"] == "strong"
    assert get_body["news_research"]["sentiment"] == "bullish"
    assert get_body["macro_research"]["summary"] == "Rates steady."
    assert get_body["catalyst_research"]["summary"] == "Earnings just beat; fresh 8-K."
    assert get_body["catalyst_research"]["catalysts"][1]["title"] == "8-K filed"


def test_generate_thesis_pipeline_warnings_round_trip(client):
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)
    fake_result = _fake_thesis_result(run_id)
    fake_result.pipeline_warnings = [
        "news_research: provider failed during the run — GDELT rate limited: 429"
    ]

    with patch("agentic_options_reporter.main.build_llm_client", return_value=MagicMock()), patch(
        "agentic_options_reporter.main.run_thesis_pipeline", return_value=fake_result
    ):
        response = test_client.post(f"/runs/{run_id}/thesis")

    # The run succeeds (200, not 502) and carries the warning.
    assert response.status_code == 200
    assert response.json()["pipeline_warnings"] == fake_result.pipeline_warnings

    # Round-trip through persistence.
    get_response = test_client.get(f"/runs/{run_id}/thesis")
    assert get_response.json()["pipeline_warnings"] == fake_result.pipeline_warnings


def test_generate_thesis_defaults_to_empty_pipeline_warnings(client):
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)
    fake_result = _fake_thesis_result(run_id)

    with patch("agentic_options_reporter.main.build_llm_client", return_value=MagicMock()), patch(
        "agentic_options_reporter.main.run_thesis_pipeline", return_value=fake_result
    ):
        test_client.post(f"/runs/{run_id}/thesis")

    response = test_client.get(f"/runs/{run_id}/thesis")
    assert response.json()["pipeline_warnings"] == []


def test_generate_thesis_absent_research_findings_round_trip_as_null(client):
    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)
    fake_result = _fake_thesis_result(run_id)

    with patch("agentic_options_reporter.main.build_llm_client", return_value=MagicMock()), patch(
        "agentic_options_reporter.main.run_thesis_pipeline", return_value=fake_result
    ):
        test_client.post(f"/runs/{run_id}/thesis")

    response = test_client.get(f"/runs/{run_id}/thesis")
    body = response.json()
    assert body["financial_research"] is None
    assert body["news_research"] is None
    assert body["macro_research"] is None
    assert body["catalyst_research"] is None


@pytest.mark.parametrize(
    "error_cls_path",
    [
        "agentic_options_reporter.data.financial.FinancialProviderError",
        "agentic_options_reporter.data.news.NewsProviderError",
        "agentic_options_reporter.data.macro.MacroProviderError",
        "agentic_options_reporter.data.sec_provider.SecProviderError",
    ],
)
def test_generate_thesis_configured_provider_failure_returns_502(client, error_cls_path):
    """A provider that IS configured but fails at call time (rate limited,
    bad ticker, network down) is a real error — unlike an unconfigured
    provider, which is handled silently as None (see provider_availability
    in specs/providers.yaml)."""
    import importlib

    module_path, _, cls_name = error_cls_path.rpartition(".")
    error_cls = getattr(importlib.import_module(module_path), cls_name)

    test_client, session_factory = client
    run_id = _persist_full_run(session_factory)

    with patch("agentic_options_reporter.main.build_llm_client", return_value=MagicMock()), patch(
        "agentic_options_reporter.main.run_thesis_pipeline", side_effect=error_cls("provider call failed")
    ):
        response = test_client.post(f"/runs/{run_id}/thesis")

    assert response.status_code == 502


def test_optional_financial_provider_returns_keyless_yahoo_when_unconfigured(monkeypatch):
    # Yahoo Finance needs no API key, so the fundamentals provider is never
    # fully "unconfigured" — it always falls back to Yahoo alone (like
    # Hacker News for news, IMF/World Bank for macro).
    for var in ("FMP_API_KEY", "FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    provider = main_module._optional_financial_provider()

    assert provider is not None
    assert provider.provider_names == ["yfinance"]


def test_optional_news_provider_returns_router_with_keyless_sources_when_others_unconfigured(monkeypatch):
    # Yahoo and Hacker News need no API key, so the news provider is never
    # fully "unconfigured" — it always falls back to those keyless sources.
    for var in (
        "FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEY", "NEWSAPI_API_KEY",
        "NEWSDATA_API_KEY", "GUARDIAN_API_KEY", "GNEWS_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    provider = main_module._optional_news_provider()

    assert provider is not None
    assert provider.provider_names == ["yfinance", "hackernews"]


def test_optional_macro_provider_returns_router_with_keyless_sources_when_unconfigured(monkeypatch):
    # IMF and the World Bank need no API key, so the macro provider is
    # never fully "unconfigured" — it always falls back to them.
    for var in ("FRED_API_KEY", "BLS_API_KEY", "BEA_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    provider = main_module._optional_macro_provider()

    assert provider is not None
    assert provider.provider_names == ["imf", "worldbank"]


def test_optional_financial_provider_returns_router_when_configured(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)

    provider = main_module._optional_financial_provider()

    assert provider is not None
    # FMP plus keyless Yahoo (always available).
    assert provider.provider_names == ["fmp", "yfinance"]


def test_optional_sec_provider_is_always_available():
    # SEC EDGAR is keyless, so the catalyst agent always has at least the
    # filings stream (like Hacker News for news, IMF/World Bank for macro).
    provider = main_module._optional_sec_provider()
    assert provider is not None
