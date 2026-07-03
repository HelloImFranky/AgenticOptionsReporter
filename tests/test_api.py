from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agentic_options_reporter import main as main_module
from agentic_options_reporter.models.schemas import (
    AgentThesisResult,
    IndicatorSnapshot,
    InvestmentThesis,
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
        score_breakdown={"trend_alignment": 1.0, "liquidity": 0.8},
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
        quant_interpretation=QuantInterpretation(
            narrative="Strong trend.", key_factors=["trend"], score_breakdown={"x": 1.0}, overall_score=78.5
        ),
        risk_assessment=RiskAssessment(risk_level="medium", concerns=["high IV"], position_sizing_note="Size at 2%."),
        strategy_suggestion=StrategySuggestion(strategy="Bull Call Spread", rationale="Defined risk."),
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


def test_generate_thesis_default_provider_uses_settings_model(client):
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
    assert captured["provider"] == "anthropic"
    assert captured["model"] == main_module.get_settings().llm_model


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
            narrative="no candidates", key_factors=[], score_breakdown={}, overall_score=0.0
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
