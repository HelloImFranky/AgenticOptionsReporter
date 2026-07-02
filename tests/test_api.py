from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from agentic_options_reporter import main as main_module
from agentic_options_reporter.models.schemas import (
    IndicatorSnapshot,
    Recommendation,
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


def test_get_run_and_list_runs(client):
    test_client, session_factory = client
    recommendation = Recommendation(
        action="BUY", contract_symbol="TESTC00100000", confidence=0.7, rationale="test"
    )
    with session_factory() as session:
        run_id = persist_analysis_run(
            session, "TEST", 260, None, _indicator_snapshot(), [], recommendation
        )

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
