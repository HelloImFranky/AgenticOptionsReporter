from datetime import date, datetime

import sqlalchemy
import sqlalchemy.exc
from sqlalchemy import inspect

from agentic_options_reporter.analysis.composite_score import compute_composite_score
from agentic_options_reporter.models.db import AnalysisRun, Base
from agentic_options_reporter.models.schemas import (
    DomainScore,
    IndicatorSnapshot,
    Recommendation,
    ScoredCandidate,
    SupportResistanceLevel,
    TrendAssessment,
    VolumeAssessment,
)
from agentic_options_reporter.persistence import (
    fetch_recent_runs_for_symbol,
    make_session_factory,
    persist_analysis_run,
    run_migrations,
)


def _indicators() -> IndicatorSnapshot:
    return IndicatorSnapshot(
        sma_20=100, sma_50=98, sma_200=None, ema_12=101, ema_26=99, adx_14=30,
        rsi_14=55, macd=1.2, macd_signal=1.0, macd_histogram=0.2, stoch_k=60,
        stoch_d=58, bb_upper=110, bb_middle=100, bb_lower=90, atr_14=2.5,
        obv=1_000_000, volume_sma_20=900_000,
    )


def _candidate(symbol: str, option_type: str = "call", score: float = 80.0) -> ScoredCandidate:
    domain_scores = {
        "technical": DomainScore(
            domain="technical", score=score, confidence=90.0, evidence=[], factors=[],
            source="quant", generated_at=datetime(2026, 1, 1),
        )
    }
    return ScoredCandidate(
        contract_symbol=f"{symbol}C00100000", option_type=option_type, strike=100.0, expiration=date(2026, 1, 16),
        delta=0.55, gamma=0.02, theta=-0.05, vega=0.1, rho=0.02,
        open_interest=800, spread_pct=0.05, volume=100,
        max_loss=250.0, max_gain=None, breakeven=102.5, reward_risk_ratio=None,
        probability_of_profit=0.6, score=score, domain_scores=domain_scores,
    )


def _persist_run(session_factory, symbol: str, action: str, option_type: str, score: float) -> int:
    candidate = _candidate(symbol, option_type, score)
    recommendation = Recommendation(
        action=action, contract_symbol=candidate.contract_symbol, confidence=score / 100, rationale="test"
    )
    trade_quality = compute_composite_score(
        candidate.domain_scores, source="quant", contract_symbol=candidate.contract_symbol
    )
    with session_factory() as session:
        return persist_analysis_run(
            session, symbol, 260, None, _indicators(),
            TrendAssessment(direction="bullish", strength="moderate", adx=25),
            VolumeAssessment(relative_volume=1.2, flags=["normal_volume"]),
            [SupportResistanceLevel(price=95.0, level_type="support", touches=3, last_touch_index=10)],
            [candidate], recommendation, trade_quality, "swing",
        )


def test_persist_analysis_run_stores_trade_quality_score_row():
    session_factory = make_session_factory("sqlite:///:memory:")
    run_id = _persist_run(session_factory, "AAPL", "BUY", "call", 85.0)

    with session_factory() as session:
        run = session.get(AnalysisRun, run_id)
        assert run.trade_quality_score is not None
        assert run.trade_quality_score.composite_score == 85.0
        assert run.trade_quality_score.weighting_profile == "swing"
        assert "technical" in run.trade_quality_score.domain_scores


def test_fetch_recent_runs_for_symbol_returns_most_recent_first():
    session_factory = make_session_factory("sqlite:///:memory:")
    _persist_run(session_factory, "MSFT", "BUY", "call", 70.0)
    _persist_run(session_factory, "MSFT", "AVOID", "call", 30.0)
    _persist_run(session_factory, "AAPL", "BUY", "call", 90.0)  # different symbol, excluded

    with session_factory() as session:
        outcomes = fetch_recent_runs_for_symbol(session, "msft")  # case-insensitive

    assert len(outcomes) == 2
    assert outcomes[0].action == "AVOID"   # most recently persisted first
    assert outcomes[1].action == "BUY"
    assert all(o.option_type == "call" for o in outcomes)


def test_run_migrations_repair_sqlite_schema_on_existing_table_error(monkeypatch, tmp_path):
    database_url = f"sqlite:///{tmp_path / 'app.db'}"
    calls = []

    def fake_upgrade(cfg, revision):
        calls.append(("upgrade", revision))
        raise sqlalchemy.exc.OperationalError(
            "table analysis_run already exists",
            None,
            None,
        )

    def fake_stamp(cfg, revision):
        calls.append(("stamp", revision))

    monkeypatch.setattr("alembic.command.upgrade", fake_upgrade)
    monkeypatch.setattr("alembic.command.stamp", fake_stamp)

    run_migrations(database_url)

    engine = sqlalchemy.create_engine(database_url, connect_args={"check_same_thread": False})
    try:
        inspector = inspect(engine)
        assert inspector.has_table("analysis_run")
        assert inspector.has_table("support_resistance_level")
    finally:
        engine.dispose()

    assert calls == [("upgrade", "head"), ("stamp", "head")]


def test_run_migrations_backfills_columns_on_pre_alembic_database(tmp_path):
    """A database whose tables predate Alembic tracking (no alembic_version
    row) hits "table already exists" on the very first migration, which
    aborts the whole upgrade chain before any later ADD COLUMN migration
    (fundamentals/data_warnings, catalyst_research, pipeline_warnings) runs.
    Regression: the recovery path used to stamp the revision as fully
    applied anyway, silently leaving those columns missing forever — every
    later attempt to persist a run then failed with 'no such column'."""
    database_url = f"sqlite:///{tmp_path / 'app.db'}"
    engine = sqlalchemy.create_engine(database_url, connect_args={"check_same_thread": False})
    try:
        # Build today's full schema, then physically strip it back down to
        # what a database predating the fundamentals migration looked like.
        Base.metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(sqlalchemy.text(
                "CREATE TABLE analysis_run_old AS "
                "SELECT id, symbol, generated_at, lookback_days, expiration FROM analysis_run"
            ))
            conn.execute(sqlalchemy.text("DROP TABLE analysis_run"))
            conn.execute(sqlalchemy.text("ALTER TABLE analysis_run_old RENAME TO analysis_run"))
    finally:
        engine.dispose()

    run_migrations(database_url)

    engine = sqlalchemy.create_engine(database_url, connect_args={"check_same_thread": False})
    try:
        inspector = inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("analysis_run")}
        assert {"fundamentals", "data_warnings"} <= columns
    finally:
        engine.dispose()
