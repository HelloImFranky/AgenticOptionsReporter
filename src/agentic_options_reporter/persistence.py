"""Persistence layer. Schema is authoritative in specs/database.yaml."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from agentic_options_reporter.models.db import (
    AnalysisRun,
    Base,
    IndicatorSnapshotRow,
    RecommendationRow,
    ScoredCandidateRow,
)
from agentic_options_reporter.models.schemas import (
    IndicatorSnapshot,
    Recommendation,
    ScoredCandidate,
)


def make_engine(database_url: str):
    is_sqlite = database_url.startswith("sqlite")
    is_memory = ":memory:" in database_url
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    # In-memory SQLite lives per-connection; pin the pool to one shared
    # connection so it survives across the request-handling thread pool.
    poolclass = StaticPool if is_memory else None
    kwargs = {"connect_args": connect_args}
    if poolclass is not None:
        kwargs["poolclass"] = poolclass
    engine = create_engine(database_url, **kwargs)
    Base.metadata.create_all(engine)
    return engine


def make_session_factory(database_url: str) -> sessionmaker:
    engine = make_engine(database_url)
    return sessionmaker(bind=engine, expire_on_commit=False)


def persist_analysis_run(
    session: Session,
    symbol: str,
    lookback_days: int,
    expiration: str | None,
    indicators: IndicatorSnapshot,
    candidates: list[ScoredCandidate],
    recommendation: Recommendation,
) -> int:
    run = AnalysisRun(
        symbol=symbol,
        generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
        lookback_days=lookback_days,
        expiration=expiration,
    )
    session.add(run)
    session.flush()  # populate run.id

    session.add(
        IndicatorSnapshotRow(run_id=run.id, **indicators.model_dump())
    )

    for candidate in candidates:
        session.add(
            ScoredCandidateRow(
                run_id=run.id,
                **candidate.model_dump(exclude={"score_breakdown"}),
                score_breakdown=candidate.score_breakdown,
            )
        )

    session.add(
        RecommendationRow(
            run_id=run.id,
            action=recommendation.action,
            contract_symbol=recommendation.contract_symbol,
            confidence=recommendation.confidence,
            rationale=recommendation.rationale,
        )
    )

    session.commit()
    return run.id
