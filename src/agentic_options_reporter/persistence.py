"""Persistence layer. Schema is authoritative in specs/database.yaml."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_DIR = _PROJECT_ROOT / "alembic"


def run_migrations(database_url: str) -> None:
    """Bring the target database up to the latest Alembic revision."""
    from alembic import command
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(_ALEMBIC_DIR))
    cfg.set_main_option("sqlalchemy.url", database_url)

    try:
        command.upgrade(cfg, "head")
    except OperationalError as exc:
        message = str(exc).lower()
        if database_url.startswith("sqlite") and (
            "already exists" in message or "table" in message and "exists" in message
        ):
            # Under uvicorn --reload, multiple worker processes can race to apply
            # the same initial migration. If that happens, repair any missing
            # tables from the current ORM model and then stamp the revision so
            # subsequent starts use the same schema state.
            engine = create_engine(database_url, connect_args={"check_same_thread": False})
            try:
                Base.metadata.create_all(engine)
            finally:
                engine.dispose()
            command.stamp(cfg, "head")
            return
        raise

from agentic_options_reporter.models.db import (
    AgentThesisRow,
    AnalysisRun,
    Base,
    IndicatorSnapshotRow,
    RecommendationRow,
    ScoredCandidateRow,
    SupportResistanceLevelRow,
    TrendAssessmentRow,
    VolumeAssessmentRow,
)
from agentic_options_reporter.models.schemas import (
    AgentThesisResult,
    FundamentalsSnapshot,
    IndicatorSnapshot,
    Recommendation,
    ScoredCandidate,
    SupportResistanceLevel,
    TrendAssessment,
    VolumeAssessment,
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
    if is_memory:
        # In-memory databases are ephemeral (tests); create the schema directly
        # rather than paying Alembic's migration overhead on every run.
        Base.metadata.create_all(engine)
    else:
        # Persistent databases are migrated to head so schema changes ship
        # automatically without manual ALTER TABLE.
        run_migrations(database_url)
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
    trend: TrendAssessment,
    volume: VolumeAssessment,
    levels: list[SupportResistanceLevel],
    candidates: list[ScoredCandidate],
    recommendation: Recommendation,
    fundamentals: FundamentalsSnapshot | None = None,
    data_warnings: list[str] | None = None,
) -> int:
    run = AnalysisRun(
        symbol=symbol,
        generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
        lookback_days=lookback_days,
        expiration=expiration,
        # mode="json" so nested dates (earnings calendar, insider filings)
        # serialize to ISO strings the JSON column can hold.
        fundamentals=fundamentals.model_dump(mode="json") if fundamentals is not None else None,
        data_warnings=data_warnings or None,
    )
    session.add(run)
    session.flush()  # populate run.id

    session.add(
        IndicatorSnapshotRow(run_id=run.id, **indicators.model_dump())
    )
    session.add(TrendAssessmentRow(run_id=run.id, **trend.model_dump()))
    session.add(VolumeAssessmentRow(run_id=run.id, **volume.model_dump()))
    for level in levels:
        session.add(SupportResistanceLevelRow(run_id=run.id, **level.model_dump()))

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


def persist_thesis(session: Session, thesis_result: AgentThesisResult) -> None:
    quant = thesis_result.quant_interpretation
    financial = thesis_result.financial_research
    news = thesis_result.news_research
    macro = thesis_result.macro_research
    catalyst = thesis_result.catalyst_research
    risk = thesis_result.risk_assessment
    strategy = thesis_result.strategy_suggestion

    session.add(
        AgentThesisRow(
            run_id=thesis_result.run_id,
            generated_at=thesis_result.generated_at,
            quant_narrative=quant.narrative,
            quant_key_factors=quant.key_factors,
            quant_score_breakdown=quant.score_breakdown,
            quant_overall_score=quant.overall_score,
            financial_company_health=financial.company_health if financial else None,
            financial_growth=financial.growth if financial else None,
            financial_profitability=financial.profitability if financial else None,
            financial_cash_flow=financial.cash_flow if financial else None,
            financial_analyst_consensus=financial.analyst_consensus if financial else None,
            financial_narrative=financial.narrative if financial else None,
            news_sentiment=news.sentiment if news else None,
            news_summary=news.summary if news else None,
            news_catalysts=news.catalysts if news else None,
            news_risks=news.risks if news else None,
            macro_regime=macro.regime if macro else None,
            macro_outlook=macro.outlook if macro else None,
            macro_summary=macro.summary if macro else None,
            catalyst_net_bias=catalyst.net_bias if catalyst else None,
            catalyst_summary=catalyst.summary if catalyst else None,
            catalyst_items=(
                [item.model_dump() for item in catalyst.catalysts] if catalyst else None
            ),
            risk_level=risk.risk_level if risk else None,
            risk_concerns=risk.concerns if risk else None,
            risk_position_sizing_note=risk.position_sizing_note if risk else None,
            strategy=strategy.strategy if strategy else None,
            strategy_rationale=strategy.rationale if strategy else None,
            thesis=thesis_result.investment_thesis.thesis,
            consensus=thesis_result.investment_thesis.consensus,
            pipeline_warnings=thesis_result.pipeline_warnings,
        )
    )
    session.commit()


def delete_thesis(session: Session, run_id: int) -> None:
    existing = session.query(AgentThesisRow).filter(AgentThesisRow.run_id == run_id).one_or_none()
    if existing is not None:
        session.delete(existing)
        session.commit()
