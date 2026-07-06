"""Persistence layer. Schema is authoritative in specs/database.yaml."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, func, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_DIR = _PROJECT_ROOT / "alembic"


def _add_missing_columns(engine) -> None:
    """Reconcile any column an ORM model declares but the live table lacks.

    Every add-column migration to date is a plain nullable column (see
    alembic/versions/*), so a bare ALTER TABLE ADD COLUMN reproduces exactly
    what upgrade() would have run — this exists only for the recovery path
    below, where upgrade() aborted before it could apply them."""
    inspector = inspect(engine)
    missing = [
        (table.name, column)
        for table in Base.metadata.sorted_tables
        if inspector.has_table(table.name)
        for column in table.columns
        if column.name not in {col["name"] for col in inspector.get_columns(table.name)}
    ]
    if not missing:
        return
    with engine.begin() as conn:
        for table_name, column in missing:
            ddl_type = column.type.compile(dialect=engine.dialect)
            conn.execute(text(f'ALTER TABLE "{table_name}" ADD COLUMN "{column.name}" {ddl_type}'))


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
            # Two different situations raise the same "table already
            # exists" error here: (a) under uvicorn --reload, multiple
            # worker processes racing to apply the same initial migration
            # against an already-current database, and (b) a database
            # created before Alembic tracking existed (or before a later
            # add-column migration was written), where upgrade() aborts on
            # the very first migration and every add-column migration after
            # it — fundamentals/data_warnings, catalyst_research,
            # pipeline_warnings — never runs. create_all() alone only
            # handles (a): it fills in wholly-missing tables but can't add a
            # missing column to a table that already exists, so before
            # stamping the revision (which marks all of that as "already
            # applied", whether or not it actually was) reconcile any
            # columns the ORM model has that the live table doesn't.
            engine = create_engine(database_url, connect_args={"check_same_thread": False})
            try:
                Base.metadata.create_all(engine)
                _add_missing_columns(engine)
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
    TradeQualityScoreRow,
    TrendAssessmentRow,
    VolumeAssessmentRow,
)
from agentic_options_reporter.models.schemas import (
    AgentThesisResult,
    FundamentalsSnapshot,
    IndicatorSnapshot,
    PastRunOutcome,
    Recommendation,
    ScoredCandidate,
    SupportResistanceLevel,
    TradeQualityScore,
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


def _dump(model) -> dict | None:
    """model_dump(mode="json") so nested dates/datetimes serialize to ISO
    strings the JSON column can hold; None passes through unchanged."""
    return model.model_dump(mode="json") if model is not None else None


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
    trade_quality: TradeQualityScore | None,
    weighting_profile: str,
    fundamentals: FundamentalsSnapshot | None = None,
    data_warnings: list[str] | None = None,
) -> int:
    run = AnalysisRun(
        symbol=symbol,
        generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
        lookback_days=lookback_days,
        expiration=expiration,
        weighting_profile=weighting_profile,
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
                **candidate.model_dump(exclude={"domain_scores"}),
                domain_scores={name: _dump(ds) for name, ds in candidate.domain_scores.items()},
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

    if trade_quality is not None:
        session.add(
            TradeQualityScoreRow(
                run_id=run.id,
                contract_symbol=trade_quality.contract_symbol,
                composite_score=trade_quality.composite_score,
                confidence=trade_quality.confidence,
                recommendation_action=trade_quality.recommendation_action,
                weighting_profile=trade_quality.weighting_profile,
                domain_scores={name: _dump(ds) for name, ds in trade_quality.domain_scores.items()},
                explainability=trade_quality.explainability,
                generated_at=trade_quality.generated_at,
            )
        )

    session.commit()
    return run.id


def fetch_recent_runs_for_symbol(
    session: Session, symbol: str, limit: int = 20
) -> list[PastRunOutcome]:
    """This symbol's past AnalysisRuns, most recent first, for the
    Statistical Edge domain scorer's historical hit-rate lookback
    (analysis/statistical_edge.py). Called before the current run is
    persisted, so it naturally only sees prior runs."""
    rows = (
        session.query(AnalysisRun, RecommendationRow)
        .join(RecommendationRow, RecommendationRow.run_id == AnalysisRun.id)
        .filter(func.upper(AnalysisRun.symbol) == symbol.upper())
        .order_by(AnalysisRun.generated_at.desc())
        .limit(limit)
        .all()
    )
    outcomes: list[PastRunOutcome] = []
    for run, rec in rows:
        option_type = None
        if rec.contract_symbol:
            candidate = (
                session.query(ScoredCandidateRow)
                .filter(
                    ScoredCandidateRow.run_id == run.id,
                    ScoredCandidateRow.contract_symbol == rec.contract_symbol,
                )
                .one_or_none()
            )
            option_type = candidate.option_type if candidate else None
        outcomes.append(
            PastRunOutcome(
                run_id=run.id,
                generated_at=run.generated_at,
                action=rec.action,
                option_type=option_type,
                contract_symbol=rec.contract_symbol,
            )
        )
    return outcomes


def persist_thesis(session: Session, thesis_result: AgentThesisResult) -> None:
    quant = thesis_result.quant_interpretation
    financial = thesis_result.financial_research
    news = thesis_result.news_research
    macro = thesis_result.macro_research
    catalyst = thesis_result.catalyst_research
    relative_strength = thesis_result.relative_strength_research
    statistical_edge = thesis_result.statistical_edge_research
    risk = thesis_result.risk_assessment
    strategy = thesis_result.strategy_suggestion

    session.add(
        AgentThesisRow(
            run_id=thesis_result.run_id,
            generated_at=thesis_result.generated_at,
            quant_narrative=quant.narrative,
            quant_key_factors=quant.key_factors,
            quant_trade_quality=_dump(quant.quant_trade_quality),
            technical_domain_score=_dump(quant.technical_domain_score),
            financial_company_health=financial.company_health if financial else None,
            financial_growth=financial.growth if financial else None,
            financial_profitability=financial.profitability if financial else None,
            financial_cash_flow=financial.cash_flow if financial else None,
            financial_analyst_consensus=financial.analyst_consensus if financial else None,
            financial_narrative=financial.narrative if financial else None,
            fundamental_domain_score=_dump(financial.domain_score) if financial else None,
            news_sentiment=news.sentiment if news else None,
            news_summary=news.summary if news else None,
            news_catalysts=news.catalysts if news else None,
            news_risks=news.risks if news else None,
            sentiment_domain_score=_dump(news.domain_score) if news else None,
            macro_regime=macro.regime if macro else None,
            macro_outlook=macro.outlook if macro else None,
            macro_summary=macro.summary if macro else None,
            macro_domain_score=_dump(macro.domain_score) if macro else None,
            catalyst_net_bias=catalyst.net_bias if catalyst else None,
            catalyst_summary=catalyst.summary if catalyst else None,
            catalyst_items=(
                [item.model_dump() for item in catalyst.catalysts] if catalyst else None
            ),
            relative_strength_narrative=relative_strength.narrative if relative_strength else None,
            relative_strength_domain_score=(
                _dump(relative_strength.domain_score) if relative_strength else None
            ),
            statistical_edge_narrative=statistical_edge.narrative if statistical_edge else None,
            statistical_edge_domain_score=(
                _dump(statistical_edge.domain_score) if statistical_edge else None
            ),
            risk_level=risk.risk_level if risk else None,
            risk_concerns=risk.concerns if risk else None,
            risk_position_sizing_note=risk.position_sizing_note if risk else None,
            risk_domain_score=_dump(risk.domain_score) if risk else None,
            strategy=strategy.strategy if strategy else None,
            strategy_rationale=strategy.rationale if strategy else None,
            liquidity_domain_score=_dump(strategy.domain_score) if strategy else None,
            thesis=thesis_result.investment_thesis.thesis,
            consensus=thesis_result.investment_thesis.consensus,
            agent_trade_quality=_dump(thesis_result.agent_trade_quality),
            pipeline_warnings=thesis_result.pipeline_warnings,
        )
    )
    session.commit()


def delete_thesis(session: Session, run_id: int) -> None:
    existing = session.query(AgentThesisRow).filter(AgentThesisRow.run_id == run_id).one_or_none()
    if existing is not None:
        session.delete(existing)
        session.commit()
