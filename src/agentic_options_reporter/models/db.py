"""SQLAlchemy models. Schema is authoritative in specs/database.yaml."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class AnalysisRun(Base):
    __tablename__ = "analysis_run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    lookback_days: Mapped[int] = mapped_column(Integer)
    expiration: Mapped[str | None] = mapped_column(String, nullable=True)
    # Weighting profile used to compute this run's Trade Quality Score
    # (day_trade | swing | long_term, see specs/scoring.yaml). Defaulted
    # for older rows that predate the column.
    weighting_profile: Mapped[str] = mapped_column(String, default="swing", server_default="swing")
    # Cross-provider fundamentals snapshot (FundamentalsSnapshot serialized
    # to JSON) gathered alongside the technicals, plus any non-fatal
    # warnings from gathering it. Nullable: older runs predate the column,
    # and a run where no provider returned fundamentals stores null.
    fundamentals: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    data_warnings: Mapped[list | None] = mapped_column(JSON, nullable=True)

    indicator_snapshot: Mapped["IndicatorSnapshotRow"] = relationship(
        back_populates="run", uselist=False, cascade="all, delete-orphan"
    )
    trend_assessment: Mapped["TrendAssessmentRow"] = relationship(
        back_populates="run", uselist=False, cascade="all, delete-orphan"
    )
    volume_assessment: Mapped["VolumeAssessmentRow"] = relationship(
        back_populates="run", uselist=False, cascade="all, delete-orphan"
    )
    support_resistance_levels: Mapped[list["SupportResistanceLevelRow"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    scored_candidates: Mapped[list["ScoredCandidateRow"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    recommendation: Mapped["RecommendationRow"] = relationship(
        back_populates="run", uselist=False, cascade="all, delete-orphan"
    )
    trade_quality_score: Mapped["TradeQualityScoreRow | None"] = relationship(
        back_populates="run", uselist=False, cascade="all, delete-orphan"
    )
    agent_thesis: Mapped["AgentThesisRow | None"] = relationship(
        back_populates="run", uselist=False, cascade="all, delete-orphan"
    )


class IndicatorSnapshotRow(Base):
    __tablename__ = "indicator_snapshot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_run.id"))

    sma_20: Mapped[float] = mapped_column(Float)
    sma_50: Mapped[float] = mapped_column(Float)
    sma_200: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema_12: Mapped[float] = mapped_column(Float)
    ema_26: Mapped[float] = mapped_column(Float)
    adx_14: Mapped[float] = mapped_column(Float)
    rsi_14: Mapped[float] = mapped_column(Float)
    macd: Mapped[float] = mapped_column(Float)
    macd_signal: Mapped[float] = mapped_column(Float)
    macd_histogram: Mapped[float] = mapped_column(Float)
    stoch_k: Mapped[float] = mapped_column(Float)
    stoch_d: Mapped[float] = mapped_column(Float)
    bb_upper: Mapped[float] = mapped_column(Float)
    bb_middle: Mapped[float] = mapped_column(Float)
    bb_lower: Mapped[float] = mapped_column(Float)
    atr_14: Mapped[float] = mapped_column(Float)
    obv: Mapped[float] = mapped_column(Float)
    volume_sma_20: Mapped[float] = mapped_column(Float)

    run: Mapped[AnalysisRun] = relationship(back_populates="indicator_snapshot")


class TrendAssessmentRow(Base):
    __tablename__ = "trend_assessment"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_run.id"), unique=True)

    direction: Mapped[str] = mapped_column(String)
    strength: Mapped[str] = mapped_column(String)
    adx: Mapped[float] = mapped_column(Float)

    run: Mapped[AnalysisRun] = relationship(back_populates="trend_assessment")


class VolumeAssessmentRow(Base):
    __tablename__ = "volume_assessment"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_run.id"), unique=True)

    relative_volume: Mapped[float] = mapped_column(Float)
    flags: Mapped[list] = mapped_column(JSON)

    run: Mapped[AnalysisRun] = relationship(back_populates="volume_assessment")


class SupportResistanceLevelRow(Base):
    __tablename__ = "support_resistance_level"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_run.id"))

    price: Mapped[float] = mapped_column(Float)
    level_type: Mapped[str] = mapped_column(String)
    touches: Mapped[int] = mapped_column(Integer)
    last_touch_index: Mapped[int] = mapped_column(Integer)

    run: Mapped[AnalysisRun] = relationship(back_populates="support_resistance_levels")


class ScoredCandidateRow(Base):
    __tablename__ = "scored_candidate"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_run.id"))

    contract_symbol: Mapped[str] = mapped_column(String)
    option_type: Mapped[str] = mapped_column(String)
    strike: Mapped[float] = mapped_column(Float)
    expiration: Mapped[date] = mapped_column(Date)
    delta: Mapped[float] = mapped_column(Float)
    gamma: Mapped[float] = mapped_column(Float)
    theta: Mapped[float] = mapped_column(Float)
    vega: Mapped[float] = mapped_column(Float)
    rho: Mapped[float] = mapped_column(Float)
    open_interest: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    spread_pct: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    volume: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    max_loss: Mapped[float] = mapped_column(Float)
    max_gain: Mapped[float | None] = mapped_column(Float, nullable=True)
    breakeven: Mapped[float] = mapped_column(Float)
    reward_risk_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    probability_of_profit: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    # dict[str, DomainScore] serialized to JSON — the 8-domain Trade
    # Quality Score breakdown (specs/scoring.yaml). Replaces the legacy
    # flat 5-factor score_breakdown column.
    domain_scores: Mapped[dict] = mapped_column(JSON, default=dict, server_default="{}")

    run: Mapped[AnalysisRun] = relationship(back_populates="scored_candidates")


class RecommendationRow(Base):
    __tablename__ = "recommendation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_run.id"))

    action: Mapped[str] = mapped_column(String)
    contract_symbol: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(String)

    run: Mapped[AnalysisRun] = relationship(back_populates="recommendation")


class TradeQualityScoreRow(Base):
    """The quant-side composite Trade Quality Score for a run's recommended
    candidate (specs/scoring.yaml, specs/database.yaml). Null/absent when
    the run had no liquid candidate (AVOID), mirroring RecommendationRow's
    nullable contract_symbol."""

    __tablename__ = "trade_quality_score"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_run.id"), unique=True)

    contract_symbol: Mapped[str | None] = mapped_column(String, nullable=True)
    composite_score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    recommendation_action: Mapped[str] = mapped_column(String)
    weighting_profile: Mapped[str] = mapped_column(String)
    domain_scores: Mapped[dict] = mapped_column(JSON)
    explainability: Mapped[list] = mapped_column(JSON)
    generated_at: Mapped[datetime] = mapped_column(DateTime)

    run: Mapped[AnalysisRun] = relationship(back_populates="trade_quality_score")


class AgentThesisRow(Base):
    """See specs/agents.yaml for the pipeline that produces this row."""

    __tablename__ = "agent_thesis"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_run.id"), unique=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime)

    quant_narrative: Mapped[str] = mapped_column(String)
    quant_key_factors: Mapped[list] = mapped_column(JSON)
    # TradeQualityScore (source="quant"), pass-through verbatim from the
    # quant engine — replaces the legacy quant_score_breakdown/
    # quant_overall_score columns.
    quant_trade_quality: Mapped[dict] = mapped_column(JSON, default=dict, server_default="{}")
    # This agent's own, independently-authored Technical DomainScore.
    technical_domain_score: Mapped[dict] = mapped_column(JSON, default=dict, server_default="{}")

    # Null when the corresponding provider wasn't configured (see
    # specs/providers.yaml provider_availability).
    financial_company_health: Mapped[str | None] = mapped_column(String, nullable=True)
    financial_growth: Mapped[str | None] = mapped_column(String, nullable=True)
    financial_profitability: Mapped[str | None] = mapped_column(String, nullable=True)
    financial_cash_flow: Mapped[str | None] = mapped_column(String, nullable=True)
    financial_analyst_consensus: Mapped[str | None] = mapped_column(String, nullable=True)
    financial_narrative: Mapped[str | None] = mapped_column(String, nullable=True)
    fundamental_domain_score: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    news_sentiment: Mapped[str | None] = mapped_column(String, nullable=True)
    news_summary: Mapped[str | None] = mapped_column(String, nullable=True)
    news_catalysts: Mapped[list | None] = mapped_column(JSON, nullable=True)
    news_risks: Mapped[list | None] = mapped_column(JSON, nullable=True)
    sentiment_domain_score: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    macro_regime: Mapped[str | None] = mapped_column(String, nullable=True)
    macro_outlook: Mapped[str | None] = mapped_column(String, nullable=True)
    macro_summary: Mapped[str | None] = mapped_column(String, nullable=True)
    macro_domain_score: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Catalyst research (news + SEC filings + macro). Null when none of
    # those providers was configured. `catalyst_items` is the list of
    # {title, category, horizon, direction, detail} dicts.
    catalyst_net_bias: Mapped[str | None] = mapped_column(String, nullable=True)
    catalyst_summary: Mapped[str | None] = mapped_column(String, nullable=True)
    catalyst_items: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # New in phase 3 (specs/agents.yaml): relative-strength / statistical-
    # edge agent findings, null when neither ran (e.g. no_candidate_short_
    # circuit or a provider was unconfigured).
    relative_strength_narrative: Mapped[str | None] = mapped_column(String, nullable=True)
    relative_strength_domain_score: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    statistical_edge_narrative: Mapped[str | None] = mapped_column(String, nullable=True)
    statistical_edge_domain_score: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    risk_level: Mapped[str | None] = mapped_column(String, nullable=True)
    risk_concerns: Mapped[list | None] = mapped_column(JSON, nullable=True)
    risk_position_sizing_note: Mapped[str | None] = mapped_column(String, nullable=True)
    risk_domain_score: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    strategy: Mapped[str | None] = mapped_column(String, nullable=True)
    strategy_rationale: Mapped[str | None] = mapped_column(String, nullable=True)
    liquidity_domain_score: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    thesis: Mapped[str] = mapped_column(String)
    consensus: Mapped[str] = mapped_column(String)

    # The agent-side composite Trade Quality Score (source="agent"),
    # blended from whichever agent DomainScores this run produced. Null
    # only if literally none did.
    agent_trade_quality: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Non-fatal mid-pipeline problems (e.g. a research provider rate
    # limited during the run). Null on rows written before this column
    # existed; read back as an empty list.
    pipeline_warnings: Mapped[list | None] = mapped_column(JSON, nullable=True)

    run: Mapped[AnalysisRun] = relationship(back_populates="agent_thesis")
