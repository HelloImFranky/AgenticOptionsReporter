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
    max_loss: Mapped[float] = mapped_column(Float)
    max_gain: Mapped[float | None] = mapped_column(Float, nullable=True)
    breakeven: Mapped[float] = mapped_column(Float)
    reward_risk_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    probability_of_profit: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    score_breakdown: Mapped[dict] = mapped_column(JSON)

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


class AgentThesisRow(Base):
    """See specs/agents.yaml for the pipeline that produces this row."""

    __tablename__ = "agent_thesis"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_run.id"), unique=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime)

    quant_narrative: Mapped[str] = mapped_column(String)
    quant_key_factors: Mapped[list] = mapped_column(JSON)
    quant_score_breakdown: Mapped[dict] = mapped_column(JSON)
    quant_overall_score: Mapped[float] = mapped_column(Float)

    # Null when the corresponding provider wasn't configured (see
    # specs/providers.yaml provider_availability).
    financial_company_health: Mapped[str | None] = mapped_column(String, nullable=True)
    financial_growth: Mapped[str | None] = mapped_column(String, nullable=True)
    financial_profitability: Mapped[str | None] = mapped_column(String, nullable=True)
    financial_cash_flow: Mapped[str | None] = mapped_column(String, nullable=True)
    financial_analyst_consensus: Mapped[str | None] = mapped_column(String, nullable=True)
    financial_narrative: Mapped[str | None] = mapped_column(String, nullable=True)

    news_sentiment: Mapped[str | None] = mapped_column(String, nullable=True)
    news_summary: Mapped[str | None] = mapped_column(String, nullable=True)
    news_catalysts: Mapped[list | None] = mapped_column(JSON, nullable=True)
    news_risks: Mapped[list | None] = mapped_column(JSON, nullable=True)

    macro_regime: Mapped[str | None] = mapped_column(String, nullable=True)
    macro_outlook: Mapped[str | None] = mapped_column(String, nullable=True)
    macro_summary: Mapped[str | None] = mapped_column(String, nullable=True)

    risk_level: Mapped[str | None] = mapped_column(String, nullable=True)
    risk_concerns: Mapped[list | None] = mapped_column(JSON, nullable=True)
    risk_position_sizing_note: Mapped[str | None] = mapped_column(String, nullable=True)

    strategy: Mapped[str | None] = mapped_column(String, nullable=True)
    strategy_rationale: Mapped[str | None] = mapped_column(String, nullable=True)

    thesis: Mapped[str] = mapped_column(String)
    consensus: Mapped[str] = mapped_column(String)

    run: Mapped[AnalysisRun] = relationship(back_populates="agent_thesis")
