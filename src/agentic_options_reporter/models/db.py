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
    scored_candidates: Mapped[list["ScoredCandidateRow"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    recommendation: Mapped["RecommendationRow"] = relationship(
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
