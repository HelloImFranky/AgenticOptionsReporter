"""FastAPI surface. Contract is authoritative in specs/api.yaml."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from agentic_options_reporter.config import get_settings
from agentic_options_reporter.data.market_data import MarketDataError
from agentic_options_reporter.models.db import AnalysisRun
from agentic_options_reporter.models.schemas import (
    AnalysisResult,
    AnalysisRunSummary,
    IndicatorSnapshot,
    Recommendation,
    ScoredCandidate,
)
from agentic_options_reporter.persistence import make_session_factory
from agentic_options_reporter.workflow import run_analysis

app = FastAPI(title="AgenticOptionsReporter", version="0.1.0")

_settings = get_settings()
_session_factory = make_session_factory(_settings.database_url)


def _to_analysis_result(run: AnalysisRun) -> AnalysisResult:
    indicators = IndicatorSnapshot.model_validate(run.indicator_snapshot, from_attributes=True)
    candidates = [
        ScoredCandidate.model_validate(row, from_attributes=True)
        for row in sorted(run.scored_candidates, key=lambda c: c.score, reverse=True)
    ]
    recommendation = (
        Recommendation.model_validate(run.recommendation, from_attributes=True)
        if run.recommendation
        else Recommendation(action="AVOID", contract_symbol=None, confidence=0.0, rationale="")
    )
    return AnalysisResult(
        symbol=run.symbol,
        run_id=run.id,
        generated_at=run.generated_at,
        indicators=indicators,
        # Support/resistance levels are transient analysis output, not
        # persisted (see specs/database.yaml); unavailable on replay.
        trend=_placeholder_trend(),
        volume=_placeholder_volume(),
        support_resistance=[],
        candidates=candidates,
        recommendation=recommendation,
    )


def _placeholder_trend():
    from agentic_options_reporter.models.schemas import TrendAssessment

    return TrendAssessment(direction="neutral", strength="weak", adx=0.0)


def _placeholder_volume():
    from agentic_options_reporter.models.schemas import VolumeAssessment

    return VolumeAssessment(relative_volume=0.0, flags=[])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/analyze/{symbol}", response_model=AnalysisResult)
def analyze(symbol: str, lookback_days: int = 365, expiration: str | None = None) -> AnalysisResult:
    try:
        return run_analysis(
            symbol=symbol,
            lookback_days=lookback_days,
            expiration=expiration,
            session_factory=_session_factory,
        )
    except MarketDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/runs/{run_id}", response_model=AnalysisResult)
def get_run(run_id: int) -> AnalysisResult:
    with _session_factory() as session:
        run = session.get(AnalysisRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        return _to_analysis_result(run)


@app.get("/runs", response_model=list[AnalysisRunSummary])
def list_runs(symbol: str | None = None, limit: int = 20) -> list[AnalysisRunSummary]:
    with _session_factory() as session:
        query = session.query(AnalysisRun)
        if symbol:
            query = query.filter(AnalysisRun.symbol == symbol)
        runs = query.order_by(AnalysisRun.generated_at.desc()).limit(limit).all()
        return [
            AnalysisRunSummary(
                run_id=run.id,
                symbol=run.symbol,
                generated_at=run.generated_at,
                recommendation_action=run.recommendation.action if run.recommendation else "AVOID",
                recommendation_confidence=(
                    run.recommendation.confidence if run.recommendation else 0.0
                ),
            )
            for run in runs
        ]
