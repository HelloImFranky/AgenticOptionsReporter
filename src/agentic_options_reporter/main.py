"""FastAPI surface. Contract is authoritative in specs/api.yaml."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from agentic_options_reporter.config import get_settings
from agentic_options_reporter.data.financial import (
    FinancialProvider,
    FinancialProviderError,
    build_financial_provider,
)
from agentic_options_reporter.data.macro import (
    MacroProvider,
    MacroProviderError,
    build_macro_provider,
)
from agentic_options_reporter.data.market_data import MarketDataError
from agentic_options_reporter.data.news import (
    NewsProvider,
    NewsProviderError,
    build_news_provider,
)
from agentic_options_reporter.models.db import AgentThesisRow, AnalysisRun
from agentic_options_reporter.models.schemas import (
    AgentThesisResult,
    AnalysisResult,
    AnalysisRunSummary,
    FinancialResearchFinding,
    IndicatorSnapshot,
    InvestmentThesis,
    MacroResearchFinding,
    NewsResearchFinding,
    QuantInterpretation,
    Recommendation,
    RiskAssessment,
    ScoredCandidate,
    StrategySuggestion,
    SupportResistanceLevel,
    ThesisGenerationRequest,
    TrendAssessment,
    VolumeAssessment,
)
from agentic_options_reporter.persistence import (
    delete_thesis,
    make_session_factory,
    persist_thesis,
)
from agentic_options_reporter.thesis.llm_client import LlmError, build_llm_client
from agentic_options_reporter.thesis.orchestrator import run_thesis_pipeline
from agentic_options_reporter.thesis.parsing import ThesisGenerationError
from agentic_options_reporter.workflow import run_analysis

app = FastAPI(title="AgenticOptionsReporter", version="0.1.0")

_settings = get_settings()
_session_factory = make_session_factory(_settings.database_url)


def _to_analysis_result(run: AnalysisRun) -> AnalysisResult:
    indicators = IndicatorSnapshot.model_validate(run.indicator_snapshot, from_attributes=True)
    trend = TrendAssessment.model_validate(run.trend_assessment, from_attributes=True)
    volume = VolumeAssessment.model_validate(run.volume_assessment, from_attributes=True)
    levels = [
        SupportResistanceLevel.model_validate(row, from_attributes=True)
        for row in run.support_resistance_levels
    ]
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
        trend=trend,
        volume=volume,
        support_resistance=levels,
        candidates=candidates,
        recommendation=recommendation,
    )


def _to_thesis_result(row: AgentThesisRow) -> AgentThesisResult:
    financial = (
        FinancialResearchFinding(
            company_health=row.financial_company_health,
            growth=row.financial_growth,
            profitability=row.financial_profitability,
            cash_flow=row.financial_cash_flow,
            analyst_consensus=row.financial_analyst_consensus or "",
            narrative=row.financial_narrative or "",
        )
        if row.financial_company_health is not None
        else None
    )
    news = (
        NewsResearchFinding(
            sentiment=row.news_sentiment,
            summary=row.news_summary or "",
            catalysts=row.news_catalysts or [],
            risks=row.news_risks or [],
        )
        if row.news_sentiment is not None
        else None
    )
    macro = (
        MacroResearchFinding(
            regime=row.macro_regime, outlook=row.macro_outlook or "", summary=row.macro_summary or ""
        )
        if row.macro_regime is not None
        else None
    )
    risk = (
        RiskAssessment(
            risk_level=row.risk_level,
            concerns=row.risk_concerns or [],
            position_sizing_note=row.risk_position_sizing_note or "",
        )
        if row.risk_level is not None
        else None
    )
    strategy = (
        StrategySuggestion(strategy=row.strategy, rationale=row.strategy_rationale or "")
        if row.strategy is not None
        else None
    )
    return AgentThesisResult(
        run_id=row.run_id,
        generated_at=row.generated_at,
        quant_interpretation=QuantInterpretation(
            narrative=row.quant_narrative,
            key_factors=row.quant_key_factors,
            score_breakdown=row.quant_score_breakdown,
            overall_score=row.quant_overall_score,
        ),
        financial_research=financial,
        news_research=news,
        macro_research=macro,
        risk_assessment=risk,
        strategy_suggestion=strategy,
        investment_thesis=InvestmentThesis(thesis=row.thesis, consensus=row.consensus),
        pipeline_warnings=row.pipeline_warnings or [],
    )


def _optional_financial_provider() -> FinancialProvider | None:
    try:
        return build_financial_provider()
    except FinancialProviderError:
        return None


def _optional_news_provider() -> NewsProvider | None:
    try:
        return build_news_provider()
    except NewsProviderError:
        return None


def _optional_macro_provider() -> MacroProvider | None:
    try:
        return build_macro_provider()
    except MacroProviderError:
        return None


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


@app.post("/runs/{run_id}/thesis", response_model=AgentThesisResult)
def generate_thesis(
    run_id: int, request: ThesisGenerationRequest = ThesisGenerationRequest()
) -> AgentThesisResult:
    with _session_factory() as session:
        run = session.get(AnalysisRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        if run.agent_thesis is not None and not request.regenerate:
            raise HTTPException(
                status_code=409,
                detail=f"Run {run_id} already has a thesis; pass regenerate=true to replace it",
            )

        if request.provider.strip().lower() == "auto" and request.api_key:
            raise HTTPException(
                status_code=422,
                detail="api_key cannot be combined with provider='auto'; choose a specific "
                "provider to use a custom key.",
            )

        analysis_result = _to_analysis_result(run)

        # settings.llm_model is only meaningful for the default (anthropic)
        # provider; other providers use their own built-in default model.
        model = _settings.llm_model if request.provider == "anthropic" else None
        try:
            llm_client = build_llm_client(
                request.provider,
                api_key=request.api_key,
                model=model,
                max_tokens=_settings.llm_max_tokens,
            )
            thesis_result = run_thesis_pipeline(
                analysis_result,
                llm_client,
                financial_provider=_optional_financial_provider(),
                news_provider=_optional_news_provider(),
                macro_provider=_optional_macro_provider(),
            )
        except (
            LlmError,
            ThesisGenerationError,
            FinancialProviderError,
            NewsProviderError,
            MacroProviderError,
        ) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        if run.agent_thesis is not None:
            delete_thesis(session, run_id)
        persist_thesis(session, thesis_result)
        return thesis_result


@app.get("/runs/{run_id}/thesis", response_model=AgentThesisResult)
def get_thesis(run_id: int) -> AgentThesisResult:
    with _session_factory() as session:
        run = session.get(AnalysisRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        if run.agent_thesis is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} has no thesis generated yet")
        return _to_thesis_result(run.agent_thesis)
