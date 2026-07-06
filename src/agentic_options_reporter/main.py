"""FastAPI surface. Contract is authoritative in specs/api.yaml."""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

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
from agentic_options_reporter.data.market_data import (
    MarketDataError,
    build_market_data_provider,
)
from agentic_options_reporter.data.news import (
    NewsProvider,
    NewsProviderError,
    build_news_provider,
)
from agentic_options_reporter.data.sec_provider import (
    SECProvider,
    SecProviderError,
    build_sec_provider,
)
from agentic_options_reporter.models.db import AgentThesisRow, AnalysisRun
from agentic_options_reporter.models.schemas import (
    AgentThesisResult,
    AnalysisResult,
    AnalysisRunSummary,
    CatalystFinding,
    CatalystItem,
    DomainScore,
    FinancialResearchFinding,
    FundamentalsSnapshot,
    IndicatorSnapshot,
    InvestmentThesis,
    MacroResearchFinding,
    NewsResearchFinding,
    QuantInterpretation,
    Recommendation,
    RelativeStrengthFinding,
    RiskAssessment,
    ScoredCandidate,
    StatisticalEdgeFinding,
    StrategySuggestion,
    SupportResistanceLevel,
    ThesisGenerationRequest,
    TradeQualityScore,
    TrendAssessment,
    VolumeAssessment,
    WeightingProfileId,
)
from agentic_options_reporter.persistence import (
    delete_thesis,
    make_session_factory,
    persist_thesis,
)
from agentic_options_reporter.thesis.llm_client import LlmClient, LlmError, build_llm_client
from agentic_options_reporter.thesis.orchestrator import run_thesis_pipeline
from agentic_options_reporter.thesis.parsing import ThesisGenerationError
from agentic_options_reporter.thesis.streaming import run_thesis_streaming
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
    # Fundamentals were persisted as JSON; rehydrate them (null on runs that
    # predate the column or where no provider returned any).
    fundamentals = (
        FundamentalsSnapshot.model_validate(run.fundamentals)
        if run.fundamentals is not None
        else None
    )
    trade_quality = (
        TradeQualityScore.model_validate(run.trade_quality_score, from_attributes=True)
        if run.trade_quality_score is not None
        else None
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
        trade_quality=trade_quality,
        weighting_profile=run.weighting_profile,
        fundamentals=fundamentals,
        data_warnings=run.data_warnings or [],
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
            domain_score=DomainScore.model_validate(row.fundamental_domain_score),
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
            domain_score=DomainScore.model_validate(row.sentiment_domain_score),
        )
        if row.news_sentiment is not None
        else None
    )
    macro = (
        MacroResearchFinding(
            regime=row.macro_regime,
            outlook=row.macro_outlook or "",
            summary=row.macro_summary or "",
            domain_score=DomainScore.model_validate(row.macro_domain_score),
        )
        if row.macro_regime is not None
        else None
    )
    catalyst = (
        CatalystFinding(
            net_bias=row.catalyst_net_bias,
            summary=row.catalyst_summary or "",
            catalysts=[CatalystItem(**item) for item in (row.catalyst_items or [])],
        )
        if row.catalyst_net_bias is not None
        else None
    )
    relative_strength = (
        RelativeStrengthFinding(
            narrative=row.relative_strength_narrative or "",
            domain_score=DomainScore.model_validate(row.relative_strength_domain_score),
        )
        if row.relative_strength_narrative is not None
        else None
    )
    statistical_edge = (
        StatisticalEdgeFinding(
            narrative=row.statistical_edge_narrative or "",
            domain_score=DomainScore.model_validate(row.statistical_edge_domain_score),
        )
        if row.statistical_edge_narrative is not None
        else None
    )
    risk = (
        RiskAssessment(
            risk_level=row.risk_level,
            concerns=row.risk_concerns or [],
            position_sizing_note=row.risk_position_sizing_note or "",
            domain_score=DomainScore.model_validate(row.risk_domain_score),
        )
        if row.risk_level is not None
        else None
    )
    strategy = (
        StrategySuggestion(
            strategy=row.strategy,
            rationale=row.strategy_rationale or "",
            domain_score=DomainScore.model_validate(row.liquidity_domain_score),
        )
        if row.strategy is not None
        else None
    )
    return AgentThesisResult(
        run_id=row.run_id,
        generated_at=row.generated_at,
        quant_interpretation=QuantInterpretation(
            narrative=row.quant_narrative,
            key_factors=row.quant_key_factors,
            quant_trade_quality=TradeQualityScore.model_validate(row.quant_trade_quality),
            technical_domain_score=DomainScore.model_validate(row.technical_domain_score),
        ),
        financial_research=financial,
        news_research=news,
        macro_research=macro,
        catalyst_research=catalyst,
        relative_strength_research=relative_strength,
        statistical_edge_research=statistical_edge,
        risk_assessment=risk,
        strategy_suggestion=strategy,
        investment_thesis=InvestmentThesis(thesis=row.thesis, consensus=row.consensus),
        agent_trade_quality=(
            TradeQualityScore.model_validate(row.agent_trade_quality)
            if row.agent_trade_quality is not None
            else None
        ),
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


def _optional_sec_provider() -> SECProvider | None:
    # SEC EDGAR is keyless, so this always builds — the catalyst agent
    # always has at least the filings stream available.
    try:
        return build_sec_provider()
    except SecProviderError:
        return None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/analyze/{symbol}", response_model=AnalysisResult)
def analyze(
    symbol: str,
    lookback_days: int = 365,
    expiration: str | None = None,
    weighting_profile: WeightingProfileId = "swing",
) -> AnalysisResult:
    try:
        return run_analysis(
            symbol=symbol,
            lookback_days=lookback_days,
            expiration=expiration,
            session_factory=_session_factory,
            weighting_profile=weighting_profile,
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


def _load_run_for_thesis(session, run_id: int, request: ThesisGenerationRequest) -> AnalysisResult:
    """Shared guards for thesis generation (blocking + streaming): 404 if
    the run is missing, 409 if a thesis already exists without regenerate,
    422 if provider='auto' is combined with a custom key. Returns the
    AnalysisResult to run the pipeline over."""
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
    return _to_analysis_result(run)


def _build_thesis_llm_client(request: ThesisGenerationRequest) -> LlmClient:
    # settings.llm_model is only meaningful for the default (anthropic)
    # provider; other providers use their own built-in default model.
    model = _settings.llm_model if request.provider == "anthropic" else None
    return build_llm_client(
        request.provider, api_key=request.api_key, model=model, max_tokens=_settings.llm_max_tokens
    )


def _persist_generated_thesis(run_id: int, thesis_result: AgentThesisResult) -> None:
    with _session_factory() as session:
        run = session.get(AnalysisRun, run_id)
        if run is not None and run.agent_thesis is not None:
            delete_thesis(session, run_id)
        persist_thesis(session, thesis_result)


@app.post("/runs/{run_id}/thesis", response_model=AgentThesisResult)
def generate_thesis(
    run_id: int, request: ThesisGenerationRequest = ThesisGenerationRequest()
) -> AgentThesisResult:
    with _session_factory() as session:
        analysis_result = _load_run_for_thesis(session, run_id, request)

    try:
        thesis_result = run_thesis_pipeline(
            analysis_result,
            _build_thesis_llm_client(request),
            market_data_provider=build_market_data_provider(),
            financial_provider=_optional_financial_provider(),
            news_provider=_optional_news_provider(),
            macro_provider=_optional_macro_provider(),
            sec_provider=_optional_sec_provider(),
        )
    except (
        LlmError,
        ThesisGenerationError,
        FinancialProviderError,
        NewsProviderError,
        MacroProviderError,
        SecProviderError,
    ) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    _persist_generated_thesis(run_id, thesis_result)
    return thesis_result


def _sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/runs/{run_id}/thesis/stream")
def generate_thesis_stream(
    run_id: int, request: ThesisGenerationRequest = ThesisGenerationRequest()
) -> StreamingResponse:
    """Server-Sent Events variant of generate_thesis: emits one `agent`
    frame per agent as the pipeline runs (with its raw prompt/response and
    parsed output — the live 'under the hood' view), then a terminal
    `result` frame carrying the full AgentThesisResult (persisted, same as
    the blocking endpoint) or an `error` frame if a required agent failed.
    The 404/409/422 guards run before streaming starts, so they still
    surface as normal HTTP errors."""
    with _session_factory() as session:
        analysis_result = _load_run_for_thesis(session, run_id, request)

    # Build the client + providers up front so a bad provider/key fails as a
    # clean HTTP error rather than mid-stream.
    try:
        llm_client = _build_thesis_llm_client(request)
    except LlmError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    market_data_provider = build_market_data_provider()
    financial_provider = _optional_financial_provider()
    news_provider = _optional_news_provider()
    macro_provider = _optional_macro_provider()
    sec_provider = _optional_sec_provider()

    def _run_pipeline(on_event):
        return run_thesis_pipeline(
            analysis_result,
            llm_client,
            market_data_provider=market_data_provider,
            financial_provider=financial_provider,
            news_provider=news_provider,
            macro_provider=macro_provider,
            sec_provider=sec_provider,
            on_event=on_event,
        )

    def event_stream():
        for kind, payload in run_thesis_streaming(_run_pipeline):
            if kind == "event":
                yield _sse_frame("agent", payload.model_dump(mode="json"))
            elif kind == "result":
                _persist_generated_thesis(run_id, payload)
                yield _sse_frame("result", payload.model_dump(mode="json"))
            else:  # "error" — a required agent failed (LlmError/ThesisGenerationError)
                yield _sse_frame("error", {"detail": str(payload)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/runs/{run_id}/thesis", response_model=AgentThesisResult)
def get_thesis(run_id: int) -> AgentThesisResult:
    with _session_factory() as session:
        run = session.get(AnalysisRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        if run.agent_thesis is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} has no thesis generated yet")
        return _to_thesis_result(run.agent_thesis)
