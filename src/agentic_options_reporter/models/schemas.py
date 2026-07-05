"""Pydantic models shared across module boundaries.

Field names and shapes here must stay in sync with specs/api.yaml and
specs/database.yaml. See docs/architecture.md for the module map.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

OptionType = Literal["call", "put"]
TrendDirection = Literal["bullish", "bearish", "neutral"]
TrendStrength = Literal["weak", "moderate", "strong"]
RecommendationAction = Literal["STRONG_BUY", "BUY", "HOLD", "AVOID"]


class Bar(BaseModel):
    """A single OHLCV bar."""

    dt: date
    open: float
    high: float
    low: float
    close: float
    volume: float


class PriceHistory(BaseModel):
    symbol: str
    bars: list[Bar]

    def to_dataframe(self) -> pd.DataFrame:
        if not self.bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame([b.model_dump() for b in self.bars])
        df["dt"] = pd.to_datetime(df["dt"])
        df = df.set_index("dt").sort_index()
        return df


class OptionContract(BaseModel):
    contract_symbol: str
    option_type: OptionType
    strike: float
    expiration: date
    bid: float
    ask: float
    last_price: float
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float | None = None
    in_the_money: bool = False

    @property
    def mid_price(self) -> float:
        if self.bid <= 0 and self.ask <= 0:
            return self.last_price
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        mid = self.mid_price
        if mid <= 0:
            return float("inf")
        return (self.ask - self.bid) / mid


class OptionChain(BaseModel):
    symbol: str
    underlying_price: float
    as_of: datetime
    contracts: list[OptionContract]


class IndicatorSnapshot(BaseModel):
    sma_20: float
    sma_50: float
    sma_200: float | None = None
    ema_12: float
    ema_26: float
    adx_14: float
    rsi_14: float
    macd: float
    macd_signal: float
    macd_histogram: float
    stoch_k: float
    stoch_d: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    atr_14: float
    obv: float
    volume_sma_20: float


class TrendAssessment(BaseModel):
    direction: TrendDirection
    strength: TrendStrength
    adx: float


class VolumeAssessment(BaseModel):
    relative_volume: float
    flags: list[str]


class SupportResistanceLevel(BaseModel):
    price: float
    level_type: Literal["support", "resistance"]
    touches: int
    last_touch_index: int


class Greeks(BaseModel):
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class EvaluatedContract(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    contract: OptionContract
    greeks: Greeks
    liquidity_ok: bool
    mid_price: float
    spread_pct: float
    days_to_expiration: int
    underlying_price: float
    implied_volatility: float


class RiskProfile(BaseModel):
    contract_symbol: str
    max_loss: float
    max_gain: float | None
    breakeven: float
    reward_risk_ratio: float | None
    probability_of_profit: float


class ScoredCandidate(BaseModel):
    contract_symbol: str
    option_type: OptionType
    strike: float
    expiration: date
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    max_loss: float
    max_gain: float | None
    breakeven: float
    reward_risk_ratio: float | None
    probability_of_profit: float
    score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class Recommendation(BaseModel):
    action: RecommendationAction
    contract_symbol: str | None
    confidence: float
    rationale: str


class AnalysisResult(BaseModel):
    symbol: str
    run_id: int
    generated_at: datetime
    indicators: IndicatorSnapshot
    trend: TrendAssessment
    volume: VolumeAssessment
    support_resistance: list[SupportResistanceLevel]
    candidates: list[ScoredCandidate]
    recommendation: Recommendation


class AnalysisRunSummary(BaseModel):
    run_id: int
    symbol: str
    generated_at: datetime
    recommendation_action: str
    recommendation_confidence: float


# ---------------------------------------------------------------------------
# Provider-normalized domain models (specs/providers.yaml). Agents consume
# these, never a provider's raw API response shape, so a provider can be
# swapped without touching agent code.
# ---------------------------------------------------------------------------


class NewsArticle(BaseModel):
    headline: str
    source: str
    url: str
    published_at: datetime
    summary: str = ""


class CompanyProfile(BaseModel):
    ticker: str
    name: str
    sector: str = ""
    industry: str = ""
    market_cap: float | None = None
    description: str = ""


class FinancialStatementSummary(BaseModel):
    ticker: str
    period: str   # e.g. "FY2025" or "Q3 2025"
    revenue: float | None = None
    net_income: float | None = None
    operating_cash_flow: float | None = None
    free_cash_flow: float | None = None


class FinancialRatios(BaseModel):
    ticker: str
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    return_on_equity: float | None = None
    gross_margin: float | None = None
    net_margin: float | None = None


class AnalystEstimates(BaseModel):
    ticker: str
    consensus_rating: str = "N/A"   # e.g. "buy" | "hold" | "sell", provider vocabulary varies
    price_target_mean: float | None = None
    price_target_high: float | None = None
    price_target_low: float | None = None
    num_analysts: int = 0


class MacroObservation(BaseModel):
    """One normalized macro data point, whatever source served it — the
    capability-based macro layer's single output type (see
    data/macro/metrics.py and specs/providers.yaml). `source` records
    which provider actually answered; `yoy_change_pct` is a derived
    year-over-year change where meaningful (prices/output), null for
    point-in-time rates or when the lookback observation is missing."""

    metric_id: str          # e.g. "policy_rate", "cpi", "gdp"
    label: str              # human label from the metric registry
    value: float
    unit: str               # "percent" | "index" | "usd"
    as_of: date
    source: str             # provider name that served this observation
    yoy_change_pct: float | None = None


class SecFiling(BaseModel):
    ticker: str
    form_type: str   # "10-K" | "10-Q" | "8-K" | ...
    filed_at: date
    url: str
    accession_number: str


# ---------------------------------------------------------------------------
# Investment-thesis agent pipeline (specs/agents.yaml). These are the only
# models an LLM ever authors fields of; score_breakdown/overall_score on
# QuantInterpretation are pass-throughs from the already-computed
# ScoredCandidate, never LLM-derived. analyst_consensus on
# FinancialResearchFinding is likewise a pass-through from AnalystEstimates.
# ---------------------------------------------------------------------------

# -- Lenient enums for LLM-authored fields ---------------------------------
# LLMs occasionally slip an off-vocabulary value — or a whole sentence — into
# an enum field. Rather than fail the entire finding (and, before this,
# 502 the whole thesis), coerce an unknown value to a safe neutral default so
# only that one field degrades. The coercion rides on the TYPE (a
# BeforeValidator on an Annotated Literal), so every model using it is
# resilient with no per-model boilerplate. Only LLM-authored enums are made
# lenient; engine-computed enums (OptionType, TrendDirection, …) stay strict.
# See specs/agents.yaml: llm_output_resilience.


def _lenient_enum(*allowed: str, default: str) -> BeforeValidator:
    valid = frozenset(allowed)

    def _coerce(value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in valid:
            return value.strip().lower()
        return default

    return BeforeValidator(_coerce)


RiskLevel = Annotated[
    Literal["low", "medium", "high"], _lenient_enum("low", "medium", "high", default="medium")
]
Consensus = Annotated[
    Literal["bullish", "bearish", "neutral", "mixed"],
    _lenient_enum("bullish", "bearish", "neutral", "mixed", default="neutral"),
]
CompanyHealth = Annotated[
    Literal["strong", "stable", "weak"],
    _lenient_enum("strong", "stable", "weak", default="stable"),
]
GrowthTrend = Annotated[
    Literal["accelerating", "steady", "decelerating"],
    _lenient_enum("accelerating", "steady", "decelerating", default="steady"),
]
ProfitabilityLevel = Annotated[
    Literal["high", "moderate", "low"],
    _lenient_enum("high", "moderate", "low", default="moderate"),
]
CashFlowState = Annotated[
    Literal["positive", "neutral", "negative"],
    _lenient_enum("positive", "neutral", "negative", default="neutral"),
]
NewsSentiment = Annotated[
    Literal["bullish", "bearish", "neutral"],
    _lenient_enum("bullish", "bearish", "neutral", default="neutral"),
]
MacroRegime = Annotated[
    Literal["risk_on", "risk_off", "neutral"],
    _lenient_enum("risk_on", "risk_off", "neutral", default="neutral"),
]
CatalystCategory = Annotated[
    Literal["earnings", "filing", "news", "macro", "corporate_action", "other"],
    _lenient_enum("earnings", "filing", "news", "macro", "corporate_action", "other", default="other"),
]
# When a catalyst sits relative to now: recent = already occurred,
# near_term = expected within weeks, long_term = months out, unknown =
# no datable timing in the source material.
CatalystHorizon = Annotated[
    Literal["recent", "near_term", "long_term", "unknown"],
    _lenient_enum("recent", "near_term", "long_term", "unknown", default="unknown"),
]
CatalystDirection = Annotated[
    Literal["bullish", "bearish", "uncertain"],
    _lenient_enum("bullish", "bearish", "uncertain", default="uncertain"),
]


class QuantInterpretation(BaseModel):
    narrative: str
    key_factors: list[str]
    score_breakdown: dict[str, float]
    overall_score: float


class FinancialResearchFinding(BaseModel):
    company_health: CompanyHealth
    growth: GrowthTrend
    profitability: ProfitabilityLevel
    cash_flow: CashFlowState
    analyst_consensus: str
    narrative: str


class NewsResearchFinding(BaseModel):
    sentiment: NewsSentiment
    summary: str
    catalysts: list[str]
    risks: list[str]


class MacroResearchFinding(BaseModel):
    regime: MacroRegime
    outlook: str
    summary: str


class CatalystItem(BaseModel):
    """One discrete, dateable event that could move the stock, extracted
    from provider material (news article, SEC filing, or macro release).
    All fields are the catalyst agent's qualitative reading of the given
    material — it never invents an event not grounded in an input."""

    title: str
    category: CatalystCategory
    horizon: CatalystHorizon
    direction: CatalystDirection
    detail: str = ""


class CatalystFinding(BaseModel):
    catalysts: list[CatalystItem]
    summary: str
    net_bias: Consensus
    # How many individual catalysts were skipped because they couldn't be
    # validated even after lenient coercion (e.g. a missing title). Transient
    # run metadata, not part of the finding's data — excluded from
    # serialization/persistence; the orchestrator turns a non-zero count into
    # a pipeline_warning so a silent drop is still surfaced at the end of the
    # run (see specs/agents.yaml llm_output_resilience).
    dropped_count: int = Field(default=0, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _drop_invalid_catalysts(cls, data: Any) -> Any:
        """Skip any individual catalyst that still can't be validated (e.g. a
        missing title) rather than failing the whole finding, and record how
        many were dropped. Enum slips are already repaired by the lenient
        field types above, so this only drops genuinely unusable items."""
        if not isinstance(data, dict):
            return data
        raw = data.get("catalysts")
        if not isinstance(raw, list):
            return data
        valid: list[CatalystItem] = []
        for item in raw:
            try:
                valid.append(CatalystItem.model_validate(item))
            except ValidationError:
                continue
        return {**data, "catalysts": valid, "dropped_count": len(raw) - len(valid)}


class RiskAssessment(BaseModel):
    risk_level: RiskLevel
    concerns: list[str]
    position_sizing_note: str


class StrategySuggestion(BaseModel):
    strategy: str
    rationale: str


class InvestmentThesis(BaseModel):
    thesis: str
    consensus: Consensus


class AgentThesisResult(BaseModel):
    run_id: int
    generated_at: datetime
    quant_interpretation: QuantInterpretation
    financial_research: FinancialResearchFinding | None = None
    news_research: NewsResearchFinding | None = None
    macro_research: MacroResearchFinding | None = None
    catalyst_research: CatalystFinding | None = None
    risk_assessment: RiskAssessment | None
    strategy_suggestion: StrategySuggestion | None
    investment_thesis: InvestmentThesis
    # Non-fatal problems hit mid-pipeline (e.g. a configured research
    # provider rate-limited during the run). Each entry is prefixed with
    # the agent it affected, e.g. "news_research: ...". The affected
    # finding is null; the rest of the pipeline still completed.
    pipeline_warnings: list[str] = []


AgentPhase = Literal["started", "completed", "skipped", "failed"]


class AgentExchange(BaseModel):
    """The raw LLM call an agent made — the 'under the hood' view: exactly
    what was sent to the model and what came back, before parsing."""

    system_prompt: str
    user_prompt: str
    raw_response: str


class AgentEvent(BaseModel):
    """One live event emitted as the thesis pipeline runs a single agent
    (see thesis/orchestrator.py: on_event). Streamed to the client for a
    live per-agent view; transient — not persisted with the run."""

    agent: str          # the agent's stable id, e.g. "news_research"
    phase: AgentPhase
    at: datetime
    exchange: AgentExchange | None = None      # the raw prompt/response, when the agent called the LLM
    output: dict[str, Any] | None = None       # the parsed finding (model_dump), on `completed`
    detail: str | None = None                  # skip reason or failure message


class ThesisGenerationRequest(BaseModel):
    """Request body for POST /runs/{run_id}/thesis.

    `provider="auto"` (the default) builds an `LlmRouter` across every
    provider with a configured API key and fails over between them; a
    named provider bypasses the router for a single, explicit choice.
    `api_key`, if supplied, is used only to construct the LlmClient for
    this one request; it is never logged or persisted (see
    thesis.llm_client.build_llm_client and main.generate_thesis).
    """

    provider: str = "auto"
    api_key: str | None = None
    regenerate: bool = False
