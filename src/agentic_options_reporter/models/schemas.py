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

# ---------------------------------------------------------------------------
# Trade Quality Score (specs/scoring.yaml). Computed twice from the same
# taxonomy/composite engine: once deterministically ("quant", during
# /analyze) and once by the LLM agent pipeline ("agent", Agents tab).
# ---------------------------------------------------------------------------

DomainId = Literal[
    "technical",
    "risk",
    "liquidity",
    "fundamental",
    "macro",
    "sentiment",
    "relative_strength",
    "statistical_edge",
]
ScoreSource = Literal["quant", "agent"]
WeightingProfileId = Literal["day_trade", "swing", "long_term"]


def _clamp_0_100(value: Any) -> Any:
    """Defensive coercion for an LLM-authored 0-100 field. Extends the
    llm_output_resilience pattern (specs/agents.yaml _lenient_enum) to
    numbers: strips a trailing '%', treats a value in (0, 1] as an
    accidentally-fractional 0-1 score, then clamps to [0, 100]. A no-op for
    quant-authored values, which are already well-formed."""
    if isinstance(value, str):
        value = value.strip().rstrip("%")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 < number <= 1.0:
        number *= 100
    return max(0.0, min(100.0, number))


Score0to100 = Annotated[float, BeforeValidator(_clamp_0_100)]


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


class DomainFactor(BaseModel):
    """One named sub-factor inside a domain score, for drill-down
    explainability. Quant scorers always populate this (analysis/
    domain_scoring.py); agent scorers may leave it empty — an LLM isn't
    asked to invent a sub-factor weight table, only the domain-level
    score/confidence/evidence."""

    name: str
    value: float  # 0-1
    weight: float  # 0-1, within-domain weight
    detail: str = ""


class DomainScore(BaseModel):
    """One domain's contribution to a Trade Quality Score (specs/
    scoring.yaml). `source` records whether this came from the
    deterministic quant engine or an LLM agent's independent judgment;
    the composite engine (analysis/composite_score.py) treats both
    identically."""

    domain: DomainId
    score: Score0to100
    confidence: Score0to100
    evidence: list[str] = Field(default_factory=list)
    factors: list[DomainFactor] = Field(default_factory=list)
    source: ScoreSource
    generated_at: datetime


class TradeQualityScore(BaseModel):
    """The composite engine's output: a weighted blend of whichever
    DomainScores are present, coverage-discounted for confidence (see
    analysis/composite_score.py). Produced twice per run — once with
    source="quant" (during /analyze) and once with source="agent" (Agents
    tab) — from the same weighting profile, so the two are comparable."""

    contract_symbol: str | None
    domain_scores: dict[str, DomainScore]
    composite_score: float
    confidence: float
    recommendation_action: RecommendationAction
    weighting_profile: WeightingProfileId
    source: ScoreSource
    generated_at: datetime
    # Deterministic "led by X, held back by Y" bullets ranking the present
    # domains — part of the persisted, API-visible record, not just UI text.
    explainability: list[str] = Field(default_factory=list)


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
    open_interest: int = 0
    spread_pct: float = 0.0
    volume: int = 0
    max_loss: float
    max_gain: float | None
    breakeven: float
    reward_risk_ratio: float | None
    probability_of_profit: float
    score: float
    domain_scores: dict[str, DomainScore] = Field(default_factory=dict)


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
    # The quant-side Trade Quality Score for the recommended candidate; None
    # on AVOID / no liquid candidates, mirroring recommendation.contract_symbol.
    trade_quality: TradeQualityScore | None = None
    weighting_profile: WeightingProfileId = "swing"
    # Cross-provider fundamentals gathered alongside the technicals (merged
    # across every configured source) and persisted with the run, so a run
    # reloaded via /runs/{id} carries the same snapshot. None on runs where
    # no provider returned fundamentals (or that predate the feature).
    fundamentals: FundamentalsSnapshot | None = None
    # Non-fatal problems while gathering fundamentals (e.g. a provider rate
    # limited); [] when clean. The technical analysis always completes.
    data_warnings: list[str] = Field(default_factory=list)


class AnalysisRunSummary(BaseModel):
    run_id: int
    symbol: str
    generated_at: datetime
    recommendation_action: str
    recommendation_confidence: float


class PastRunOutcome(BaseModel):
    """One past AnalysisRun's recommendation, read back for the Statistical
    Edge domain scorer's historical hit-rate lookback (persistence.py
    fetch_recent_runs_for_symbol / analysis/statistical_edge.py). Derived
    fresh from existing rows each time; never persisted itself."""

    run_id: int
    generated_at: datetime
    action: RecommendationAction
    option_type: OptionType | None
    contract_symbol: str | None


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


class CompanyMetrics(BaseModel):
    """Current valuation and quality key-stats snapshot for a ticker.

    Most fields are provider facts (Finnhub /stock/metric, yfinance .info).
    The 1-week/1-month high/low are the exception — no provider exposes
    them, so /analyze DERIVES them from the daily price history it already
    fetches (see workflow._price_range_metrics); they stay null when no
    price history is available (e.g. metrics built inside the thesis
    pipeline). Every field is optional so a partial provider still
    contributes."""

    ticker: str
    market_cap: float | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_book: float | None = None
    price_to_sales: float | None = None
    beta: float | None = None
    dividend_yield: float | None = None      # as a fraction, e.g. 0.012 for 1.2%
    week1_high: float | None = None          # derived from price history (~5 trading days)
    week1_low: float | None = None
    month1_high: float | None = None         # derived from price history (~21 trading days)
    month1_low: float | None = None
    week52_high: float | None = None
    week52_low: float | None = None
    gross_margin: float | None = None        # fraction
    operating_margin: float | None = None    # fraction
    profit_margin: float | None = None       # fraction
    revenue_growth: float | None = None      # YoY fraction
    earnings_growth: float | None = None     # YoY fraction


class EarningsSurprise(BaseModel):
    """One past reporting period's actual EPS vs. the consensus estimate."""

    period: str                     # e.g. "2026-03-31" or "Q1 2026"
    actual_eps: float | None = None
    estimate_eps: float | None = None
    surprise: float | None = None          # actual - estimate
    surprise_percent: float | None = None  # (actual - estimate) / |estimate|, fraction


class EarningsHistory(BaseModel):
    ticker: str
    surprises: list[EarningsSurprise] = Field(default_factory=list)


class EarningsCalendar(BaseModel):
    """The next scheduled earnings report and its consensus estimate."""

    ticker: str
    next_date: date | None = None
    eps_estimate: float | None = None
    revenue_estimate: float | None = None


class InsiderTransaction(BaseModel):
    name: str
    relationship: str = ""          # role/title where the provider gives one
    transaction_type: str = ""      # e.g. "buy" | "sell" | raw provider code
    shares: float | None = None
    value: float | None = None      # transaction value where available
    filed_at: date | None = None


class InsiderActivity(BaseModel):
    ticker: str
    transactions: list[InsiderTransaction] = Field(default_factory=list)
    net_shares: float | None = None   # sum of signed shares across transactions, when derivable


class FundamentalsSnapshot(BaseModel):
    """Everything the fundamentals layer could gather for a ticker, merged
    across every configured provider. Surfaced by /analyze; each field is
    optional so partial provider coverage still yields a useful snapshot."""

    ticker: str
    profile: CompanyProfile | None = None
    statements: FinancialStatementSummary | None = None
    ratios: FinancialRatios | None = None
    estimates: AnalystEstimates | None = None
    metrics: CompanyMetrics | None = None
    earnings_history: EarningsHistory | None = None
    earnings_calendar: EarningsCalendar | None = None
    insider_activity: InsiderActivity | None = None


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
# Investment-thesis agent pipeline (specs/agents.yaml). quant_trade_quality
# on QuantInterpretation is a pass-through from the already-computed
# ScoredCandidate.domain_scores, never LLM-derived; analyst_consensus on
# FinancialResearchFinding is likewise a pass-through from AnalystEstimates.
# Since phase 3, most other models here ALSO carry an LLM-authored
# `domain_score: DomainScore` field (specs/scoring.yaml) — the agent's own
# independent judgment for one Trade Quality Score domain.
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
    # Pass-through of the quant engine's own composite score, verbatim —
    # unchanged contract, see specs/agents.yaml.
    quant_trade_quality: TradeQualityScore
    # This agent's OWN, independently-authored Technical domain score — a
    # judgment call over the same indicators, not a copy of any quant
    # sub-factor (see specs/agents.yaml phase_3 scoring relaxation).
    technical_domain_score: DomainScore


class FinancialResearchFinding(BaseModel):
    company_health: CompanyHealth
    growth: GrowthTrend
    profitability: ProfitabilityLevel
    cash_flow: CashFlowState
    analyst_consensus: str
    narrative: str
    domain_score: DomainScore  # Fundamental domain, this agent's own judgment


class NewsResearchFinding(BaseModel):
    sentiment: NewsSentiment
    summary: str
    catalysts: list[str]
    risks: list[str]
    domain_score: DomainScore  # Sentiment domain, this agent's own judgment


class MacroResearchFinding(BaseModel):
    regime: MacroRegime
    outlook: str
    summary: str
    domain_score: DomainScore  # Macro domain, this agent's own judgment


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
    domain_score: DomainScore  # Risk domain, this agent's own judgment


class StrategySuggestion(BaseModel):
    strategy: str
    rationale: str
    domain_score: DomainScore  # Liquidity domain, this agent's own judgment


class RelativeStrengthFinding(BaseModel):
    """New in phase 3 (specs/agents.yaml): the agent-side counterpart to
    the quant Relative Strength domain scorer (analysis/domain_scoring.py).
    Reasons over the same symbol-vs-benchmark return facts, fetched by the
    orchestrator via MarketDataProvider."""

    narrative: str
    domain_score: DomainScore


class StatisticalEdgeFinding(BaseModel):
    """New in phase 3: the agent-side counterpart to the quant Statistical
    Edge domain scorer (analysis/statistical_edge.py). Reasons over the
    quant-computed win-rate/expectancy/Monte-Carlo numbers plus qualitative
    pattern context — the one place an agent's prompt includes a quant
    DomainScore as input, mirroring quant_interpreter's existing precedent
    of receiving quant numbers as context."""

    narrative: str
    domain_score: DomainScore


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
    relative_strength_research: RelativeStrengthFinding | None = None
    statistical_edge_research: StatisticalEdgeFinding | None = None
    risk_assessment: RiskAssessment | None
    strategy_suggestion: StrategySuggestion | None
    investment_thesis: InvestmentThesis
    # The agent-side Trade Quality Score: the same composite engine used by
    # the quant path (analysis/composite_score.py), fed whichever agent
    # DomainScores this run produced. None only if literally none did
    # (e.g. every research provider unconfigured and no_candidate_short_circuit).
    agent_trade_quality: TradeQualityScore | None = None
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
