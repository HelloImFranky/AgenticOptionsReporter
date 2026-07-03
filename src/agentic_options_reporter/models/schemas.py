"""Pydantic models shared across module boundaries.

Field names and shapes here must stay in sync with specs/api.yaml and
specs/database.yaml. See docs/architecture.md for the module map.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

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
# Investment-thesis agent pipeline (specs/agents.yaml). These are the only
# models an LLM ever authors fields of; score_breakdown/overall_score on
# QuantInterpretation are pass-throughs from the already-computed
# ScoredCandidate, never LLM-derived.
# ---------------------------------------------------------------------------

RiskLevel = Literal["low", "medium", "high"]
Consensus = Literal["bullish", "bearish", "neutral", "mixed"]


class QuantInterpretation(BaseModel):
    narrative: str
    key_factors: list[str]
    score_breakdown: dict[str, float]
    overall_score: float


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
    risk_assessment: RiskAssessment | None
    strategy_suggestion: StrategySuggestion | None
    investment_thesis: InvestmentThesis
