"""Opportunity scoring and recommendation generation.

Weights and thresholds mirror specs/scoring.yaml exactly; update both
together if the model changes.
"""

from __future__ import annotations

from agentic_options_reporter.models.schemas import (
    EvaluatedContract,
    Recommendation,
    RecommendationAction,
    RiskProfile,
    ScoredCandidate,
    SupportResistanceLevel,
    TrendAssessment,
    VolumeAssessment,
)

WEIGHTS = {
    "trend_alignment": 0.30,
    "volume_confirmation": 0.15,
    "support_resistance_proximity": 0.15,
    "liquidity": 0.20,
    "risk_reward": 0.20,
}

_SR_MAX_DISTANCE_PCT = 0.05
_LIQUIDITY_OI_TARGET = 500
_LIQUIDITY_MAX_SPREAD_PCT = 0.10
_RISK_REWARD_FLOOR = 0.5
_RISK_REWARD_CEIL = 2.0

ACTION_THRESHOLDS: list[tuple[float, RecommendationAction]] = [
    (75, "STRONG_BUY"),
    (60, "BUY"),
    (40, "HOLD"),
    (0, "AVOID"),
]


def _bias(option_type: str) -> str:
    return "bullish" if option_type == "call" else "bearish"


def _trend_alignment(option_type: str, trend: TrendAssessment) -> float:
    bias = _bias(option_type)
    if trend.direction == "neutral":
        return 0.5
    if trend.direction == bias:
        return 1.0 if trend.strength != "weak" else 0.5
    return 0.0


def _volume_confirmation(option_type: str, volume: VolumeAssessment) -> float:
    bias = _bias(option_type)
    if "high_volume" in volume.flags:
        return 1.0
    if bias == "bullish" and "bearish_divergence" in volume.flags:
        return 0.0
    if bias == "bearish" and "bullish_divergence" in volume.flags:
        return 0.0
    if "low_volume" in volume.flags:
        return 0.0
    return 0.5


def _support_resistance_proximity(
    option_type: str, underlying_price: float, levels: list[SupportResistanceLevel]
) -> float:
    relevant_type = "support" if option_type == "call" else "resistance"
    relevant = [lvl for lvl in levels if lvl.level_type == relevant_type]
    if not relevant or underlying_price <= 0:
        return 0.0
    nearest = min(relevant, key=lambda lvl: abs(lvl.price - underlying_price))
    distance_pct = abs(nearest.price - underlying_price) / underlying_price
    return max(0.0, 1.0 - distance_pct / _SR_MAX_DISTANCE_PCT)


def _liquidity(candidate: EvaluatedContract) -> float:
    oi_score = min(candidate.contract.open_interest / _LIQUIDITY_OI_TARGET, 1.0)
    spread_score = max(0.0, 1 - candidate.spread_pct / _LIQUIDITY_MAX_SPREAD_PCT)
    return 0.5 * oi_score + 0.5 * spread_score


def _risk_reward(reward_risk_ratio: float | None) -> float:
    if reward_risk_ratio is None:
        return 1.0  # unlimited upside (e.g. long call) with defined risk
    if reward_risk_ratio >= _RISK_REWARD_CEIL:
        return 1.0
    if reward_risk_ratio <= _RISK_REWARD_FLOOR:
        return 0.0
    return (reward_risk_ratio - _RISK_REWARD_FLOOR) / (_RISK_REWARD_CEIL - _RISK_REWARD_FLOOR)


def score_candidates(
    evaluated_contracts: list[EvaluatedContract],
    risk_profiles: list[RiskProfile],
    trend: TrendAssessment,
    volume: VolumeAssessment,
    levels: list[SupportResistanceLevel],
) -> list[ScoredCandidate]:
    risk_by_symbol = {rp.contract_symbol: rp for rp in risk_profiles}

    scored: list[ScoredCandidate] = []
    for candidate in evaluated_contracts:
        if not candidate.liquidity_ok:
            continue
        risk = risk_by_symbol.get(candidate.contract.contract_symbol)
        if risk is None:
            continue

        breakdown = {
            "trend_alignment": _trend_alignment(candidate.contract.option_type, trend),
            "volume_confirmation": _volume_confirmation(candidate.contract.option_type, volume),
            "support_resistance_proximity": _support_resistance_proximity(
                candidate.contract.option_type, candidate.underlying_price, levels
            ),
            "liquidity": _liquidity(candidate),
            "risk_reward": _risk_reward(risk.reward_risk_ratio),
        }
        score = 100 * sum(WEIGHTS[name] * value for name, value in breakdown.items())

        scored.append(
            ScoredCandidate(
                contract_symbol=candidate.contract.contract_symbol,
                option_type=candidate.contract.option_type,
                strike=candidate.contract.strike,
                expiration=candidate.contract.expiration,
                delta=candidate.greeks.delta,
                gamma=candidate.greeks.gamma,
                theta=candidate.greeks.theta,
                vega=candidate.greeks.vega,
                rho=candidate.greeks.rho,
                max_loss=risk.max_loss,
                max_gain=risk.max_gain,
                breakeven=risk.breakeven,
                reward_risk_ratio=risk.reward_risk_ratio,
                probability_of_profit=risk.probability_of_profit,
                score=score,
                score_breakdown=breakdown,
            )
        )

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored


def _action_for_score(score: float) -> RecommendationAction:
    for floor, action in ACTION_THRESHOLDS:
        if score >= floor:
            return action
    return "AVOID"


def build_recommendation(candidates: list[ScoredCandidate]) -> Recommendation:
    if not candidates:
        return Recommendation(
            action="AVOID",
            contract_symbol=None,
            confidence=0.0,
            rationale="No liquid, scoreable candidates were found in the option chain.",
        )

    top = max(candidates, key=lambda c: c.score)
    factors = ", ".join(f"{name}={value:.2f}" for name, value in top.score_breakdown.items())
    rationale = f"{top.contract_symbol} scored {top.score:.1f}/100 ({factors})."

    return Recommendation(
        action=_action_for_score(top.score),
        contract_symbol=top.contract_symbol,
        confidence=round(top.score / 100, 4),
        rationale=rationale,
    )
