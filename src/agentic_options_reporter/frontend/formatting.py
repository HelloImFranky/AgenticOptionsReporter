"""Pure functions turning ApiClient JSON payloads into display-ready data.

Kept separate from app.py so this logic is unit-testable without a Flet
runtime. Input shapes mirror the models in specs/api.yaml.
"""

from __future__ import annotations

from typing import Any

CANDIDATE_COLUMNS = [
    "Contract",
    "Type",
    "Strike",
    "Expiration",
    "Score",
    "Delta",
    "Breakeven",
    "Max Loss",
    "Max Gain",
    "PoP",
]

RUN_COLUMNS = ["Run ID", "Symbol", "Generated At", "Action", "Confidence"]

# Semantic tones, mapped to actual colors in app.py. Kept as plain strings
# here so this module has no dependency on Flet and stays unit-testable.
TONE_SUCCESS = "success"
TONE_WARNING = "warning"
TONE_DANGER = "danger"
TONE_NEUTRAL = "neutral"

_RECOMMENDATION_TONES = {
    "STRONG_BUY": TONE_SUCCESS,
    "BUY": TONE_SUCCESS,
    "HOLD": TONE_WARNING,
    "AVOID": TONE_DANGER,
}

_TREND_TONES = {
    "bullish": TONE_SUCCESS,
    "bearish": TONE_DANGER,
    "neutral": TONE_NEUTRAL,
}

_CONSENSUS_TONES = {
    "bullish": TONE_SUCCESS,
    "bearish": TONE_DANGER,
    "neutral": TONE_NEUTRAL,
    "mixed": TONE_WARNING,
}

_RISK_LEVEL_TONES = {
    "low": TONE_SUCCESS,
    "medium": TONE_WARNING,
    "high": TONE_DANGER,
}

_MACRO_REGIME_TONES = {
    "risk_on": TONE_SUCCESS,
    "risk_off": TONE_DANGER,
    "neutral": TONE_NEUTRAL,
}

# Financial-research finding tones. Mirrors the enums in
# specs/api.yaml (CompanyHealth/GrowthTrend/ProfitabilityLevel/CashFlowState):
# the healthiest value reads success, the weakest danger, the middle neutral.
_COMPANY_HEALTH_TONES = {
    "strong": TONE_SUCCESS,
    "stable": TONE_NEUTRAL,
    "weak": TONE_DANGER,
}

_GROWTH_TONES = {
    "accelerating": TONE_SUCCESS,
    "steady": TONE_NEUTRAL,
    "decelerating": TONE_DANGER,
}

_PROFITABILITY_TONES = {
    "high": TONE_SUCCESS,
    "moderate": TONE_NEUTRAL,
    "low": TONE_DANGER,
}

_CASH_FLOW_TONES = {
    "positive": TONE_SUCCESS,
    "neutral": TONE_NEUTRAL,
    "negative": TONE_DANGER,
}


def recommendation_tone(action: str) -> str:
    return _RECOMMENDATION_TONES.get(action, TONE_NEUTRAL)


def trend_tone(direction: str) -> str:
    return _TREND_TONES.get(direction, TONE_NEUTRAL)


def consensus_tone(consensus: str) -> str:
    return _CONSENSUS_TONES.get(consensus, TONE_NEUTRAL)


def risk_level_tone(risk_level: str) -> str:
    return _RISK_LEVEL_TONES.get(risk_level, TONE_NEUTRAL)


def macro_regime_tone(regime: str) -> str:
    return _MACRO_REGIME_TONES.get(regime, TONE_NEUTRAL)


def company_health_tone(value: str) -> str:
    return _COMPANY_HEALTH_TONES.get(value, TONE_NEUTRAL)


def growth_tone(value: str) -> str:
    return _GROWTH_TONES.get(value, TONE_NEUTRAL)


def profitability_tone(value: str) -> str:
    return _PROFITABILITY_TONES.get(value, TONE_NEUTRAL)


def cash_flow_tone(value: str) -> str:
    return _CASH_FLOW_TONES.get(value, TONE_NEUTRAL)


def quant_score_tone(score: float) -> str:
    """Tone for the quant overall_score (0-100), aligned with the
    recommendation thresholds in specs/scoring.yaml: BUY/STRONG_BUY (>=60)
    reads success, HOLD (>=40) warning, AVOID danger."""
    if score >= 60:
        return TONE_SUCCESS
    if score >= 40:
        return TONE_WARNING
    return TONE_DANGER


def format_recommendation(recommendation: dict[str, Any]) -> str:
    contract = recommendation.get("contract_symbol") or "—"
    rationale = recommendation.get("rationale", "")
    return f"{contract}\n{rationale}" if rationale else contract


def format_timestamp(value: str) -> str:
    """Trim an ISO datetime string down to minute precision for display."""
    return str(value).replace("T", " ")[:16]


def format_trend_summary(trend: dict[str, Any]) -> str:
    direction = str(trend.get("direction", "?")).capitalize()
    strength = trend.get("strength", "?")
    adx = trend.get("adx") or 0.0
    return f"Trend: {direction} · {strength} (ADX {adx:.1f})"


def format_volume_summary(volume: dict[str, Any]) -> str:
    relative_volume = volume.get("relative_volume") or 0.0
    flags = ", ".join(volume.get("flags") or []) or "none"
    return f"Volume: {relative_volume:.2f}x average · flags: {flags}"


def format_indicator_summary(indicators: dict[str, Any]) -> str:
    sma_20 = indicators.get("sma_20") or 0.0
    sma_50 = indicators.get("sma_50") or 0.0
    rsi_14 = indicators.get("rsi_14") or 0.0
    atr_14 = indicators.get("atr_14") or 0.0
    return f"SMA20 {sma_20:.2f} · SMA50 {sma_50:.2f} · RSI14 {rsi_14:.1f} · ATR14 {atr_14:.2f}"


def candidates_to_rows(candidates: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for candidate in candidates:
        max_gain = candidate.get("max_gain")
        rows.append(
            [
                str(candidate.get("contract_symbol", "")),
                str(candidate.get("option_type", "")).upper(),
                f"{candidate.get('strike', 0):.2f}",
                str(candidate.get("expiration", "")),
                f"{candidate.get('score', 0):.1f}",
                f"{candidate.get('delta', 0):.3f}",
                f"{candidate.get('breakeven', 0):.2f}",
                f"{candidate.get('max_loss', 0):.2f}",
                "unlimited" if max_gain is None else f"{max_gain:.2f}",
                f"{(candidate.get('probability_of_profit') or 0):.0%}",
            ]
        )
    return rows


def runs_to_rows(runs: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for run in runs:
        rows.append(
            [
                str(run.get("run_id", "")),
                str(run.get("symbol", "")),
                format_timestamp(run.get("generated_at", "")),
                str(run.get("recommendation_action", "")),
                f"{(run.get('recommendation_confidence') or 0):.0%}",
            ]
        )
    return rows
