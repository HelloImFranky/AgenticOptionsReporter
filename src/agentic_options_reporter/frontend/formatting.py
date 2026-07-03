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


def format_recommendation(recommendation: dict[str, Any]) -> str:
    action = recommendation.get("action", "UNKNOWN")
    confidence = recommendation.get("confidence") or 0.0
    contract = recommendation.get("contract_symbol") or "—"
    rationale = recommendation.get("rationale", "")
    return f"{action} ({confidence:.0%}) — {contract}\n{rationale}"


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
                str(run.get("generated_at", "")),
                str(run.get("recommendation_action", "")),
                f"{(run.get('recommendation_confidence') or 0):.0%}",
            ]
        )
    return rows
