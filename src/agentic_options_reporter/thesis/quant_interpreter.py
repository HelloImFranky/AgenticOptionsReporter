"""Quant Interpreter agent.

Narrates the deterministic engine's output in plain language. Never
computes or alters a score: score_breakdown and overall_score are passed
through verbatim from the already-scored candidate (see
specs/agents.yaml).
"""

from __future__ import annotations

from pydantic import BaseModel

from agentic_options_reporter.models.schemas import (
    IndicatorSnapshot,
    QuantInterpretation,
    ScoredCandidate,
    TrendAssessment,
    VolumeAssessment,
)
from agentic_options_reporter.thesis.llm_client import LlmClient
from agentic_options_reporter.thesis.parsing import parse_response


class _NarrativeResponse(BaseModel):
    """The subset of QuantInterpretation the LLM is allowed to author."""

    narrative: str
    key_factors: list[str]

_SYSTEM_PROMPT = """\
You are a quantitative markets analyst. You are given technical
indicators and a scoring breakdown that a deterministic engine already
computed. Your job is ONLY to explain, in plain language, what these
numbers mean and why the candidate scored the way it did.

Rules:
- Do not invent, recompute, or contradict any number you are given.
- Do not output any numeric score yourself.
- Respond with a single JSON object with exactly these keys:
  {"narrative": "<2-4 sentence plain-language explanation>",
   "key_factors": ["<short phrase>", "..."]}
- key_factors should have 2-5 short phrases naming the strongest
  contributors to the score (positive or negative).
- Output ONLY the JSON object, no markdown fences, no extra text.
"""


def _build_prompt(
    indicators: IndicatorSnapshot,
    trend: TrendAssessment,
    volume: VolumeAssessment,
    top_candidate: ScoredCandidate,
) -> str:
    factors = ", ".join(f"{name}={value:.2f}" for name, value in top_candidate.score_breakdown.items())
    return f"""\
Trend: {trend.direction} ({trend.strength}, ADX {trend.adx:.1f})
Volume: {volume.relative_volume:.2f}x average, flags: {", ".join(volume.flags) or "none"}
Indicators: SMA20={indicators.sma_20:.2f} SMA50={indicators.sma_50:.2f} \
RSI14={indicators.rsi_14:.1f} MACD={indicators.macd:.3f} ATR14={indicators.atr_14:.2f}

Top candidate: {top_candidate.contract_symbol} ({top_candidate.option_type}, \
strike {top_candidate.strike})
Overall score: {top_candidate.score:.1f}/100
Score breakdown (each 0-1): {factors}
"""


def run(
    llm_client: LlmClient,
    indicators: IndicatorSnapshot,
    trend: TrendAssessment,
    volume: VolumeAssessment,
    top_candidate: ScoredCandidate,
) -> QuantInterpretation:
    user_prompt = _build_prompt(indicators, trend, volume, top_candidate)
    raw = llm_client.complete(_SYSTEM_PROMPT, user_prompt)

    # narrative/key_factors come from the model; score_breakdown/overall_score
    # are pass-throughs we control, so a malformed narrative can't corrupt them.
    parsed = parse_response(_NarrativeResponse, raw, "quant_interpreter")
    return QuantInterpretation(
        narrative=parsed.narrative,
        key_factors=parsed.key_factors,
        score_breakdown=dict(top_candidate.score_breakdown),
        overall_score=top_candidate.score,
    )
