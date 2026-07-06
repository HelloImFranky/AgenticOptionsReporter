"""Quant Interpreter agent.

Narrates the deterministic engine's output in plain language, and never
alters the quant engine's own numbers: quant_trade_quality is passed
through verbatim from the already-scored candidate (see specs/agents.yaml).
Since phase 3, it ALSO independently authors its own Technical domain
score (technical_domain_score) — a separate, agent-side judgment call over
the same indicators, feeding the agent-side composite Trade Quality Score
(thesis/orchestrator.py) alongside (not instead of) the quant one.
"""

from __future__ import annotations

from pydantic import BaseModel

from agentic_options_reporter.analysis.composite_score import compute_composite_score
from agentic_options_reporter.models.schemas import (
    IndicatorSnapshot,
    QuantInterpretation,
    ScoredCandidate,
    TrendAssessment,
    VolumeAssessment,
    WeightingProfileId,
)
from agentic_options_reporter.thesis.agent_domain_score import (
    DOMAIN_SCORE_PROMPT_FIELD,
    DOMAIN_SCORE_PROMPT_RULE,
    LlmDomainScoreFields,
    assemble_domain_score,
)
from agentic_options_reporter.thesis.llm_client import LlmClient
from agentic_options_reporter.thesis.parsing import parse_response


class _NarrativeResponse(BaseModel):
    """The subset of QuantInterpretation the LLM is allowed to author."""

    narrative: str
    key_factors: list[str]
    domain_score: LlmDomainScoreFields


_SYSTEM_PROMPT = f"""\
You are a quantitative markets analyst. You are given technical
indicators and a Trade Quality Score breakdown that a deterministic engine
already computed. Your job is to explain, in plain language, what these
numbers mean and why the candidate scored the way it did.

Rules:
- Do not invent, recompute, or contradict any number you are given.
- Do not output your own composite score.
- You ALSO independently score the Technical domain (0-100) from the
  indicators given — {DOMAIN_SCORE_PROMPT_RULE} This is YOUR OWN read of
  the technicals, separate from (and may disagree with) the quant
  engine's own Technical domain score shown to you.
- Respond with a single JSON object with exactly these keys:
  {{"narrative": "<2-4 sentence plain-language explanation>",
   "key_factors": ["<short phrase>", "..."],
   {DOMAIN_SCORE_PROMPT_FIELD}}}
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
    technical = top_candidate.domain_scores.get("technical")
    factors = (
        ", ".join(f"{f.name}={f.value:.2f}" for f in technical.factors)
        if technical is not None
        else "not available"
    )
    technical_score_line = f"{technical.score:.1f}/100" if technical is not None else "not available"
    return f"""\
Trend: {trend.direction} ({trend.strength}, ADX {trend.adx:.1f})
Volume: {volume.relative_volume:.2f}x average, flags: {", ".join(volume.flags) or "none"}
Indicators: SMA20={indicators.sma_20:.2f} SMA50={indicators.sma_50:.2f} \
RSI14={indicators.rsi_14:.1f} MACD={indicators.macd:.3f} ATR14={indicators.atr_14:.2f}

Top candidate: {top_candidate.contract_symbol} ({top_candidate.option_type}, \
strike {top_candidate.strike})
Overall Trade Quality Score: {top_candidate.score:.1f}/100
Quant engine's own Technical domain score: {technical_score_line}
Technical sub-factors (each 0-1): {factors}
"""


def run(
    llm_client: LlmClient,
    indicators: IndicatorSnapshot,
    trend: TrendAssessment,
    volume: VolumeAssessment,
    top_candidate: ScoredCandidate,
    weighting_profile: WeightingProfileId = "swing",
) -> QuantInterpretation:
    user_prompt = _build_prompt(indicators, trend, volume, top_candidate)
    raw = llm_client.complete(_SYSTEM_PROMPT, user_prompt)

    # narrative/key_factors/domain_score come from the model; the quant
    # composite is a pass-through we control, so a malformed narrative
    # can't corrupt it.
    parsed = parse_response(_NarrativeResponse, raw, "quant_interpreter")
    quant_trade_quality = compute_composite_score(
        top_candidate.domain_scores,
        source="quant",
        weighting_profile=weighting_profile,
        contract_symbol=top_candidate.contract_symbol,
    )
    return QuantInterpretation(
        narrative=parsed.narrative,
        key_factors=parsed.key_factors,
        quant_trade_quality=quant_trade_quality,
        technical_domain_score=assemble_domain_score("technical", parsed.domain_score),
    )
