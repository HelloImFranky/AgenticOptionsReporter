"""Risk Challenger agent.

Argues against the trade: given the already-computed risk profile and
Greeks for the top candidate, plus trend and support/resistance context,
produces a qualitative risk judgment. Unlike QuantInterpretation, the
risk_level label here IS agent-authored — it is a judgment call about
soft, competing factors, not a formula the deterministic engine already
computes (see specs/agents.yaml).
"""

from __future__ import annotations

from pydantic import BaseModel

from agentic_options_reporter.models.schemas import (
    RiskAssessment,
    RiskLevel,
    ScoredCandidate,
    SupportResistanceLevel,
    TrendAssessment,
)
from agentic_options_reporter.thesis.agent_domain_score import (
    DOMAIN_SCORE_PROMPT_FIELD,
    DOMAIN_SCORE_PROMPT_RULE,
    LlmDomainScoreFields,
    assemble_domain_score,
)
from agentic_options_reporter.thesis.llm_client import LlmClient
from agentic_options_reporter.thesis.parsing import parse_response

_SYSTEM_PROMPT = f"""\
You are a skeptical risk manager reviewing a proposed options trade. Your
job is to argue against the trade: identify concrete reasons it could
fail, given the data provided. Do not recompute any numbers; reason about
the ones you are given. You ALSO independently score the Risk domain of a
Trade Quality Score (0-100) from this same material — a HIGH score means
LOW risk (a well-managed, favorable risk profile), not high danger —
{DOMAIN_SCORE_PROMPT_RULE}

Respond with a single JSON object with exactly these keys:
{{"risk_level": "low" | "medium" | "high",
 "concerns": ["<short concrete concern>", "..."],
 "position_sizing_note": "<one sentence position-sizing guidance>",
 {DOMAIN_SCORE_PROMPT_FIELD}}}

concerns should have 1-5 items. Output ONLY the JSON object, no markdown
fences, no extra text.
"""


class _LlmAuthoredFields(BaseModel):
    risk_level: RiskLevel
    concerns: list[str]
    position_sizing_note: str
    domain_score: LlmDomainScoreFields


def _build_prompt(
    top_candidate: ScoredCandidate,
    trend: TrendAssessment,
    levels: list[SupportResistanceLevel],
) -> str:
    nearby_levels = ", ".join(
        f"{lvl.level_type} @ {lvl.price:.2f} ({lvl.touches} touches)"
        for lvl in sorted(levels, key=lambda l: l.touches, reverse=True)[:3]
    ) or "none identified"

    return f"""\
Candidate: {top_candidate.contract_symbol} ({top_candidate.option_type}, \
strike {top_candidate.strike}, expires {top_candidate.expiration})
Greeks: delta={top_candidate.delta:.3f} gamma={top_candidate.gamma:.4f} \
theta={top_candidate.theta:.3f} vega={top_candidate.vega:.3f}
Risk: max_loss={top_candidate.max_loss:.2f} \
max_gain={"unlimited" if top_candidate.max_gain is None else f"{top_candidate.max_gain:.2f}"} \
breakeven={top_candidate.breakeven:.2f} \
reward_risk_ratio={top_candidate.reward_risk_ratio} \
probability_of_profit={top_candidate.probability_of_profit:.0%}
Trend: {trend.direction} ({trend.strength}, ADX {trend.adx:.1f})
Nearby support/resistance levels: {nearby_levels}
"""


def run(
    llm_client: LlmClient,
    top_candidate: ScoredCandidate,
    trend: TrendAssessment,
    levels: list[SupportResistanceLevel],
) -> RiskAssessment:
    user_prompt = _build_prompt(top_candidate, trend, levels)
    raw = llm_client.complete(_SYSTEM_PROMPT, user_prompt)
    parsed = parse_response(_LlmAuthoredFields, raw, "risk_challenger")
    return RiskAssessment(
        risk_level=parsed.risk_level,
        concerns=parsed.concerns,
        position_sizing_note=parsed.position_sizing_note,
        domain_score=assemble_domain_score("risk", parsed.domain_score),
    )
