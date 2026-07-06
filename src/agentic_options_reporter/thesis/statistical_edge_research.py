"""Statistical Edge Research agent.

New in phase 3 (specs/agents.yaml) — the agent-side counterpart to the
quant Statistical Edge domain scorer (analysis/statistical_edge.py).
Reasons over the quant-computed win-rate/expectancy/Monte-Carlo numbers
plus qualitative pattern context (trend, support/resistance) — the one
place an agent's prompt includes a quant DomainScore as input, mirroring
quant_interpreter's existing precedent of receiving quant numbers as
context. Never recomputes the underlying statistics itself.
"""

from __future__ import annotations

from pydantic import BaseModel

from agentic_options_reporter.models.schemas import (
    DomainScore,
    ScoredCandidate,
    StatisticalEdgeFinding,
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
You are a quantitative-pattern analyst. You are given a deterministic
engine's Statistical Edge readout (historical win rate, expectancy,
pattern-match success, and a Monte Carlo bootstrap confidence) for the
current setup, plus the trend/support-resistance context it was computed
from — all already calculated, never by you. Interpret what this edge
readout means for the trade and how much weight it deserves given its own
stated confidence and sample size.

Respond with a single JSON object with exactly these keys:
{{"narrative": "<2-4 sentence plain-language summary>",
 {DOMAIN_SCORE_PROMPT_FIELD}}}

domain_score reflects the Statistical Edge domain: {DOMAIN_SCORE_PROMPT_RULE}
If the quant readout is unavailable or based on very few samples, say so
plainly and keep your confidence low rather than inventing certainty.

Output ONLY the JSON object, no markdown fences, no extra text.
"""


class _LlmAuthoredFields(BaseModel):
    narrative: str
    domain_score: LlmDomainScoreFields


def _build_prompt(
    quant_statistical_edge: DomainScore | None,
    trend: TrendAssessment,
    top_candidate: ScoredCandidate,
) -> str:
    if quant_statistical_edge is not None:
        evidence = "; ".join(quant_statistical_edge.evidence) or "none"
        quant_line = (
            f"Quant Statistical Edge score: {quant_statistical_edge.score:.1f}/100 "
            f"(confidence {quant_statistical_edge.confidence:.0f}/100). Evidence: {evidence}"
        )
    else:
        quant_line = "Quant Statistical Edge readout: not available for this run."

    return f"""\
{quant_line}

Current setup: {top_candidate.contract_symbol} ({top_candidate.option_type}, \
strike {top_candidate.strike})
Trend context: {trend.direction} ({trend.strength}, ADX {trend.adx:.1f})
"""


def run(
    llm_client: LlmClient,
    quant_statistical_edge: DomainScore | None,
    trend: TrendAssessment,
    top_candidate: ScoredCandidate,
) -> StatisticalEdgeFinding:
    user_prompt = _build_prompt(quant_statistical_edge, trend, top_candidate)
    raw = llm_client.complete(_SYSTEM_PROMPT, user_prompt)
    parsed = parse_response(_LlmAuthoredFields, raw, "statistical_edge_research")
    return StatisticalEdgeFinding(
        narrative=parsed.narrative,
        domain_score=assemble_domain_score("statistical_edge", parsed.domain_score),
    )
