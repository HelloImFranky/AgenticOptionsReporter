"""Options Strategy agent.

Suggests a strategy *shape* (e.g. "Bull Call Spread", "Cash-Secured Put",
"Long Call", "Avoid new position") given directional bias, the top
candidate's risk profile, and the risk_challenger's concerns. Does not
price a spread or select specific strikes/legs — that would require new
deterministic multi-leg pricing, out of scope for phase 1 (see
specs/agents.yaml).
"""

from __future__ import annotations

from agentic_options_reporter.models.schemas import (
    RiskAssessment,
    ScoredCandidate,
    StrategySuggestion,
    TrendAssessment,
)
from agentic_options_reporter.thesis.llm_client import LlmClient
from agentic_options_reporter.thesis.parsing import parse_response

_SYSTEM_PROMPT = """\
You are an options strategist. Given a directional view, a specific
candidate contract already selected by a deterministic scoring engine,
and a risk manager's concerns, recommend a strategy SHAPE — not a
specific priced instrument. Consider whether a defined-risk structure
(e.g. a debit spread, a cash-secured put, a covered call) would be more
appropriate than the single-leg contract shown, given the stated risk
concerns.

Respond with a single JSON object with exactly these keys:
{"strategy": "<short strategy name, e.g. 'Bull Call Spread'>",
 "rationale": "<1-3 sentence justification>"}

Output ONLY the JSON object, no markdown fences, no extra text.
"""


def _build_prompt(
    trend: TrendAssessment,
    top_candidate: ScoredCandidate,
    risk_assessment: RiskAssessment,
) -> str:
    return f"""\
Directional view: {trend.direction} ({trend.strength})
Candidate under consideration: {top_candidate.contract_symbol} \
({top_candidate.option_type}, strike {top_candidate.strike}, \
expires {top_candidate.expiration})
Candidate risk/reward: max_loss={top_candidate.max_loss:.2f} \
max_gain={"unlimited" if top_candidate.max_gain is None else f"{top_candidate.max_gain:.2f}"} \
probability_of_profit={top_candidate.probability_of_profit:.0%}
Risk manager's assessment: level={risk_assessment.risk_level}, \
concerns={"; ".join(risk_assessment.concerns)}
"""


def run(
    llm_client: LlmClient,
    trend: TrendAssessment,
    top_candidate: ScoredCandidate,
    risk_assessment: RiskAssessment,
) -> StrategySuggestion:
    user_prompt = _build_prompt(trend, top_candidate, risk_assessment)
    raw = llm_client.complete(_SYSTEM_PROMPT, user_prompt)
    return parse_response(StrategySuggestion, raw, "options_strategy")
