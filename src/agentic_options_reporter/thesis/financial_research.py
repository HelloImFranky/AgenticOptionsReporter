"""Financial Research agent.

Interprets provider-supplied fundamentals (CompanyProfile,
FinancialStatementSummary, FinancialRatios, AnalystEstimates — all
FinancialProvider facts, never computed by this codebase) into
qualitative labels plus a narrative. analyst_consensus is passed through
verbatim from AnalystEstimates, never LLM-authored (see specs/agents.yaml).
"""

from __future__ import annotations

from pydantic import BaseModel

from agentic_options_reporter.models.schemas import (
    AnalystEstimates,
    CashFlowState,
    CompanyHealth,
    CompanyProfile,
    FinancialRatios,
    FinancialResearchFinding,
    FinancialStatementSummary,
    GrowthTrend,
    ProfitabilityLevel,
)
from agentic_options_reporter.thesis.llm_client import LlmClient
from agentic_options_reporter.thesis.parsing import parse_response

_SYSTEM_PROMPT = """\
You are a financial research analyst. You are given a company's profile,
financial statement summary, ratios, and analyst estimates, all already
retrieved from a data provider. Interpret them into qualitative labels
and a narrative — do not recompute or contradict any number you are given.

Respond with a single JSON object with exactly these keys:
{"company_health": "strong" | "stable" | "weak",
 "growth": "accelerating" | "steady" | "decelerating",
 "profitability": "high" | "moderate" | "low",
 "cash_flow": "positive" | "neutral" | "negative",
 "narrative": "<2-4 sentence plain-language summary>"}

Output ONLY the JSON object, no markdown fences, no extra text.
"""


class _LlmAuthoredFields(BaseModel):
    company_health: CompanyHealth
    growth: GrowthTrend
    profitability: ProfitabilityLevel
    cash_flow: CashFlowState
    narrative: str


def _build_prompt(
    profile: CompanyProfile,
    statements: FinancialStatementSummary,
    ratios: FinancialRatios,
    estimates: AnalystEstimates,
) -> str:
    return f"""\
Company: {profile.name} ({profile.ticker}) - {profile.sector}/{profile.industry}
Market cap: {profile.market_cap}

Latest statement ({statements.period}): revenue={statements.revenue} \
net_income={statements.net_income} operating_cash_flow={statements.operating_cash_flow} \
free_cash_flow={statements.free_cash_flow}

Ratios: PE={ratios.pe_ratio} PB={ratios.pb_ratio} debt_to_equity={ratios.debt_to_equity} \
current_ratio={ratios.current_ratio} ROE={ratios.return_on_equity} \
gross_margin={ratios.gross_margin} net_margin={ratios.net_margin}

Analyst estimates: consensus_rating={estimates.consensus_rating} \
price_target_mean={estimates.price_target_mean} num_analysts={estimates.num_analysts}
"""


def run(
    llm_client: LlmClient,
    profile: CompanyProfile,
    statements: FinancialStatementSummary,
    ratios: FinancialRatios,
    estimates: AnalystEstimates,
) -> FinancialResearchFinding:
    user_prompt = _build_prompt(profile, statements, ratios, estimates)
    raw = llm_client.complete(_SYSTEM_PROMPT, user_prompt)
    parsed = parse_response(_LlmAuthoredFields, raw, "financial_research")
    return FinancialResearchFinding(
        company_health=parsed.company_health,
        growth=parsed.growth,
        profitability=parsed.profitability,
        cash_flow=parsed.cash_flow,
        analyst_consensus=estimates.consensus_rating,
        narrative=parsed.narrative,
    )
