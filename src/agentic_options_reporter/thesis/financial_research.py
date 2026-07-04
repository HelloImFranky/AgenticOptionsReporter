"""Financial Research agent.

Interprets provider-supplied fundamentals (CompanyProfile,
FinancialStatementSummary, FinancialRatios, AnalystEstimates — all
FinancialProvider facts, never computed by this codebase) into
qualitative labels plus a narrative. analyst_consensus is passed through
verbatim from AnalystEstimates, never LLM-authored (see specs/agents.yaml).

The company profile is the required anchor; statements, ratios, and
estimates are each optional — a provider set that doesn't cover one
(e.g. Finnhub has no statements) simply omits that section, and the
agent reasons over what's present, the same way macro_research handles a
partial metric set.
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
You are a financial research analyst. You are given a company's profile
and, where available, its financial statement summary, ratios, and
analyst estimates, all already retrieved from a data provider. Some
sections may be marked unavailable — reason over what is present.
Interpret them into qualitative labels and a narrative — do not recompute
or contradict any number you are given.

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
    statements: FinancialStatementSummary | None,
    ratios: FinancialRatios | None,
    estimates: AnalystEstimates | None,
) -> str:
    statements_line = (
        f"Latest statement ({statements.period}): revenue={statements.revenue} "
        f"net_income={statements.net_income} operating_cash_flow={statements.operating_cash_flow} "
        f"free_cash_flow={statements.free_cash_flow}"
        if statements is not None
        else "Financial statements: not available."
    )
    ratios_line = (
        f"Ratios: PE={ratios.pe_ratio} PB={ratios.pb_ratio} debt_to_equity={ratios.debt_to_equity} "
        f"current_ratio={ratios.current_ratio} ROE={ratios.return_on_equity} "
        f"gross_margin={ratios.gross_margin} net_margin={ratios.net_margin}"
        if ratios is not None
        else "Ratios: not available."
    )
    estimates_line = (
        f"Analyst estimates: consensus_rating={estimates.consensus_rating} "
        f"price_target_mean={estimates.price_target_mean} num_analysts={estimates.num_analysts}"
        if estimates is not None
        else "Analyst estimates: not available."
    )
    return f"""\
Company: {profile.name} ({profile.ticker}) - {profile.sector}/{profile.industry}
Market cap: {profile.market_cap}

{statements_line}

{ratios_line}

{estimates_line}
"""


def run(
    llm_client: LlmClient,
    profile: CompanyProfile,
    statements: FinancialStatementSummary | None = None,
    ratios: FinancialRatios | None = None,
    estimates: AnalystEstimates | None = None,
) -> FinancialResearchFinding:
    user_prompt = _build_prompt(profile, statements, ratios, estimates)
    raw = llm_client.complete(_SYSTEM_PROMPT, user_prompt)
    parsed = parse_response(_LlmAuthoredFields, raw, "financial_research")
    return FinancialResearchFinding(
        company_health=parsed.company_health,
        growth=parsed.growth,
        profitability=parsed.profitability,
        cash_flow=parsed.cash_flow,
        # Pass-through from the provider, never LLM-authored; "N/A" when
        # no provider served analyst estimates.
        analyst_consensus=estimates.consensus_rating if estimates is not None else "N/A",
        narrative=parsed.narrative,
    )
