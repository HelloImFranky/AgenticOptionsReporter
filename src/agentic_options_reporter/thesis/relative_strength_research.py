"""Relative Strength Research agent.

New in phase 3 (specs/agents.yaml) — the agent-side counterpart to the
quant Relative Strength domain scorer (analysis/domain_scoring.py). Reasons
over the same symbol-vs-benchmark/sector return facts (fetched by the
orchestrator via MarketDataProvider — no new provider interface) and
independently scores the domain; never computes the returns itself.
"""

from __future__ import annotations

from pydantic import BaseModel

from agentic_options_reporter.models.schemas import RelativeStrengthFinding
from agentic_options_reporter.thesis.agent_domain_score import (
    DOMAIN_SCORE_PROMPT_FIELD,
    DOMAIN_SCORE_PROMPT_RULE,
    LlmDomainScoreFields,
    assemble_domain_score,
)
from agentic_options_reporter.thesis.llm_client import LlmClient
from agentic_options_reporter.thesis.parsing import parse_response

_SYSTEM_PROMPT = f"""\
You are a relative-strength analyst. You are given a stock's trailing
return alongside a broad market benchmark (SPY) and, where known, its
sector ETF — both already retrieved, never computed by you. Characterize
whether the stock is leading or lagging its market/sector, and how that
bears on a directional options trade.

Respond with a single JSON object with exactly these keys:
{{"narrative": "<2-4 sentence plain-language summary>",
 {DOMAIN_SCORE_PROMPT_FIELD}}}

domain_score reflects the Relative Strength domain: {DOMAIN_SCORE_PROMPT_RULE}
A bullish trade is supported by outperformance; a bearish trade is
supported by underperformance.

Output ONLY the JSON object, no markdown fences, no extra text.
"""


class _LlmAuthoredFields(BaseModel):
    narrative: str
    domain_score: LlmDomainScoreFields


def _format_return(value: float | None) -> str:
    return f"{value:+.1%}" if value is not None else "not available"


def _build_prompt(
    option_type: str,
    symbol: str,
    symbol_return: float | None,
    benchmark_return: float | None,
    sector_return: float | None,
    sector_label: str | None,
) -> str:
    bias = "bullish (call)" if option_type == "call" else "bearish (put)"
    return f"""\
Directional bias under consideration: {bias}
{symbol} 21-trading-day return: {_format_return(symbol_return)}
SPY (market) 21-trading-day return: {_format_return(benchmark_return)}
Sector ETF ({sector_label or "unknown sector"}) 21-trading-day return: {_format_return(sector_return)}
"""


def run(
    llm_client: LlmClient,
    option_type: str,
    symbol: str,
    symbol_return: float | None,
    benchmark_return: float | None,
    sector_return: float | None,
    sector_label: str | None = None,
) -> RelativeStrengthFinding:
    user_prompt = _build_prompt(
        option_type, symbol, symbol_return, benchmark_return, sector_return, sector_label
    )
    raw = llm_client.complete(_SYSTEM_PROMPT, user_prompt)
    parsed = parse_response(_LlmAuthoredFields, raw, "relative_strength_research")
    return RelativeStrengthFinding(
        narrative=parsed.narrative,
        domain_score=assemble_domain_score("relative_strength", parsed.domain_score),
    )
