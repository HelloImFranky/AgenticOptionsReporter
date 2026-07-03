"""Macro Research agent.

Interprets provider-supplied macroeconomic data (InterestRates,
CpiSnapshot, GdpSnapshot, MacroEvent list — all MacroProvider facts,
never computed by this codebase) into a risk-on/risk-off regime label
plus an outlook. Market-wide, not ticker-specific.
"""

from __future__ import annotations

from agentic_options_reporter.models.schemas import (
    CpiSnapshot,
    GdpSnapshot,
    InterestRates,
    MacroEvent,
    MacroResearchFinding,
)
from agentic_options_reporter.thesis.llm_client import LlmClient
from agentic_options_reporter.thesis.parsing import parse_response

_SYSTEM_PROMPT = """\
You are a macroeconomic analyst. You are given interest rates, CPI, GDP,
and any upcoming macro calendar events, all already retrieved from a
data provider. Characterize the overall regime and outlook for risk
assets — do not recompute or contradict any figure you are given.

Respond with a single JSON object with exactly these keys:
{"regime": "risk_on" | "risk_off" | "neutral",
 "outlook": "<1-3 sentence forward-looking view>",
 "summary": "<2-4 sentence plain-language summary of current conditions>"}

Output ONLY the JSON object, no markdown fences, no extra text.
"""


def _build_prompt(
    rates: InterestRates, cpi: CpiSnapshot, gdp: GdpSnapshot, calendar: list[MacroEvent]
) -> str:
    events = (
        "\n".join(f"- {event.event_date}: {event.name} ({event.importance})" for event in calendar)
        or "(none available)"
    )
    return f"""\
Interest rates (as of {rates.as_of}): fed_funds={rates.fed_funds_rate} \
10yr={rates.ten_year_yield} 2yr={rates.two_year_yield}
CPI (as of {cpi.as_of}): {cpi.value} (YoY {cpi.yoy_change_pct})
GDP (as of {gdp.as_of}): {gdp.value} (YoY {gdp.yoy_growth_pct})
Upcoming macro calendar:
{events}
"""


def run(
    llm_client: LlmClient,
    rates: InterestRates,
    cpi: CpiSnapshot,
    gdp: GdpSnapshot,
    calendar: list[MacroEvent],
) -> MacroResearchFinding:
    user_prompt = _build_prompt(rates, cpi, gdp, calendar)
    raw = llm_client.complete(_SYSTEM_PROMPT, user_prompt)
    return parse_response(MacroResearchFinding, raw, "macro_research")
