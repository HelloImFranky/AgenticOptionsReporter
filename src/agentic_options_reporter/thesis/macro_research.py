"""Macro Research agent.

Interprets provider-supplied macroeconomic observations (a list of
MacroObservation — all MacroProvider facts, never computed by this
codebase) into a risk-on/risk-off regime label plus an outlook.
Market-wide, not ticker-specific. Whatever metrics the router could
serve are passed in; the agent reasons over what's present rather than
assuming a fixed set (a keyless-only deployment, for example, sees CPI
and GDP but no policy rate).
"""

from __future__ import annotations

from agentic_options_reporter.models.schemas import MacroObservation, MacroResearchFinding
from agentic_options_reporter.thesis.llm_client import LlmClient
from agentic_options_reporter.thesis.parsing import parse_response

_SYSTEM_PROMPT = """\
You are a macroeconomic analyst. You are given a set of recent
macroeconomic indicators, all already retrieved from data providers.
Some indicators may be absent — reason over what is present, and do not
invent figures. Characterize the overall regime and outlook for risk
assets; do not recompute or contradict any figure you are given.

Respond with a single JSON object with exactly these keys:
{"regime": "risk_on" | "risk_off" | "neutral",
 "outlook": "<1-3 sentence forward-looking view>",
 "summary": "<2-4 sentence plain-language summary of current conditions>"}

Output ONLY the JSON object, no markdown fences, no extra text.
"""


def _format_observation(obs: MacroObservation) -> str:
    line = f"- {obs.label} (as of {obs.as_of}, {obs.source}): {obs.value} {obs.unit}"
    if obs.yoy_change_pct is not None:
        line += f", YoY {obs.yoy_change_pct:.1f}%"
    return line


def _build_prompt(observations: list[MacroObservation]) -> str:
    if observations:
        lines = "\n".join(_format_observation(obs) for obs in observations)
    else:
        lines = "(no macroeconomic indicators available)"
    return f"""\
Macroeconomic indicators:
{lines}
"""


def run(llm_client: LlmClient, observations: list[MacroObservation]) -> MacroResearchFinding:
    user_prompt = _build_prompt(observations)
    raw = llm_client.complete(_SYSTEM_PROMPT, user_prompt)
    return parse_response(MacroResearchFinding, raw, "macro_research")
