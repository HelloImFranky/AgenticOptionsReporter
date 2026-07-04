"""Catalyst Research agent.

Combines three provider streams — recent news (NewsProvider), recent SEC
filings (SECProvider), and macro indicators (MacroProvider) — into a
structured list of discrete CATALYSTS: dateable events that could move
the stock, each classified by category, timing horizon, and directional
impact, plus a net directional bias.

Distinct from news_research (which authors an overall sentiment summary):
this agent extracts individual, dateable events and reasons about their
expected impact. It reasons over whatever streams are present — a
keyless-only deployment still has SEC filings — and never invents an
event not grounded in the given material. Every field is a qualitative
judgment over provider facts, never a number this codebase computes.
"""

from __future__ import annotations

from agentic_options_reporter.models.schemas import (
    CatalystFinding,
    MacroObservation,
    NewsArticle,
    SecFiling,
)
from agentic_options_reporter.thesis.llm_client import LlmClient
from agentic_options_reporter.thesis.parsing import parse_response

_SYSTEM_PROMPT = """\
You are a catalyst analyst. You are given recent news articles, recent
SEC filings, and macroeconomic indicators for a company. Identify the
discrete CATALYSTS — dateable events that could move the stock, such as
earnings, regulatory filings, product launches, corporate actions, or
macro releases. For each, judge its category, timing horizon, and likely
directional impact. Do not invent events not grounded in the material;
some streams may be empty — reason over what is present.

Classify each catalyst's `horizon` as:
- "recent": already occurred (e.g. a filing just made, news just broken)
- "near_term": expected within roughly the next few weeks
- "long_term": expected months out
- "unknown": the material gives no datable timing

Respond with a single JSON object with exactly these keys:
{"catalysts": [
   {"title": "<short label>",
    "category": "earnings" | "filing" | "news" | "macro" | "corporate_action" | "other",
    "horizon": "recent" | "near_term" | "long_term" | "unknown",
    "direction": "bullish" | "bearish" | "uncertain",
    "detail": "<1-2 sentence explanation grounded in the material>"}
 ],
 "summary": "<2-4 sentence plain-language summary>",
 "net_bias": "bullish" | "bearish" | "neutral" | "mixed"}

catalysts should have 0-8 items, most material first. Output ONLY the
JSON object, no markdown fences, no extra text.
"""


def _format_articles(articles: list[NewsArticle]) -> str:
    if not articles:
        return "(no recent articles)"
    return "\n".join(
        f"- [{a.published_at:%Y-%m-%d}] {a.source}: {a.headline} — {a.summary}" for a in articles
    )


def _format_filings(filings: list[SecFiling]) -> str:
    if not filings:
        return "(no recent filings)"
    return "\n".join(f"- [{f.filed_at:%Y-%m-%d}] {f.form_type} ({f.ticker})" for f in filings)


def _format_observations(observations: list[MacroObservation]) -> str:
    if not observations:
        return "(no macroeconomic indicators)"
    lines = []
    for obs in observations:
        line = f"- {obs.label} (as of {obs.as_of}, {obs.source}): {obs.value} {obs.unit}"
        if obs.yoy_change_pct is not None:
            line += f", YoY {obs.yoy_change_pct:.1f}%"
        lines.append(line)
    return "\n".join(lines)


def _build_prompt(
    articles: list[NewsArticle],
    filings: list[SecFiling],
    observations: list[MacroObservation],
) -> str:
    return f"""\
Recent news:
{_format_articles(articles)}

Recent SEC filings:
{_format_filings(filings)}

Macroeconomic indicators:
{_format_observations(observations)}
"""


def run(
    llm_client: LlmClient,
    articles: list[NewsArticle],
    filings: list[SecFiling],
    observations: list[MacroObservation],
) -> CatalystFinding:
    user_prompt = _build_prompt(articles, filings, observations)
    raw = llm_client.complete(_SYSTEM_PROMPT, user_prompt)
    return parse_response(CatalystFinding, raw, "catalyst_research")
