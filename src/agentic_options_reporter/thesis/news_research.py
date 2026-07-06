"""News Research agent.

Summarizes recent company news (NewsProvider-supplied articles) into a
sentiment label, narrative summary, and lists of catalysts/risks. The
sentiment label is the agent's own qualitative judgment over the actual
articles — a judgment call over given facts, like risk_level — and never
a number the quant engine computes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_options_reporter.models.schemas import NewsArticle, NewsResearchFinding, NewsSentiment
from agentic_options_reporter.thesis.agent_domain_score import (
    DOMAIN_SCORE_PROMPT_FIELD,
    DOMAIN_SCORE_PROMPT_RULE,
    LlmDomainScoreFields,
    assemble_domain_score,
)
from agentic_options_reporter.thesis.llm_client import LlmClient
from agentic_options_reporter.thesis.parsing import parse_response

_SYSTEM_PROMPT = f"""\
You are a news research analyst. You are given recent news articles
about a company. Summarize what's happening and identify concrete
catalysts and risks — do not invent facts not present in the articles. You
ALSO independently score the Sentiment domain of a Trade Quality Score
(0-100) from this same material — {DOMAIN_SCORE_PROMPT_RULE}

Respond with a single JSON object with exactly these keys:
{{"sentiment": "bullish" | "bearish" | "neutral",
 "summary": "<2-4 sentence plain-language summary>",
 "catalysts": ["<short phrase>", "..."],
 "risks": ["<short phrase>", "..."],
 {DOMAIN_SCORE_PROMPT_FIELD}}}

catalysts and risks should each have 0-5 items. Output ONLY the JSON
object, no markdown fences, no extra text.
"""


class _LlmAuthoredFields(BaseModel):
    sentiment: NewsSentiment
    summary: str
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    domain_score: LlmDomainScoreFields


def _build_prompt(articles: list[NewsArticle]) -> str:
    if articles:
        article_lines = "\n".join(
            f"- [{article.published_at:%Y-%m-%d}] {article.source}: {article.headline} — {article.summary}"
            for article in articles
        )
    else:
        article_lines = "(no recent articles)"

    return f"""\
Recent articles:
{article_lines}
"""


def run(llm_client: LlmClient, articles: list[NewsArticle]) -> NewsResearchFinding:
    user_prompt = _build_prompt(articles)
    raw = llm_client.complete(_SYSTEM_PROMPT, user_prompt)
    parsed = parse_response(_LlmAuthoredFields, raw, "news_research")
    return NewsResearchFinding(
        sentiment=parsed.sentiment,
        summary=parsed.summary,
        catalysts=parsed.catalysts,
        risks=parsed.risks,
        domain_score=assemble_domain_score("sentiment", parsed.domain_score),
    )
