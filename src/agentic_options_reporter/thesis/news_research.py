"""News Research agent.

Summarizes recent company news (NewsProvider-supplied articles) into a
sentiment label, narrative summary, and lists of catalysts/risks. The
sentiment label is the agent's own qualitative judgment over the actual
articles — a judgment call over given facts, like risk_level — and never
a number the quant engine computes.
"""

from __future__ import annotations

from agentic_options_reporter.models.schemas import NewsArticle, NewsResearchFinding
from agentic_options_reporter.thesis.llm_client import LlmClient
from agentic_options_reporter.thesis.parsing import parse_response

_SYSTEM_PROMPT = """\
You are a news research analyst. You are given recent news articles
about a company. Summarize what's happening and identify concrete
catalysts and risks — do not invent facts not present in the articles.

Respond with a single JSON object with exactly these keys:
{"sentiment": "bullish" | "bearish" | "neutral",
 "summary": "<2-4 sentence plain-language summary>",
 "catalysts": ["<short phrase>", "..."],
 "risks": ["<short phrase>", "..."]}

catalysts and risks should each have 0-5 items. Output ONLY the JSON
object, no markdown fences, no extra text.
"""


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
    return parse_response(NewsResearchFinding, raw, "news_research")
