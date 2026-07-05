"""Render a full analysis + agent-pipeline run as a polished PDF report.

Kept separate from app.py (like formatting.py) so it has no Flet dependency
and can be unit-tested by asserting on the returned bytes. `build_report_pdf`
is a pure function: given the same JSON payloads the front end already holds
(an analysis result plus, optionally, an investment-thesis result), it returns
the bytes of a self-contained PDF. Input shapes mirror specs/api.yaml.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from agentic_options_reporter.frontend.formatting import (
    company_health_tone,
    consensus_tone,
    format_timestamp,
    macro_regime_tone,
    quant_score_tone,
    recommendation_facts,
    recommendation_tone,
    risk_level_tone,
    technical_snapshot_facts,
    trend_tone,
)

# Semantic tone -> print colour. The same success/warning/danger/neutral
# vocabulary the UI uses (formatting.py), mapped to slightly deeper shades
# that stay legible as filled badges on white paper.
_TONE_COLORS = {
    "success": colors.HexColor("#2E7D32"),
    "warning": colors.HexColor("#B26A00"),
    "danger": colors.HexColor("#C62828"),
    "neutral": colors.HexColor("#546E7A"),
}
_INK = colors.HexColor("#1A1C1E")
_MUTED = colors.HexColor("#5F6368")
_HAIRLINE = colors.HexColor("#D9DCE1")
_TABLE_HEADER_BG = colors.HexColor("#F1F3F7")


def _tone_color(tone: str) -> colors.Color:
    return _TONE_COLORS.get(tone, _TONE_COLORS["neutral"])


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["BodyText"]
    return {
        "title": ParagraphStyle(
            "AorTitle", parent=base, fontName="Helvetica-Bold", fontSize=20,
            leading=24, textColor=_INK, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "AorSubtitle", parent=base, fontName="Helvetica", fontSize=10,
            leading=13, textColor=_MUTED,
        ),
        "section": ParagraphStyle(
            "AorSection", parent=base, fontName="Helvetica-Bold", fontSize=13,
            leading=16, textColor=_INK, spaceBefore=4, spaceAfter=4,
        ),
        "agent": ParagraphStyle(
            "AorAgent", parent=base, fontName="Helvetica-Bold", fontSize=11,
            leading=14, textColor=_INK, spaceBefore=2, spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "AorBody", parent=base, fontName="Helvetica", fontSize=9.5,
            leading=13.5, textColor=_INK, alignment=TA_LEFT, spaceAfter=2,
        ),
        "muted": ParagraphStyle(
            "AorMuted", parent=base, fontName="Helvetica-Oblique", fontSize=9,
            leading=12, textColor=_MUTED, spaceAfter=2,
        ),
        "bullet": ParagraphStyle(
            "AorBullet", parent=base, fontName="Helvetica", fontSize=9.5,
            leading=13, textColor=_INK, leftIndent=10, bulletIndent=0, spaceAfter=1,
        ),
        "cell": ParagraphStyle(
            "AorCell", parent=base, fontName="Helvetica", fontSize=8,
            leading=10, textColor=_INK,
        ),
        "cellhead": ParagraphStyle(
            "AorCellHead", parent=base, fontName="Helvetica-Bold", fontSize=8,
            leading=10, textColor=_INK,
        ),
        "cellmuted": ParagraphStyle(
            "AorCellMuted", parent=base, fontName="Helvetica", fontSize=7.5,
            leading=10, textColor=_MUTED,
        ),
    }


def _facts_table(
    facts: list[tuple[str, str]], styles: dict[str, ParagraphStyle], cols: int = 2
) -> list[Any]:
    """Lay (label, value) facts out as a light key/value grid, `cols` pairs
    per row — the print counterpart of the UI's fact boxes, so the data
    reads as a scannable table instead of a run-on sentence."""
    if not facts:
        return []
    rows: list[list[Any]] = []
    for i in range(0, len(facts), cols):
        chunk = facts[i : i + cols]
        cells: list[Any] = []
        for label, value in chunk:
            cells.append(Paragraph(escape(label.upper()), styles["cellmuted"]))
            cells.append(Paragraph(f"<b>{escape(str(value))}</b>", styles["cell"]))
        while len(cells) < cols * 2:
            cells.append(Paragraph("", styles["cell"]))
        rows.append(cells)
    col_widths: list[float] = []
    for _ in range(cols):
        col_widths += [0.95 * inch, 1.55 * inch]
    table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, _HAIRLINE),
            ]
        )
    )
    return [table]


def _badge(text: str, tone: str, styles: dict[str, ParagraphStyle]) -> Table:
    """A small filled pill (a one-cell table sized to hug its text) carrying a
    tone colour — the print equivalent of the UI's leading section badges. The
    label is a plain string, not a Paragraph, so the cell shrinks to fit
    instead of greedily filling the frame width."""
    font, size, pad = "Helvetica-Bold", 8.5, 7
    width = stringWidth(text, font, size) + 2 * pad + 1
    table = Table([[text]], colWidths=[width], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _tone_color(tone)),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("FONTSIZE", (0, 0), (-1, -1), size),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), pad),
                ("RIGHTPADDING", (0, 0), (-1, -1), pad),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ("ROUNDEDCORNERS", [4, 4, 4, 4]),
            ]
        )
    )
    return table


def _section_header(text: str, styles: dict[str, ParagraphStyle]) -> list[Any]:
    return [
        Spacer(1, 10),
        Paragraph(escape(text), styles["section"]),
        HRFlowable(width="100%", thickness=1, color=_HAIRLINE, spaceAfter=6, spaceBefore=0),
    ]


def _bullets(items: list[str], styles: dict[str, ParagraphStyle]) -> list[Any]:
    return [
        Paragraph(escape(str(item)), styles["bullet"], bulletText="•")
        for item in items
        if str(item).strip()
    ]


def _recommendation_block(
    rec: dict[str, Any], candidates: list[dict[str, Any]] | None, styles: dict[str, ParagraphStyle]
) -> list[Any]:
    action = rec.get("action", "—")
    confidence = rec.get("confidence") or 0.0
    rationale = rec.get("rationale") or ""
    row = Table(
        [[_badge(action, recommendation_tone(action), styles),
          Paragraph(f"<b>{confidence:.0%}</b> confidence", styles["body"])]],
        colWidths=[1.6 * inch, None],
        hAlign="LEFT",
    )
    row.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    block: list[Any] = [row, Spacer(1, 6)]
    block.extend(_facts_table(recommendation_facts(rec, candidates), styles))
    if rationale:
        block.append(Spacer(1, 6))
        block.append(Paragraph(escape(rationale), styles["body"]))
    return block


def _candidates_table(candidates: list[dict[str, Any]], styles: dict[str, ParagraphStyle]) -> list[Any]:
    if not candidates:
        return [Paragraph("No scored candidates for this run.", styles["muted"])]

    headers = ["Contract", "Type", "Strike", "Expiration", "Score", "Delta", "PoP"]
    data = [[Paragraph(h, styles["cellhead"]) for h in headers]]
    for c in candidates[:10]:
        pop = c.get("probability_of_profit") or 0.0
        data.append(
            [
                Paragraph(escape(str(c.get("contract_symbol", ""))), styles["cell"]),
                Paragraph(escape(str(c.get("option_type", "")).upper()), styles["cell"]),
                Paragraph(f"{c.get('strike', 0):.2f}", styles["cell"]),
                Paragraph(escape(str(c.get("expiration", ""))), styles["cell"]),
                Paragraph(f"{c.get('score', 0):.1f}", styles["cell"]),
                Paragraph(f"{c.get('delta', 0):.3f}", styles["cell"]),
                Paragraph(f"{pop:.0%}", styles["cell"]),
            ]
        )
    table = Table(
        data,
        colWidths=[1.9 * inch, 0.6 * inch, 0.7 * inch, 1.1 * inch, 0.6 * inch, 0.7 * inch, 0.6 * inch],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _TABLE_HEADER_BG),
                ("LINEBELOW", (0, 0), (-1, 0), 0.75, _HAIRLINE),
                ("LINEBELOW", (0, 1), (-1, -2), 0.25, _HAIRLINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return [table]


def _agent_block(
    name: str, styles: dict[str, ParagraphStyle], *, badge: tuple[str, str] | None = None,
    body: list[Any] | None = None, skipped: str | None = None,
) -> KeepTogether:
    """One agent's contribution, kept together so a heading never orphans at a
    page break. Either a populated body (with optional leading badge) or an
    italic skipped note."""
    controls: list[Any] = [Paragraph(escape(name), styles["agent"])]
    if skipped is not None:
        controls.append(Paragraph(escape(skipped), styles["muted"]))
    else:
        if badge is not None:
            controls.append(_badge(badge[0], badge[1], styles))
            controls.append(Spacer(1, 3))
        controls.extend(body or [])
    controls.append(Spacer(1, 6))
    return KeepTogether(controls)


def _pipeline_blocks(thesis: dict[str, Any], styles: dict[str, ParagraphStyle]) -> list[Any]:
    warnings = thesis.get("pipeline_warnings") or []

    def skip_reason(agent_key: str, default: str) -> str:
        if any(str(w).startswith(f"{agent_key}:") for w in warnings):
            return "Skipped — provider failed during the run."
        return default

    blocks: list[Any] = []

    quant = thesis.get("quant_interpretation") or {}
    q_score = quant.get("overall_score") or 0.0
    q_body = [Paragraph(escape(quant.get("narrative", "")), styles["body"])]
    factors = quant.get("key_factors") or []
    if factors:
        q_body.append(Paragraph("<b>Key factors:</b> " + escape(" · ".join(factors)), styles["body"]))
    blocks.append(
        _agent_block(
            "Quant Interpreter", styles,
            badge=(f"SCORE {q_score:.0f}/100", quant_score_tone(q_score)), body=q_body,
        )
    )

    financial = thesis.get("financial_research")
    if financial is not None:
        f_body = [
            Paragraph(
                "Growth: <b>{}</b> &nbsp;·&nbsp; Profitability: <b>{}</b> &nbsp;·&nbsp; "
                "Cash flow: <b>{}</b>".format(
                    escape(str(financial.get("growth", "—"))),
                    escape(str(financial.get("profitability", "—"))),
                    escape(str(financial.get("cash_flow", "—"))),
                ),
                styles["body"],
            ),
            Paragraph(f"Analyst consensus: {escape(str(financial.get('analyst_consensus', '—')))}", styles["muted"]),
            Paragraph(escape(financial.get("narrative", "")), styles["body"]),
        ]
        blocks.append(
            _agent_block(
                "Financial Research", styles,
                badge=(f"HEALTH: {str(financial.get('company_health', '—')).upper()}",
                       company_health_tone(financial.get("company_health", ""))),
                body=f_body,
            )
        )
    else:
        blocks.append(
            _agent_block("Financial Research", styles,
                         skipped=skip_reason("financial_research", "Skipped — no financial data provider configured."))
        )

    news = thesis.get("news_research")
    if news is not None:
        n_body = [Paragraph(escape(news.get("summary", "")), styles["body"])]
        if news.get("catalysts"):
            n_body.append(Paragraph("<b>Catalysts</b>", styles["body"]))
            n_body.extend(_bullets(news["catalysts"], styles))
        if news.get("risks"):
            n_body.append(Paragraph("<b>Risks</b>", styles["body"]))
            n_body.extend(_bullets(news["risks"], styles))
        blocks.append(
            _agent_block("News Research", styles,
                         badge=(str(news.get("sentiment", "—")).upper(), trend_tone(news.get("sentiment", ""))),
                         body=n_body)
        )
    else:
        blocks.append(
            _agent_block("News Research", styles,
                         skipped=skip_reason("news_research", "Skipped — no news data provider configured."))
        )

    macro = thesis.get("macro_research")
    if macro is not None:
        m_body = [
            Paragraph(escape(macro.get("summary", "")), styles["body"]),
            Paragraph(escape(macro.get("outlook", "")), styles["muted"]),
        ]
        blocks.append(
            _agent_block("Macro Research", styles,
                         badge=(str(macro.get("regime", "—")).upper().replace("_", " "),
                                macro_regime_tone(macro.get("regime", ""))),
                         body=m_body)
        )
    else:
        blocks.append(
            _agent_block("Macro Research", styles,
                         skipped=skip_reason("macro_research", "Skipped — no macro data provider configured."))
        )

    catalyst = thesis.get("catalyst_research")
    if catalyst is not None:
        c_body = [Paragraph(escape(catalyst.get("summary", "")), styles["body"])]
        items = catalyst.get("catalysts") or []
        if items:
            for item in items:
                meta = "{} · {} · {}".format(
                    str(item.get("category", "—")).replace("_", " "),
                    str(item.get("horizon", "—")).replace("_", " "),
                    str(item.get("direction", "—")),
                )
                line = f"<b>{escape(str(item.get('title', '')))}</b> ({escape(meta)})"
                if item.get("detail"):
                    line += f" — {escape(str(item['detail']))}"
                c_body.append(Paragraph(line, styles["bullet"], bulletText="•"))
        else:
            c_body.append(Paragraph("No discrete catalysts identified.", styles["muted"]))
        net_bias = str(catalyst.get("net_bias", "—"))
        blocks.append(
            _agent_block("Catalyst Research", styles,
                         badge=(f"NET: {net_bias.upper()}", consensus_tone(net_bias)), body=c_body)
        )
    else:
        blocks.append(
            _agent_block(
                "Catalyst Research", styles,
                skipped=skip_reason("catalyst_research", "Skipped — no news, SEC, or macro provider configured."),
            )
        )

    risk = thesis.get("risk_assessment")
    if risk is not None:
        r_body = _bullets(risk.get("concerns") or [], styles)
        if risk.get("position_sizing_note"):
            r_body.append(Paragraph(escape(risk["position_sizing_note"]), styles["muted"]))
        blocks.append(
            _agent_block("Risk Challenger", styles,
                         badge=(str(risk.get("risk_level", "—")).upper(), risk_level_tone(risk.get("risk_level", ""))),
                         body=r_body)
        )
    else:
        blocks.append(
            _agent_block("Risk Challenger", styles, skipped="Skipped — no candidate contract to assess.")
        )

    strategy = thesis.get("strategy_suggestion")
    if strategy is not None:
        s_body = [
            Paragraph(f"<b>{escape(str(strategy.get('strategy', '')))}</b>", styles["body"]),
            Paragraph(escape(str(strategy.get("rationale", ""))), styles["muted"]),
        ]
        blocks.append(_agent_block("Options Strategist", styles, body=s_body))
    else:
        blocks.append(
            _agent_block("Options Strategist", styles,
                         skipped="Skipped — no candidate contract to build a strategy around.")
        )

    investment = thesis.get("investment_thesis") or {}
    consensus = str(investment.get("consensus", "—"))
    blocks.append(
        _agent_block("Investment Thesis", styles,
                     badge=(consensus.upper(), consensus_tone(consensus)),
                     body=[Paragraph(escape(investment.get("thesis", "")), styles["body"])])
    )
    return blocks


def build_report_pdf(report: dict[str, Any]) -> bytes:
    """Build a PDF from a combined analysis/thesis payload and return its bytes.

    Expected keys (all optional except an implicit best-effort render):
      symbol, generated_at, recommendation, trend, volume, indicators,
      candidates, and thesis (the investment-thesis result, or None if the
      pipeline hasn't been run).
    """
    styles = _styles()
    story: list[Any] = []

    symbol = str(report.get("symbol") or "—")
    generated = report.get("generated_at")
    subtitle = f"{symbol}"
    if generated:
        subtitle += f"  ·  generated {escape(format_timestamp(str(generated)))}"

    story.append(Paragraph("Options Analysis Report", styles["title"]))
    story.append(Paragraph(subtitle, styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=_HAIRLINE, spaceBefore=8, spaceAfter=2))

    recommendation = report.get("recommendation")
    if recommendation:
        story.extend(_section_header("Recommendation", styles))
        story.extend(_recommendation_block(recommendation, report.get("candidates"), styles))

    trend = report.get("trend")
    volume = report.get("volume")
    indicators = report.get("indicators")
    if trend or volume or indicators:
        story.extend(_section_header("Technical snapshot", styles))
        story.extend(_facts_table(technical_snapshot_facts(trend, volume, indicators), styles))

    if report.get("candidates") is not None:
        story.extend(_section_header("Scored candidates", styles))
        story.extend(_candidates_table(report.get("candidates") or [], styles))

    thesis = report.get("thesis")
    if thesis:
        story.extend(_section_header("Agent pipeline", styles))
        warnings = thesis.get("pipeline_warnings") or []
        if warnings:
            story.append(_badge("PIPELINE WARNINGS", "warning", styles))
            story.append(Spacer(1, 3))
            story.extend(_bullets([str(w) for w in warnings], styles))
            story.append(Spacer(1, 4))
        story.extend(_pipeline_blocks(thesis, styles))

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        title=f"Options Analysis Report — {symbol}",
        author="AgenticOptionsReporter",
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    doc.build(story)
    return buffer.getvalue()
