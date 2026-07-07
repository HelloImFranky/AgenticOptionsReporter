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
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from agentic_options_reporter.frontend.formatting import (
    company_health_tone,
    consensus_tone,
    domain_badges,
    domain_id_for_label,
    domain_score_items,
    earnings_surprise_facts,
    format_next_earnings,
    format_timestamp,
    fundamentals_metric_facts,
    insider_activity_header,
    insider_activity_series,
    macro_regime_tone,
    missing_domain_labels,
    recommendation_facts,
    recommendation_tone,
    risk_level_tone,
    score_severity_label,
    score_severity_tone,
    technical_snapshot_facts,
    trade_quality_agreement_summary,
    trade_quality_summary,
    trade_quality_tone,
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
# Hex strings for the same tones, for inline <font color="..."> markup
# inside Paragraph text (colors.Color objects aren't directly usable there).
_TONE_HEX = {
    "success": "#2E7D32",
    "warning": "#B26A00",
    "danger": "#C62828",
    "neutral": "#546E7A",
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
            "AorSection", parent=base, fontName="Helvetica-Bold", fontSize=17,
            leading=21, textColor=_INK, spaceBefore=0, spaceAfter=6,
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


def _section_header(text: str, styles: dict[str, ParagraphStyle], *, new_page: bool = True) -> list[Any]:
    """Puts the section name at the top of its page — each report section
    ('domain') gets its own page, header first, then its data — instead of
    flowing sections together with just a divider line between them.

    `new_page=False` is for the one exception: the FIRST section in the
    report shares the cover page with the title/subtitle rather than
    starting yet another page of its own, so a short report doesn't open
    with a near-empty title page. Every section after that gets a real
    PageBreak."""
    lead: list[Any] = [PageBreak()] if new_page else [Spacer(1, 14)]
    return [
        *lead,
        Paragraph(escape(text), styles["section"]),
        HRFlowable(width="100%", thickness=1, color=_HAIRLINE, spaceAfter=6, spaceBefore=0),
    ]


def _bullets(items: list[str], styles: dict[str, ParagraphStyle]) -> list[Any]:
    return [
        Paragraph(escape(str(item)), styles["bullet"], bulletText="•")
        for item in items
        if str(item).strip()
    ]


def _inline_badges(badges: list[tuple[str, str]]) -> str:
    """Small colored '[Label]' fragments for a Paragraph — the print
    equivalent of the UI's rounded severity/domain-specific pills
    (app.py _domain_score_row). Callers embed this inside a Paragraph
    whose OTHER dynamic text is escaped separately; each badge label here
    is escaped individually so the surrounding <font>/<b> markup survives."""
    parts = []
    for label, tone in badges:
        hex_color = _TONE_HEX.get(tone, _TONE_HEX["neutral"])
        parts.append(f'&nbsp;&nbsp;<font color="{hex_color}"><b>[{escape(label)}]</b></font>')
    return "".join(parts)


def _domain_block_flowables(
    domain_id: str | None,
    label: str,
    score: float,
    confidence: float,
    evidence: list[str],
    factors: list[dict[str, Any]] | None,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """One domain's block: label + severity/domain-specific badge(s), a
    score/confidence meter bar, and evidence — mirrors app.py's
    _domain_score_row exactly (same badge vocabulary via domain_badges,
    same 0-100 meter, same evidence line)."""
    cell_style = styles.get("cell", styles.get("body", getSampleStyleSheet()["BodyText"]))
    muted_style = styles.get("muted", cell_style)
    tone = score_severity_tone(score)
    color = _TONE_COLORS.get(tone, _TONE_COLORS["neutral"])
    badges = domain_badges(domain_id, score, confidence, factors)

    header = Paragraph(f"<b>{escape(label)}</b>{_inline_badges(badges)}", cell_style)

    ratio = max(0.0, min(1.0, score / 100))
    track_width = 1.6 * inch
    bar_height = 7
    filled_width = track_width * ratio
    # A fixed-height meter drawn as a two-column track (filled portion +
    # remainder). The cells are empty strings, not Paragraphs, with zero
    # padding — so a 0% (or tiny) fill leaves a zero-width column with no
    # flowable to wrap, instead of a Paragraph handed a negative available
    # width (the reportlab crash this replaces).
    bar = Table(
        [["", ""]],
        colWidths=[filled_width, track_width - filled_width],
        rowHeights=[bar_height],
        hAlign="LEFT",
    )
    bar.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), color),
                ("BACKGROUND", (1, 0), (1, 0), _TABLE_HEADER_BG),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    meter_row = Table(
        [[bar, Paragraph(f"{score:.0f}/100 · {confidence:.0f}% conf.", muted_style)]],
        colWidths=[track_width + 6, None],
        hAlign="LEFT",
    )
    meter_row.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    flowables: list[Any] = [header, Spacer(1, 2), meter_row]
    if evidence:
        flowables.append(Paragraph(escape(" · ".join(evidence[:2])), muted_style))
    flowables.append(Spacer(1, 5))
    return flowables


def trade_quality_flowables(
    trade_quality: dict[str, Any] | None, styles: dict[str, ParagraphStyle]
) -> list[Any]:
    """Renders a Trade Quality Score (specs/scoring.yaml): a composite
    score/severity/recommendation/confidence heading, then one block per
    domain (present domains as a badge + score/confidence meter + evidence,
    absent domains as a muted 'Not available' line) — the print
    counterpart of app.py's _trade_quality_panel."""
    if not trade_quality:
        return []
    domain_scores = trade_quality.get("domain_scores") or {}
    items = domain_score_items(domain_scores)
    missing = missing_domain_labels(domain_scores)
    if not items and not missing:
        return []

    cell_style = styles.get("cell", styles.get("body", getSampleStyleSheet()["BodyText"]))
    muted_style = styles.get("muted", cell_style)
    heading_style = styles.get("cellhead", cell_style)

    composite = float(trade_quality.get("composite_score") or 0.0)
    confidence = float(trade_quality.get("confidence") or 0.0)
    action = trade_quality.get("recommendation_action", "—")
    severity = score_severity_label(composite)
    heading = (
        f"Trade Quality Score: {composite:.0f}/100 ({action}, {severity}, "
        f"confidence {confidence:.0f}%)"
    )
    flowables: list[Any] = [Paragraph(escape(heading), heading_style), Spacer(1, 3)]

    summary = trade_quality_summary(trade_quality)
    if summary:
        flowables.append(Paragraph(escape(summary), muted_style))
        flowables.append(Spacer(1, 3))

    for label, score, confidence_pct, evidence in items:
        domain_id = domain_id_for_label(label)
        factors = (domain_scores.get(domain_id) or {}).get("factors") if domain_id else None
        flowables.extend(
            _domain_block_flowables(domain_id, label, score, confidence_pct, evidence, factors, styles)
        )
    for label in missing:
        flowables.append(Paragraph(f"{escape(label)}: <i>Not available</i>", muted_style))
        flowables.append(Spacer(1, 3))

    return flowables


def trade_quality_comparison_flowables(
    quant: dict[str, Any] | None, agent: dict[str, Any] | None, styles: dict[str, ParagraphStyle]
) -> list[Any]:
    """Mirrors the Agents-tab 'Trade Quality Score — Quant vs. Agents'
    comparison card (app.py _render_final): an agreement/divergence
    caption, then the quant and agent Trade Quality Score side by side."""
    if not quant and not agent:
        return []
    muted_style = styles.get("muted", styles["body"])
    cellhead_style = styles.get("cellhead", styles["body"])

    flowables: list[Any] = []
    agreement = trade_quality_agreement_summary(quant, agent)
    if agreement:
        flowables.append(Paragraph(escape(agreement), muted_style))
        flowables.append(Spacer(1, 6))

    quant_col: list[Any] = [Paragraph("Quant", cellhead_style), Spacer(1, 3)]
    quant_col.extend(
        trade_quality_flowables(quant, styles)
        if quant
        else [Paragraph("No Trade Quality Score available", muted_style)]
    )
    agent_col: list[Any] = [Paragraph("Agents", cellhead_style), Spacer(1, 3)]
    agent_col.extend(
        trade_quality_flowables(agent, styles)
        if agent
        else [Paragraph("No Trade Quality Score available", muted_style)]
    )

    table = Table([[quant_col, agent_col]], colWidths=[3.25 * inch, 3.25 * inch], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("LEFTPADDING", (1, 0), (1, 0), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    flowables.append(table)
    return flowables


def _recommendation_block(
    rec: dict[str, Any],
    candidates: list[dict[str, Any]] | None,
    trade_quality: dict[str, Any] | None,
    styles: dict[str, ParagraphStyle],
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

    if trade_quality:
        block.append(Spacer(1, 6))
        block.extend(trade_quality_flowables(trade_quality, styles))
        # Caption the chart with the composite engine's own explainability
        # bullets, and drop the deterministic rationale — that's the same
        # factors restated as text, now redundant with the visualization above.
        summary = trade_quality_summary(trade_quality)
        if summary:
            block.append(Paragraph(escape(summary), styles["muted"]))
    elif rationale:
        # No Trade Quality Score to visualize (e.g. AVOID / no candidate) —
        # the rationale is the only explanation, so keep it.
        block.append(Spacer(1, 6))
        block.append(Paragraph(escape(rationale), styles["body"]))
    return block


def _short_date(value: str) -> str:
    """'2026-06-01' -> '06/01' for compact time-axis ticks."""
    parts = str(value).split("-")
    return f"{parts[1]}/{parts[2]}" if len(parts) == 3 else str(value)


def _insider_timeseries_chart(insider: dict[str, Any] | None, styles: dict[str, ParagraphStyle]) -> list[Any]:
    """Time series of net insider share flow: a signed column per date drawn
    on a zero baseline — green above the line for net buying, red below for
    net selling. Print counterpart of the Analyze tab's chart.

    Vertical layout, bottom to top, all FIXED (not bar-height-dependent) so
    a full-height column's value label can never collide with the date row
    below it:
      dates row -> gap -> value-label row -> gap -> bar plot area.
    """
    series = insider_activity_series(insider)
    if not series:
        return []

    width = 6.6 * inch
    date_row_y = 4              # date-tick text baseline
    value_label_y = 17          # sell-side peak value label baseline (fixed)
    bar_bottom_gap = 6          # clearance between the lowest possible bar and the value label
    label_band = value_label_y + 7 + bar_bottom_gap  # ~30pt reserved below the baseline
    plot_half = 46               # column reach above / below the baseline
    height = label_band + 2 * plot_half + 12
    baseline_y = label_band + plot_half

    # Reserve room on the left for the y-axis's tick labels (share counts),
    # so the magnitude scale is visible instead of only inferable from bar height.
    axis_label_width = 38
    left_pad, right_pad = axis_label_width, 6
    plot_width = width - left_pad - right_pad
    step = plot_width / len(series)
    col_width = min(step * 0.5, 16)
    max_mag = max(abs(p["net"]) for p in series) or 1.0
    peak = max(range(len(series)), key=lambda i: abs(series[i]["net"]))

    drawing = Drawing(width, height)

    # Y-axis: a vertical rule plus tick marks + share-count labels at
    # 0/±50%/±100% of the largest magnitude, so the scale reads at a
    # glance instead of only via the single peak-value label.
    axis_x = left_pad - 6
    drawing.add(Line(axis_x, 4, axis_x, height - 4, strokeColor=_HAIRLINE, strokeWidth=0.75))
    for frac in (1.0, 0.5, 0.0, -0.5, -1.0):
        tick_y = baseline_y + frac * plot_half
        drawing.add(Line(axis_x - 3, tick_y, axis_x, tick_y, strokeColor=_HAIRLINE, strokeWidth=0.75))
        tick_label = f"{frac * max_mag:+,.0f}" if frac else "0"
        drawing.add(
            String(axis_x - 4, tick_y - 2.5, tick_label, fontName="Helvetica", fontSize=6,
                   fillColor=_MUTED, textAnchor="end")
        )

    # Recessive zero baseline across the plot area.
    drawing.add(Line(left_pad, baseline_y, width - right_pad, baseline_y, strokeColor=_HAIRLINE, strokeWidth=0.75))

    for i, p in enumerate(series):
        cx = left_pad + step * i + step / 2
        col_h = (abs(p["net"]) / max_mag) * plot_half
        color = _TONE_COLORS["success"] if p["is_buy"] else _TONE_COLORS["danger"]
        y = baseline_y if p["is_buy"] else baseline_y - col_h
        drawing.add(
            Rect(cx - col_width / 2, y, col_width, col_h, rx=2, ry=2, fillColor=color, strokeColor=None)
        )
        drawing.add(
            String(cx, date_row_y, _short_date(p["date"]), fontName="Helvetica", fontSize=6.5,
                   fillColor=_MUTED, textAnchor="middle")
        )
        # Selective direct label: only the largest-magnitude column. The
        # sell-side (negative) label sits at a FIXED y — below the lowest
        # point any bar can reach, with its own gap above the date row —
        # so it never touches the dates, regardless of that bar's height.
        if i == peak:
            ly = baseline_y + col_h + 3 if p["is_buy"] else value_label_y
            drawing.add(
                String(cx, ly, f"{p['net']:+,.0f}", fontName="Helvetica-Bold", fontSize=6.5,
                       fillColor=color, textAnchor="middle")
            )

    legend = Paragraph(
        '<font color="#2E7D32">■</font> Buy &nbsp;&nbsp; '
        '<font color="#C62828">■</font> Sell',
        styles.get("cellmuted", styles["body"]),
    )
    return [drawing, Spacer(1, 2), legend]


def _fundamentals_blocks(
    fundamentals: dict[str, Any], data_warnings: list[Any] | None, styles: dict[str, ParagraphStyle]
) -> list[Any]:
    """The cross-provider fundamentals snapshot (metrics, next earnings,
    recent surprises, insider activity) — mirrors the Analyze tab's
    Fundamentals card, sharing the same formatting helpers so the two can't
    drift. Absent sections are simply omitted."""
    blocks: list[Any] = []

    metric_facts = fundamentals_metric_facts(fundamentals.get("metrics"))
    if metric_facts:
        blocks.append(Paragraph("Key metrics", styles["agent"]))
        blocks.extend(_facts_table(metric_facts, styles, cols=3))

    next_earnings = format_next_earnings(fundamentals.get("earnings_calendar"))
    if next_earnings:
        blocks.append(Spacer(1, 4))
        blocks.append(Paragraph(escape(next_earnings), styles["body"]))

    surprise_facts = earnings_surprise_facts(fundamentals.get("earnings_history"))
    if surprise_facts:
        blocks.append(Spacer(1, 4))
        blocks.append(Paragraph("Recent earnings (actual vs. estimate)", styles["agent"]))
        blocks.extend(_facts_table(surprise_facts, styles, cols=2))

    insider_header = insider_activity_header(fundamentals.get("insider_activity"))
    if insider_header:
        blocks.append(Spacer(1, 4))
        blocks.append(Paragraph(escape(insider_header), styles["agent"]))
        blocks.extend(_insider_timeseries_chart(fundamentals.get("insider_activity"), styles))

    if data_warnings:
        blocks.append(Spacer(1, 4))
        blocks.append(
            Paragraph(
                "Some sources were unavailable: "
                + escape("; ".join(str(w) for w in data_warnings)),
                styles["muted"],
            )
        )

    if not blocks:
        return [Paragraph("No fundamentals available for this symbol.", styles["muted"])]
    return blocks


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
    q_trade_quality = quant.get("quant_trade_quality") or {}
    q_score = q_trade_quality.get("composite_score") or 0.0
    q_body = [Paragraph(escape(quant.get("narrative", "")), styles["body"])]
    factors = quant.get("key_factors") or []
    if factors:
        q_body.append(Paragraph("<b>Key factors:</b> " + escape(" · ".join(factors)), styles["body"]))
    blocks.append(
        _agent_block(
            "Quant Interpreter", styles,
            badge=(f"SCORE {q_score:.0f}/100", trade_quality_tone(q_score)), body=q_body,
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

    relative_strength = thesis.get("relative_strength_research")
    if relative_strength is not None:
        blocks.append(
            _agent_block(
                "Relative Strength Research", styles,
                body=[Paragraph(escape(relative_strength.get("narrative", "")), styles["body"])],
            )
        )
    else:
        blocks.append(
            _agent_block("Relative Strength Research", styles,
                         skipped="Skipped — no candidate contract to assess.")
        )

    statistical_edge = thesis.get("statistical_edge_research")
    if statistical_edge is not None:
        blocks.append(
            _agent_block(
                "Statistical Edge Research", styles,
                body=[Paragraph(escape(statistical_edge.get("narrative", "")), styles["body"])],
            )
        )
    else:
        blocks.append(
            _agent_block("Statistical Edge Research", styles,
                         skipped="Skipped — no candidate contract to assess.")
        )

    investment = thesis.get("investment_thesis") or {}
    consensus = str(investment.get("consensus", "—"))
    blocks.append(
        _agent_block("Investment Thesis", styles,
                     badge=(consensus.upper(), consensus_tone(consensus)),
                     body=[Paragraph(escape(investment.get("thesis", "")), styles["body"])])
    )

    # The agent-side Trade Quality Score is no longer shown here in
    # isolation — it renders in the dedicated "Trade Quality Score — Quant
    # vs. Agents" section (build_report_pdf), side by side with the quant
    # score, mirroring the Agents-tab comparison card.
    return blocks


def build_report_pdf(report: dict[str, Any]) -> bytes:
    """Build a PDF from a combined analysis/thesis payload and return its bytes.

    Expected keys (all optional except an implicit best-effort render):
      symbol, generated_at, recommendation, trend, volume, indicators,
      candidates, fundamentals (the cross-provider snapshot, or None),
      data_warnings, and thesis (the investment-thesis result, or None if
      the pipeline hasn't been run).
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

    # The title isn't a section itself, so it follows different rules: the
    # FIRST section shares its page with the title/subtitle above (no
    # PageBreak), and every section after that opens its own fresh page.
    on_first_section = True

    def add_section(text: str) -> None:
        nonlocal on_first_section
        story.extend(_section_header(text, styles, new_page=not on_first_section))
        on_first_section = False

    recommendation = report.get("recommendation")
    if recommendation:
        add_section("Recommendation")
        story.extend(
            _recommendation_block(
                recommendation, report.get("candidates"), report.get("trade_quality"), styles
            )
        )

    trend = report.get("trend")
    volume = report.get("volume")
    indicators = report.get("indicators")
    if trend or volume or indicators:
        add_section("Technical snapshot")
        story.extend(_facts_table(technical_snapshot_facts(trend, volume, indicators), styles))

    if report.get("candidates") is not None:
        add_section("Scored candidates")
        story.extend(_candidates_table(report.get("candidates") or [], styles))

    fundamentals = report.get("fundamentals")
    if fundamentals:
        add_section("Fundamentals")
        story.extend(_fundamentals_blocks(fundamentals, report.get("data_warnings"), styles))

    thesis = report.get("thesis")
    if thesis:
        # Mirrors the Agents-tab layout order: the Trade Quality Score
        # comparison (final_output_card) appears before the per-agent
        # conversation transcript (conversation_card).
        quant_trade_quality = (
            (thesis.get("quant_interpretation") or {}).get("quant_trade_quality")
            or report.get("trade_quality")
        )
        agent_trade_quality = thesis.get("agent_trade_quality")
        comparison = trade_quality_comparison_flowables(quant_trade_quality, agent_trade_quality, styles)
        if comparison:
            add_section("Trade Quality Score — Quant vs. Agents")
            story.extend(comparison)

        add_section("Agent pipeline")
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
