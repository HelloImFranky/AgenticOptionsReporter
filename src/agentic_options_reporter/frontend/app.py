"""Flet front end for AgenticOptionsReporter.

A Material 3 UI over `agentic_options_reporter.api_client.ApiClient` — the
same HTTP client the CLI uses. All analysis logic runs server-side; this
module only requests data (per specs/api.yaml) and renders it. Display
formatting lives in formatting.py so it can be unit tested without a Flet
runtime.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import time

import flet as ft

from agentic_options_reporter.api_client import DEFAULT_BASE_URL, ApiClient, ApiError
from agentic_options_reporter.frontend.formatting import (
    CANDIDATE_COLUMNS,
    RUN_COLUMNS,
    candidates_to_rows,
    cash_flow_tone,
    company_health_tone,
    consensus_tone,
    domain_badges,
    domain_id_for_label,
    domain_score_items,
    format_indicator_summary,
    format_next_earnings,
    format_num,
    format_pct,
    format_trend_summary,
    format_volume_summary,
    fundamentals_metric_facts,
    growth_tone,
    insider_activity_header,
    insider_activity_series,
    macro_regime_tone,
    missing_domain_labels,
    profitability_tone,
    recommendation_facts,
    recommendation_tone,
    risk_level_tone,
    runs_to_rows,
    score_severity_label,
    score_severity_tone,
    trade_quality_agreement_summary,
    trade_quality_summary,
    trade_quality_tone,
    trend_tone,
)
from agentic_options_reporter.frontend.report_pdf import build_report_pdf

_SEED_COLOR = ft.Colors.INDIGO

_TONE_COLORS = {
    "success": (ft.Colors.GREEN_700, ft.Colors.GREEN_50),
    "warning": (ft.Colors.AMBER_800, ft.Colors.AMBER_50),
    "danger": (ft.Colors.RED_700, ft.Colors.RED_50),
    "neutral": (ft.Colors.GREY_700, ft.Colors.GREY_200),
}

_TREND_ICONS = {
    "success": ft.Icons.TRENDING_UP,
    "danger": ft.Icons.TRENDING_DOWN,
    "neutral": ft.Icons.TRENDING_FLAT,
}


def _tone_colors(tone: str) -> tuple[str, str]:
    return _TONE_COLORS.get(tone, _TONE_COLORS["neutral"])


def _pill(text: str, tone: str) -> ft.Container:
    color, _ = _tone_colors(tone)
    return ft.Container(
        content=ft.Text(text, size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
        bgcolor=color,
        border_radius=20,
        padding=ft.padding.symmetric(vertical=4, horizontal=12),
    )


def _chip(text: str) -> ft.Container:
    return ft.Container(
        content=ft.Text(text, size=12, color=ft.Colors.ON_SURFACE_VARIANT),
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        border_radius=8,
        padding=ft.padding.symmetric(vertical=4, horizontal=10),
    )


def _toned_chip(text: str, tone: str) -> ft.Container:
    """A chip carrying semantic tone colour — a solid tinted pill with white
    text, so a finding's value (e.g. Growth: accelerating) reads its own
    tone the way the other agent sections' badges do."""
    color, _ = _tone_colors(tone)
    return ft.Container(
        content=ft.Text(text, size=12, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
        bgcolor=color,
        border_radius=8,
        padding=ft.padding.symmetric(vertical=4, horizontal=10),
    )


def _fill_pill(badge: ft.Container, text: str, tone: str) -> None:
    """Populate a placeholder container as a solid tone-coloured headline
    pill, matching the leading badges used across the agent sections."""
    badge.content = ft.Text(text, size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE)
    badge.bgcolor = _tone_colors(tone)[0]
    badge.border_radius = 20
    badge.padding = ft.padding.symmetric(vertical=3, horizontal=10)
    badge.visible = True


def _bullet_list(items: list[str]) -> ft.Column:
    return ft.Column(
        [
            ft.Row(
                [ft.Text("•", size=12, color=ft.Colors.ON_SURFACE_VARIANT), ft.Text(item, size=12, expand=True)],
                spacing=6,
            )
            for item in items
        ],
        spacing=4,
        tight=True,
    )


_CATALYST_DIRECTION_TONES = {"bullish": "success", "bearish": "danger", "uncertain": "neutral"}


def _catalyst_entry(item: dict) -> ft.Column:
    """One catalyst row: a direction-toned badge, the title, a muted
    category · horizon label, and the detail."""
    direction = item.get("direction", "uncertain")
    tone = _CATALYST_DIRECTION_TONES.get(direction, "neutral")
    badge = ft.Container(
        content=ft.Text(direction.upper(), size=9, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
        bgcolor=_tone_colors(tone)[0],
        border_radius=12,
        padding=ft.padding.symmetric(vertical=2, horizontal=8),
    )
    header = ft.Row(
        [badge, ft.Text(item.get("title", ""), size=12, weight=ft.FontWeight.W_600, expand=True)],
        spacing=6,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    meta = ft.Text(
        f"{item.get('category', '—')} · {item.get('horizon', '—')}".replace("_", " "),
        size=10,
        color=ft.Colors.ON_SURFACE_VARIANT,
    )
    controls = [header, meta]
    if item.get("detail"):
        controls.append(ft.Text(item["detail"], size=11, selectable=True))
    return ft.Column(controls, spacing=2, tight=True)


def _agent_message(name: str, icon: str, color: str, *body: ft.Control) -> ft.Row:
    avatar = ft.Container(
        content=ft.Icon(icon, size=16, color=ft.Colors.WHITE),
        bgcolor=color,
        border_radius=100,
        width=30,
        height=30,
        alignment=ft.alignment.center,
    )
    return ft.Row(
        [
            avatar,
            ft.Column(
                [ft.Text(name, size=13, weight=ft.FontWeight.BOLD), *body],
                spacing=4,
                tight=True,
                expand=True,
            ),
        ],
        spacing=12,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )


def _skipped_message(reason: str) -> ft.Text:
    return ft.Text(reason, size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)


def _hood_block(label: str):
    """One labelled block (System prompt / User prompt / Raw response) inside
    the under-the-hood panel. Returns (column, set_value)."""
    body = ft.Text("", size=11, selectable=True, color=ft.Colors.ON_SURFACE_VARIANT)
    container = ft.Container(
        content=body,
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        border_radius=8,
        padding=10,
    )
    column = ft.Column(
        [ft.Text(label, size=10, weight=ft.FontWeight.BOLD, color=ft.Colors.ON_SURFACE_VARIANT), container],
        spacing=4,
        tight=True,
    )

    def set_value(text: str) -> None:
        body.value = text

    return column, set_value


_STATUS_STYLE = {
    "queued": (ft.Colors.ON_SURFACE_VARIANT, ft.Colors.SURFACE_CONTAINER_HIGHEST, "Queued"),
    "running": (ft.Colors.BLUE_700, ft.Colors.BLUE_50, "Running…"),
    "done": (ft.Colors.GREEN_700, ft.Colors.GREEN_50, "Done"),
    "skipped": (ft.Colors.ON_SURFACE_VARIANT, ft.Colors.SURFACE_CONTAINER_HIGHEST, "Skipped"),
    "failed": (ft.Colors.RED_700, ft.Colors.RED_50, "Failed"),
}


def _status_pill():
    """A small status chip for one agent. Returns a Container; use
    `_set_status(pill, phase)` to update it live. Hidden until set."""
    return ft.Container(
        content=ft.Text("", size=10, weight=ft.FontWeight.BOLD),
        padding=ft.padding.symmetric(horizontal=8, vertical=2),
        border_radius=100,
        visible=False,
    )


def _set_status(pill: ft.Container, status: str) -> None:
    color, bgcolor, label = _STATUS_STYLE[status]
    pill.content.value = label
    pill.content.color = color
    pill.bgcolor = bgcolor
    pill.visible = True


def _hood():
    """A collapsible "Under the hood" panel showing the raw system prompt,
    user prompt, and model response for an agent. Returns (tile, fill) where
    `fill(exchange)` populates it from an AgentExchange dict (or hides the
    tile when the agent had no LLM exchange)."""
    system_block, set_system = _hood_block("System prompt")
    user_block, set_user = _hood_block("User prompt")
    response_block, set_response = _hood_block("Raw response")
    tile = ft.ExpansionTile(
        title=ft.Text("Under the hood", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
        controls=[
            ft.Container(
                content=ft.Column([system_block, user_block, response_block], spacing=10, tight=True),
                padding=ft.padding.only(left=8, right=8, bottom=8),
            )
        ],
        visible=False,
        dense=True,
        controls_padding=0,
    )

    def fill(exchange: dict | None) -> None:
        if not exchange:
            tile.visible = False
            return
        set_system(exchange.get("system_prompt", "") or "")
        set_user(exchange.get("user_prompt", "") or "")
        set_response(exchange.get("raw_response", "") or "")
        tile.visible = True

    return tile, fill


def _card(*controls: ft.Control, padding: int = 20, spacing: int = 12) -> ft.Card:
    return ft.Card(
        elevation=1,
        content=ft.Container(
            padding=padding,
            border_radius=16,
            content=ft.Column(list(controls), spacing=spacing, tight=True),
        ),
    )


def _domain_score_row(
    label: str, score: float, confidence: float, evidence: list[str], badges: list[tuple[str, str]]
) -> ft.Column:
    ratio = max(0.0, min(1.0, score / 100))
    tone = score_severity_tone(score)
    color, _ = _tone_colors(tone)

    badge_pills: list[ft.Control] = []
    for badge_label, badge_tone in badges:
        badge_color, _ = _tone_colors(badge_tone)
        badge_pills.append(
            ft.Container(
                content=ft.Text(badge_label, size=10, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                bgcolor=badge_color,
                border_radius=12,
                padding=ft.padding.symmetric(vertical=2, horizontal=8),
            )
        )

    return ft.Column(
        [
            ft.Row(
                [ft.Text(label, size=11, weight=ft.FontWeight.W_600), ft.Row(badge_pills, spacing=6, wrap=True)],
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.Row(
                [
                    ft.ProgressBar(value=ratio, width=220, color=color, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST),
                    ft.Text(
                        f"{score:.0f}/100 · {confidence:.0f}% conf.",
                        size=11,
                        color=ft.Colors.ON_SURFACE_VARIANT,
                    ),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.Text(
                " · ".join(evidence[:2]), size=10, color=ft.Colors.ON_SURFACE_VARIANT
            ) if evidence else ft.Container(height=0),
        ],
        spacing=4,
        tight=True,
    )


def _missing_domain_row(label: str) -> ft.Row:
    return ft.Row(
        [
            ft.Text(label, size=11, weight=ft.FontWeight.W_600, color=ft.Colors.ON_SURFACE_VARIANT, expand=True),
            ft.Text("Not available", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
        ],
        spacing=8,
    )


def _trade_quality_panel(trade_quality: dict | None) -> ft.Container:
    """Renders a Trade Quality Score (either source="quant" or "agent") —
    composite score + confidence + recommendation, then one row per
    domain (present domains as a score/confidence/evidence bar, absent
    domains as a muted 'Not available' row rather than a fabricated 0)."""
    if not trade_quality:
        return ft.Container(
            content=ft.Text("No Trade Quality Score available", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=10,
            padding=12,
        )

    domain_scores = trade_quality.get("domain_scores") or {}
    composite = float(trade_quality.get("composite_score") or 0.0)
    confidence = float(trade_quality.get("confidence") or 0.0)
    action = trade_quality.get("recommendation_action", "—")
    tone = trade_quality_tone(composite)
    color, _ = _tone_colors(tone)

    rows: list[ft.Control] = []
    for label, score, conf, evidence in domain_score_items(domain_scores):
        domain_id = domain_id_for_label(label)
        factors = (domain_scores.get(domain_id) or {}).get("factors") if domain_id else None
        badges = domain_badges(domain_id, score, conf, factors)
        rows.append(_domain_score_row(label, score, conf, evidence, badges))
    rows.extend(_missing_domain_row(label) for label in missing_domain_labels(domain_scores))

    summary = trade_quality_summary(trade_quality)
    overall_label = score_severity_label(composite)

    return ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Container(
                            content=ft.Text(f"{composite:.0f}/100", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                            bgcolor=color,
                            border_radius=16,
                            padding=ft.padding.symmetric(vertical=4, horizontal=12),
                        ),
                        ft.Column(
                            [
                                ft.Row(
                                    [
                                        ft.Text(f"Trade Quality Score · {action}", size=12, weight=ft.FontWeight.W_600),
                                        ft.Container(
                                            content=ft.Text(overall_label, size=10, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                                            bgcolor=color,
                                            border_radius=12,
                                            padding=ft.padding.symmetric(vertical=2, horizontal=8),
                                        ),
                                    ],
                                    spacing=8,
                                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                ),
                                ft.Text(f"Confidence {confidence:.0f}%", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                            ],
                            spacing=2,
                            tight=True,
                        ),
                    ],
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Text(summary, size=11, color=ft.Colors.ON_SURFACE_VARIANT) if summary else ft.Container(height=0),
                ft.Column(rows, spacing=8, tight=True),
            ],
            spacing=10,
            tight=True,
        ),
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        border_radius=10,
        padding=12,
    )


def _fact_box(label: str, value: str) -> ft.Container:
    """A small labelled box (muted caption over a bold value) for laying out
    recommendation/technical key facts as a tidy grid instead of a run-on
    sentence."""
    return ft.Container(
        content=ft.Column(
            [
                ft.Text(label.upper(), size=10, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text(value, size=14, weight=ft.FontWeight.W_600, selectable=True),
            ],
            spacing=3,
            tight=True,
        ),
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        border_radius=10,
        padding=ft.padding.symmetric(vertical=10, horizontal=12),
        col={"xs": 6, "sm": 4, "md": 3},
    )


def _legend_swatch(color: str, label: str) -> ft.Row:
    return ft.Row(
        [
            ft.Container(width=10, height=10, bgcolor=color, border_radius=2),
            ft.Text(label, size=10, color=ft.Colors.ON_SURFACE_VARIANT),
        ],
        spacing=4,
        tight=True,
    )


def _short_date(value: str) -> str:
    """'2026-06-01' -> '06/01' for compact time-axis ticks."""
    parts = str(value).split("-")
    return f"{parts[1]}/{parts[2]}" if len(parts) == 3 else str(value)


def _insider_timeseries_chart(series: list[dict]) -> ft.BarChart:
    """Time series of net insider share flow: one signed column per date,
    green above the zero line for net buying, red below for net selling —
    so direction reads from position as well as colour."""
    max_mag = max((abs(p.get("net", 0.0)) for p in series), default=0.0) or 1.0
    groups: list[ft.BarChartGroup] = []
    labels: list[ft.ChartAxisLabel] = []
    for i, p in enumerate(series):
        tone = ft.Colors.GREEN_600 if p.get("is_buy") else ft.Colors.RED_600
        net = float(p.get("net", 0.0))
        groups.append(
            ft.BarChartGroup(
                x=i,
                bar_rods=[
                    ft.BarChartRod(
                        from_y=0,
                        to_y=net,
                        width=14,
                        color=tone,
                        border_radius=2,
                        tooltip=f"{p.get('date', '')}\n{net:+,.0f} shares",
                    )
                ],
            )
        )
        labels.append(
            ft.ChartAxisLabel(
                value=i,
                label=ft.Container(
                    ft.Text(_short_date(p.get("date", "")), size=9,
                            color=ft.Colors.ON_SURFACE_VARIANT),
                    padding=ft.padding.only(top=6),
                ),
            )
        )
    return ft.BarChart(
        bar_groups=groups,
        bottom_axis=ft.ChartAxis(labels=labels, labels_size=30),
        min_y=-max_mag * 1.15,
        max_y=max_mag * 1.15,
        # Recessive gridlines land one on the zero baseline (interval=max_mag).
        horizontal_grid_lines=ft.ChartGridLines(
            interval=max_mag, color=ft.Colors.OUTLINE_VARIANT, width=0.5
        ),
        interactive=True,
        # A fixed height only — NO expand: this chart lives inside the
        # scrolling results column, and an expanding child in an
        # unbounded-height parent throws Flutter's "unbounded height" error,
        # which blanks the whole result (recommendation/candidates included).
        height=180,
    )


def _fundamentals_controls(fundamentals: dict | None, warnings: list | None) -> list[ft.Control]:
    """Render the cross-provider fundamentals snapshot (metrics, earnings,
    calendar, insider activity) surfaced by /analyze into a compact set of
    controls. Absent sections are simply omitted."""
    if not fundamentals:
        return [
            ft.Text(
                "No fundamentals available for this symbol.",
                size=12,
                italic=True,
                color=ft.Colors.ON_SURFACE_VARIANT,
            )
        ]

    controls: list[ft.Control] = []

    shown = fundamentals_metric_facts(fundamentals.get("metrics"))
    if shown:
        controls.append(ft.Text("Key metrics", size=12, weight=ft.FontWeight.BOLD))
        controls.append(ft.ResponsiveRow([_fact_box(k, v) for k, v in shown], spacing=8, run_spacing=8))

    next_earnings = format_next_earnings(fundamentals.get("earnings_calendar"))
    if next_earnings:
        controls.append(
            ft.Row(
                [ft.Icon(ft.Icons.EVENT_OUTLINED, size=16, color=ft.Colors.AMBER_800),
                 ft.Text(next_earnings, size=12, selectable=True)],
                spacing=8,
            )
        )

    earnings = fundamentals.get("earnings_history")
    surprises = (earnings or {}).get("surprises") or []
    if surprises:
        controls.append(ft.Text("Recent earnings (actual vs. estimate)", size=12, weight=ft.FontWeight.BOLD))
        rows = []
        for s in surprises[:4]:
            actual, estimate = s.get("actual_eps"), s.get("estimate_eps")
            pct = s.get("surprise_percent")
            beat = isinstance(pct, (int, float)) and pct >= 0
            tone_color = ft.Colors.GREEN_700 if beat else ft.Colors.RED_600
            pct_text = format_pct(pct) if isinstance(pct, (int, float)) else "—"
            rows.append(
                ft.Row(
                    [
                        ft.Text(str(s.get("period", "")), size=11, expand=True, selectable=True),
                        ft.Text(f"{format_num(actual)} / {format_num(estimate)}", size=11,
                                color=ft.Colors.ON_SURFACE_VARIANT),
                        ft.Text(pct_text, size=11, weight=ft.FontWeight.W_600, color=tone_color),
                    ],
                    spacing=10,
                )
            )
        controls.append(ft.Column(rows, spacing=2, tight=True))

    insider = fundamentals.get("insider_activity")
    insider_header = insider_activity_header(insider)
    if insider_header:
        controls.append(ft.Text(insider_header, size=12, weight=ft.FontWeight.BOLD))
        series = insider_activity_series(insider)
        if series:
            controls.append(_insider_timeseries_chart(series))
            controls.append(
                ft.Row(
                    [
                        _legend_swatch(ft.Colors.GREEN_600, "Buy"),
                        _legend_swatch(ft.Colors.RED_600, "Sell"),
                    ],
                    spacing=14,
                )
            )

    if warnings:
        controls.append(
            ft.Text(
                "Some sources were unavailable: " + "; ".join(str(w) for w in warnings),
                size=11,
                italic=True,
                color=ft.Colors.AMBER_800,
                selectable=True,
            )
        )

    if not controls:
        return [
            ft.Text(
                "No fundamentals available for this symbol.",
                size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT,
            )
        ]
    return controls


def _section_title(text: str, icon: str | None = None) -> ft.Row:
    controls: list[ft.Control] = []
    if icon:
        controls.append(ft.Icon(icon, size=18, color=ft.Colors.PRIMARY))
    controls.append(ft.Text(text, size=15, weight=ft.FontWeight.W_600))
    return ft.Row(controls, spacing=8)


def _stat_card(icon: str, label: str, value: ft.Text, icon_color: str | None = None) -> ft.Card:
    return ft.Card(
        elevation=0,
        content=ft.Container(
            padding=16,
            border_radius=14,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            content=ft.Row(
                [
                    ft.Container(
                        content=ft.Icon(icon, size=22, color=icon_color or ft.Colors.PRIMARY),
                        bgcolor=ft.Colors.with_opacity(0.12, icon_color or ft.Colors.PRIMARY),
                        border_radius=10,
                        padding=8,
                    ),
                    ft.Column(
                        [ft.Text(label, size=11, color=ft.Colors.ON_SURFACE_VARIANT), value],
                        spacing=2,
                        tight=True,
                        expand=True,
                    ),
                ],
                spacing=12,
            ),
        ),
        col={"xs": 12, "sm": 4},
    )


def build_view(page: ft.Page, client: ApiClient, reports_dir: str | None = None) -> None:
    page.title = "AgenticOptionsReporter"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.theme = ft.Theme(color_scheme_seed=_SEED_COLOR, use_material3=True)
    page.dark_theme = ft.Theme(color_scheme_seed=_SEED_COLOR, use_material3=True)
    page.padding = 0
    page.bgcolor = ft.Colors.SURFACE

    # ---- inputs -----------------------------------------------------
    symbol_field = ft.TextField(
        label="Symbol", value="AAPL", width=140, border_radius=10, text_size=14
    )
    lookback_field = ft.TextField(
        label="Lookback days", value="365", width=140, border_radius=10, text_size=14
    )
    expiration_field = ft.TextField(
        label="Expiration (optional)", hint_text="YYYY-MM-DD", width=200,
        border_radius=10, text_size=14,
    )
    weighting_profile_dropdown = ft.Dropdown(
        label="Weighting profile",
        value="swing",
        width=170,
        border_radius=10,
        text_size=14,
        options=[
            ft.dropdown.Option("day_trade", "Day trade"),
            ft.dropdown.Option("swing", "Swing"),
            ft.dropdown.Option("long_term", "Long term"),
        ],
    )

    current_run_id: dict[str, int | None] = {"value": None}
    last_recommendation: dict[str, object] = {"action": "—", "confidence": 0.0}
    # Raw payloads retained for the PDF export: the last analysis result and,
    # once generated, the investment-thesis result. Rebuilt on each run.
    report_state: dict[str, object | None] = {"analysis": None, "thesis": None}

    progress = ft.ProgressRing(visible=False, width=18, height=18, stroke_width=2)
    analyze_button = ft.ElevatedButton(
        "Analyze",
        icon=ft.Icons.SEARCH,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
    )

    error_banner = ft.Container(
        visible=False,
        bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.RED),
        border_radius=10,
        padding=12,
        content=ft.Row(
            [
                ft.Icon(ft.Icons.ERROR_OUTLINE, color=ft.Colors.RED_700, size=18),
                ft.Text("", color=ft.Colors.RED_700, size=13, expand=True),
            ],
            spacing=8,
        ),
    )

    def set_error(message: str) -> None:
        error_banner.content.controls[1].value = message
        error_banner.visible = bool(message)

    # ---- results: run issues (data warnings + render problems) --------
    # Sits between "Run analysis" and "Recommendation": surfaces anything
    # that went sideways for a request that otherwise still returned a
    # result — a provider timing out or rate-limiting (data_warnings from
    # the /analyze response) or a problem rendering the payload itself. A
    # hard request failure (network error, non-2xx) still goes to
    # error_banner above, since there's no result to show at all there.
    analysis_warnings_column = ft.Column([], spacing=4, tight=True)
    analysis_warnings_banner = ft.Container(
        visible=False,
        bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.AMBER),
        border_radius=10,
        padding=12,
        content=ft.Row(
            [
                ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color=ft.Colors.AMBER_800, size=18),
                analysis_warnings_column,
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.START,
        ),
    )

    def set_analysis_warnings(messages: list[str]) -> None:
        analysis_warnings_column.controls = [
            ft.Text(message, color=ft.Colors.AMBER_800, size=13, selectable=True)
            for message in messages
        ]
        analysis_warnings_banner.visible = bool(messages)

    # ---- results: recommendation card -------------------------------
    action_badge = ft.Container(
        content=ft.Text("—", size=13, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
        bgcolor=ft.Colors.GREY_400,
        border_radius=20,
        padding=ft.padding.symmetric(vertical=6, horizontal=14),
    )
    confidence_bar = ft.ProgressBar(value=0, width=160, border_radius=6, bgcolor=ft.Colors.GREY_200)
    confidence_text = ft.Text("0%", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
    trade_quality_panel = ft.Container()
    rec_facts_grid = ft.ResponsiveRow([], spacing=10, run_spacing=10)

    recommendation_card = _card(
        _section_title("Recommendation", ft.Icons.INSIGHTS_OUTLINED),
        ft.Row(
            [action_badge, ft.Column([confidence_bar, confidence_text], spacing=2)],
            spacing=16,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        rec_facts_grid,
        trade_quality_panel,
    )

    # ---- results: stat cards -----------------------------------------
    trend_value = ft.Text("Run an analysis", size=13, weight=ft.FontWeight.W_600)
    volume_value = ft.Text("—", size=13, weight=ft.FontWeight.W_600)
    indicators_value = ft.Text("—", size=13, weight=ft.FontWeight.W_600)

    trend_stat_container = ft.Container(
        content=ft.Icon(ft.Icons.TRENDING_FLAT, size=22, color=ft.Colors.GREY_700),
        bgcolor=ft.Colors.with_opacity(0.12, ft.Colors.GREY_700),
        border_radius=10,
        padding=8,
    )

    stat_row = ft.ResponsiveRow(
        [
            ft.Card(
                elevation=0,
                col={"xs": 12, "sm": 4},
                content=ft.Container(
                    padding=16,
                    border_radius=14,
                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                    content=ft.Row(
                        [
                            trend_stat_container,
                            ft.Column(
                                [ft.Text("Trend", size=11, color=ft.Colors.ON_SURFACE_VARIANT), trend_value],
                                spacing=2, tight=True, expand=True,
                            ),
                        ],
                        spacing=12,
                    ),
                ),
            ),
            _stat_card(ft.Icons.BAR_CHART_ROUNDED, "Volume", volume_value),
            _stat_card(ft.Icons.QUERY_STATS_ROUNDED, "Indicators", indicators_value),
        ],
        spacing=12,
        run_spacing=12,
    )

    # ---- results: candidates table -----------------------------------
    candidates_table = ft.DataTable(
        columns=[ft.DataColumn(ft.Text(name, weight=ft.FontWeight.W_600, size=12)) for name in CANDIDATE_COLUMNS],
        rows=[],
        heading_row_color=ft.Colors.with_opacity(0.06, ft.Colors.ON_SURFACE),
        border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
        border_radius=12,
        column_spacing=28,
        data_row_color={ft.ControlState.HOVERED: ft.Colors.with_opacity(0.05, ft.Colors.PRIMARY)},
        heading_row_height=40,
        data_row_min_height=40,
    )
    candidates_empty_state = ft.Container(
        padding=24,
        alignment=ft.alignment.center,
        content=ft.Column(
            [
                ft.Icon(ft.Icons.INBOX_OUTLINED, size=28, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text("No candidates yet — run an analysis above", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=6,
        ),
    )
    candidates_card = _card(
        _section_title("Scored candidates", ft.Icons.TABLE_CHART_OUTLINED),
        candidates_empty_state,
        ft.Row([candidates_table], scroll=ft.ScrollMode.AUTO),
    )
    candidates_table.visible = False

    fundamentals_body = ft.Column([], spacing=12, tight=True)
    fundamentals_card = _card(
        _section_title("Fundamentals", ft.Icons.ACCOUNT_BALANCE_OUTLINED),
        fundamentals_body,
    )
    fundamentals_card.visible = False

    results_column = ft.Column(
        [analysis_warnings_banner, recommendation_card, stat_row, candidates_card, fundamentals_card],
        spacing=16,
        visible=False,
    )
    results_placeholder = ft.Container(
        padding=40,
        alignment=ft.alignment.center,
        content=ft.Column(
            [
                ft.Icon(ft.Icons.QUERY_STATS_ROUNDED, size=40, color=ft.Colors.OUTLINE),
                ft.Text(
                    "Enter a symbol above and click Analyze to get a recommendation.",
                    size=13,
                    color=ft.Colors.ON_SURFACE_VARIANT,
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        ),
    )

    def run_analysis(_: ft.ControlEvent) -> None:
        set_error("")
        set_analysis_warnings([])
        progress.visible = True
        analyze_button.disabled = True
        reset_agents_tab()
        page.update()

        try:
            lookback_days = int(lookback_field.value or 365)
        except ValueError:
            progress.visible = False
            analyze_button.disabled = False
            set_error("Lookback days must be an integer.")
            page.update()
            return

        symbol = (symbol_field.value or "").strip().upper()
        if not symbol:
            progress.visible = False
            analyze_button.disabled = False
            set_error("Symbol is required.")
            page.update()
            return

        try:
            result = client.analyze(
                symbol,
                lookback_days=lookback_days,
                expiration=(expiration_field.value or "").strip() or None,
                weighting_profile=weighting_profile_dropdown.value or "swing",
            )
        except ApiError as exc:
            progress.visible = False
            analyze_button.disabled = False
            set_error(str(exc))
            page.update()
            return

        # _render_result must never leave the page stuck: a problem partway
        # through rendering the payload (rather than fetching it) still
        # needs to release the spinner/button and surface *something*,
        # instead of silently leaving the analyze button disabled forever.
        try:
            _render_result(result)
            set_analysis_warnings(result.get("data_warnings") or [])
        except Exception as exc:  # noqa: BLE001 — always resolve the run, never hang
            set_analysis_warnings([f"This run's results could not be fully rendered: {exc}"])

        progress.visible = False
        analyze_button.disabled = False
        results_placeholder.visible = False
        results_column.visible = True
        page.update()

    def _render_result(result: dict) -> None:
        current_run_id["value"] = result["run_id"]
        report_state["analysis"] = result
        recommendation = result["recommendation"]
        last_recommendation["action"] = recommendation.get("action", "—")
        last_recommendation["confidence"] = recommendation.get("confidence") or 0.0
        agents_no_run_placeholder.visible = False
        agents_ready_container.visible = True

        tone = recommendation_tone(recommendation.get("action", ""))
        color, _ = _tone_colors(tone)
        action_badge.content.value = recommendation.get("action", "—")
        action_badge.bgcolor = color
        confidence = recommendation.get("confidence") or 0.0
        confidence_bar.value = confidence
        confidence_bar.color = color
        confidence_text.value = f"{confidence:.0%} confidence"
        rec_facts_grid.controls = [
            _fact_box(label, value)
            for label, value in recommendation_facts(recommendation, result.get("candidates"))
        ]
        trade_quality_panel.content = _trade_quality_panel(result.get("trade_quality"))
        trade_quality_panel.bgcolor = ft.Colors.TRANSPARENT
        trade_quality_panel.padding = 0
        trade_quality_panel.border_radius = 0

        trend = result["trend"]
        trend_tone_name = trend_tone(trend.get("direction", ""))
        trend_color, trend_bg = _tone_colors(trend_tone_name)
        trend_stat_container.bgcolor = ft.Colors.with_opacity(0.12, trend_color)
        trend_stat_container.content = ft.Icon(
            _TREND_ICONS.get(trend_tone_name, ft.Icons.TRENDING_FLAT), size=22, color=trend_color
        )
        trend_value.value = format_trend_summary(trend).replace("Trend: ", "")

        volume_value.value = format_volume_summary(result["volume"]).replace("Volume: ", "")
        indicators_value.value = format_indicator_summary(result["indicators"])

        rows = candidates_to_rows(result["candidates"])
        candidates_table.rows = [
            ft.DataRow(cells=[ft.DataCell(ft.Text(value, size=12)) for value in row]) for row in rows
        ]
        candidates_table.visible = bool(rows)
        candidates_empty_state.visible = not rows

        # Cross-provider fundamentals (merged across every configured source).
        # Supplementary — never let a problem building this card blank the
        # core recommendation/candidates above it.
        fundamentals = result.get("fundamentals")
        try:
            fundamentals_body.controls = _fundamentals_controls(
                fundamentals, result.get("data_warnings")
            )
            fundamentals_card.visible = bool(fundamentals)
        except Exception:  # noqa: BLE001 — the analysis result must still render
            fundamentals_body.controls = [
                ft.Text(
                    "Fundamentals could not be displayed for this run.",
                    size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT,
                )
            ]
            fundamentals_card.visible = bool(fundamentals)

    analyze_button.on_click = run_analysis

    analyze_tab = ft.Container(
        padding=20,
        content=ft.Column(
            [
                _card(
                    _section_title("Run analysis", ft.Icons.SEARCH),
                    ft.ResponsiveRow(
                        [
                            ft.Column([symbol_field], col={"xs": 12, "sm": 2}),
                            ft.Column([lookback_field], col={"xs": 12, "sm": 2}),
                            ft.Column([expiration_field], col={"xs": 12, "sm": 3}),
                            ft.Column([weighting_profile_dropdown], col={"xs": 12, "sm": 3}),
                            ft.Column(
                                [ft.Row([analyze_button, progress], spacing=10)],
                                col={"xs": 12, "sm": 2},
                            ),
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.END,
                        spacing=12,
                    ),
                    error_banner,
                ),
                results_placeholder,
                results_column,
            ],
            spacing=16,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        ),
        expand=True,
    )

    # ---- agents tab: final output + agent conversation ------------------
    agents_no_run_placeholder = ft.Container(
        padding=40,
        alignment=ft.alignment.center,
        content=ft.Column(
            [
                ft.Icon(ft.Icons.FORUM_OUTLINED, size=40, color=ft.Colors.OUTLINE),
                ft.Text(
                    "Run an analysis in the Analyze tab first, then generate the "
                    "agent pipeline's interpretation of it here.",
                    size=13,
                    color=ft.Colors.ON_SURFACE_VARIANT,
                    text_align=ft.TextAlign.CENTER,
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        ),
    )

    provider_dropdown = ft.Dropdown(
        label="Provider",
        value="auto",
        width=180,
        border_radius=10,
        text_size=14,
        options=[
            ft.dropdown.Option("auto", "Auto (recommended)"),
            ft.dropdown.Option("anthropic", "Anthropic"),
            ft.dropdown.Option("openai", "OpenAI"),
            ft.dropdown.Option("groq", "Groq"),
            ft.dropdown.Option("gemini", "Gemini"),
            ft.dropdown.Option("deepseek", "DeepSeek"),
            ft.dropdown.Option("openrouter", "OpenRouter"),
        ],
    )
    api_key_field = ft.TextField(
        label="API key (optional)",
        hint_text="Uses the server's configured key if left blank",
        password=True,
        can_reveal_password=True,
        width=320,
        border_radius=10,
        text_size=14,
        disabled=True,
    )

    def _on_provider_change(_: ft.ControlEvent) -> None:
        is_auto = provider_dropdown.value == "auto"
        api_key_field.disabled = is_auto
        if is_auto:
            api_key_field.value = ""
        page.update()

    provider_dropdown.on_change = _on_provider_change

    thesis_progress = ft.ProgressRing(visible=False, width=18, height=18, stroke_width=2)
    thesis_button = ft.ElevatedButton(
        "Generate investment thesis",
        icon=ft.Icons.AUTO_AWESOME_OUTLINED,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
    )
    thesis_error_banner = ft.Container(
        visible=False,
        bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.RED),
        border_radius=10,
        padding=12,
        content=ft.Row(
            [
                ft.Icon(ft.Icons.ERROR_OUTLINE, color=ft.Colors.RED_700, size=18),
                ft.Text("", color=ft.Colors.RED_700, size=13, expand=True),
            ],
            spacing=8,
        ),
    )

    # Non-fatal problems hit mid-pipeline (pipeline_warnings in the API
    # response): the run still completed, so this renders amber alongside
    # the results rather than red instead of them.
    pipeline_warnings_column = ft.Column([], spacing=4, tight=True, expand=True)
    pipeline_warnings_banner = ft.Container(
        visible=False,
        bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.AMBER),
        border_radius=10,
        padding=12,
        content=ft.Row(
            [
                ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color=ft.Colors.AMBER_800, size=18),
                pipeline_warnings_column,
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.START,
        ),
    )

    # -- final output: a compact, scannable verdict --
    final_action_badge = ft.Container(visible=False)
    final_consensus_badge = ft.Container(visible=False)
    final_confidence_text = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)

    download_pdf_button = ft.OutlinedButton(
        "Download PDF",
        icon=ft.Icons.PICTURE_AS_PDF_OUTLINED,
        disabled=True,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
    )
    # Desktop: a persistent line confirming where the file landed. Web: a real
    # link the user taps (launches in-gesture, so the browser won't block it —
    # unlike a programmatic launch_url after the click round-trips to the server).
    download_status = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT, selectable=True, visible=False)
    download_pdf_link = ft.Markdown("", auto_follow_links=True, visible=False)

    def _assemble_report() -> dict:
        """Fold the retained analysis + thesis payloads into the single dict
        build_report_pdf expects."""
        analysis = report_state.get("analysis") or {}
        return {
            "symbol": analysis.get("symbol"),
            "generated_at": analysis.get("generated_at"),
            "recommendation": analysis.get("recommendation"),
            "trend": analysis.get("trend"),
            "volume": analysis.get("volume"),
            "indicators": analysis.get("indicators"),
            "candidates": analysis.get("candidates"),
            "trade_quality": analysis.get("trade_quality"),
            "fundamentals": analysis.get("fundamentals"),
            "data_warnings": analysis.get("data_warnings"),
            "thesis": report_state.get("thesis"),
        }

    def _report_pdf_error(message: str) -> None:
        thesis_error_banner.content.controls[1].value = message
        thesis_error_banner.visible = True
        page.update()

    def _downloads_dir() -> str:
        """Best writable spot for a saved report, preferring the user's
        Downloads folder and degrading to home, then a temp dir."""
        for candidate in (os.path.join(os.path.expanduser("~"), "Downloads"), os.path.expanduser("~")):
            if os.path.isdir(candidate):
                return candidate
        return tempfile.gettempdir()

    def _download_pdf(_: ft.ControlEvent) -> None:
        download_status.visible = False
        download_pdf_link.visible = False
        analysis = report_state.get("analysis") or {}
        symbol = re.sub(r"[^A-Za-z0-9_-]", "", str(analysis.get("symbol") or "report")) or "report"
        try:
            pdf_bytes = build_report_pdf(_assemble_report())
        except Exception as exc:  # noqa: BLE001 - surface any build failure to the user
            _report_pdf_error(f"Could not build PDF: {exc}")
            return
        filename = f"{symbol}_run{analysis.get('run_id') or 0}_report.pdf"

        if page.web:
            # Browser build: write the PDF into the served assets dir and reveal
            # a link. FilePicker.save_file has no dialog here, and a programmatic
            # launch_url gets popup-blocked; a user-clicked link does not.
            if not reports_dir:
                _report_pdf_error("PDF download isn't available in this build.")
                return
            try:
                with open(os.path.join(reports_dir, filename), "wb") as handle:
                    handle.write(pdf_bytes)
            except OSError as exc:
                _report_pdf_error(f"Could not prepare PDF: {exc}")
                return
            base = (page.url or "").rstrip("/")
            url = f"{base}/{filename}" if base else f"/{filename}"
            download_pdf_link.value = f"**[⬇  Download {filename}]({url})**  — opens in a new tab"
            download_pdf_link.visible = True
            page.update()
            return

        # Desktop build: FilePicker's native dialog is unreliable across
        # platforms, so save straight to Downloads and show the exact path.
        path = os.path.join(_downloads_dir(), filename)
        try:
            with open(path, "wb") as handle:
                handle.write(pdf_bytes)
        except OSError as exc:
            _report_pdf_error(f"Could not save PDF: {exc}")
            return
        download_status.value = f"✓  Saved to {path}"
        download_status.visible = True
        page.open(ft.SnackBar(ft.Text(f"Report saved to {path}")))
        page.update()

    download_pdf_button.on_click = _download_pdf

    trade_quality_comparison_column = ft.Column([], spacing=12, tight=True)
    trade_quality_comparison_card = _card(
        _section_title("Trade Quality Score — Quant vs. Agents", ft.Icons.COMPARE_ARROWS_OUTLINED),
        trade_quality_comparison_column,
    )

    final_output_card = ft.Column(
        [
            _card(
                _section_title("Final output", ft.Icons.FLAG_OUTLINED),
                ft.Row(
                    [final_action_badge, final_consensus_badge, final_confidence_text],
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Row([download_pdf_button], spacing=10),
                download_status,
                download_pdf_link,
            ),
            trade_quality_comparison_card,
        ],
        visible=False,
    )

    # -- agent conversation: sequential message transcript --
    quant_score_badge = ft.Container(visible=False)
    quant_narrative_text = ft.Text("", size=13, selectable=True)
    quant_factors_row = ft.Row([], wrap=True, spacing=6)
    quant_message_body = ft.Column(
        [quant_score_badge, quant_narrative_text, quant_factors_row], spacing=8, tight=True
    )

    financial_health_badge = ft.Container(visible=False)
    financial_chips_row = ft.Row([], wrap=True, spacing=6)
    financial_analyst_text = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT, italic=True)
    financial_narrative_text = ft.Text("", size=13, selectable=True)
    financial_message_body = ft.Column(
        [financial_health_badge, financial_chips_row, financial_analyst_text, financial_narrative_text],
        spacing=6,
        tight=True,
    )

    news_sentiment_badge = ft.Container(visible=False)
    news_summary_text = ft.Text("", size=13, selectable=True)
    news_catalysts_column = ft.Column([], spacing=4, tight=True)
    news_risks_column = ft.Column([], spacing=4, tight=True)
    news_message_body = ft.Column(
        [news_sentiment_badge, news_summary_text, news_catalysts_column, news_risks_column],
        spacing=8,
        tight=True,
    )

    macro_regime_badge = ft.Container(visible=False)
    macro_summary_text = ft.Text("", size=13, selectable=True)
    macro_outlook_text = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT, italic=True)
    macro_message_body = ft.Column(
        [macro_regime_badge, macro_summary_text, macro_outlook_text], spacing=6, tight=True
    )

    catalyst_bias_badge = ft.Container(visible=False)
    catalyst_summary_text = ft.Text("", size=13, selectable=True)
    catalyst_items_column = ft.Column([], spacing=8, tight=True)
    catalyst_message_body = ft.Column(
        [catalyst_bias_badge, catalyst_summary_text, catalyst_items_column], spacing=8, tight=True
    )

    risk_badge = ft.Container(visible=False)
    risk_concerns_column = ft.Column([], spacing=4, tight=True)
    risk_sizing_text = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT, italic=True)
    risk_message_body = ft.Column(
        [risk_badge, risk_concerns_column, risk_sizing_text], spacing=8, tight=True
    )

    strategy_name_text = ft.Text("", size=13, weight=ft.FontWeight.W_600, selectable=True)
    strategy_rationale_text = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
    strategy_message_body = ft.Column(
        [strategy_name_text, strategy_rationale_text], spacing=4, tight=True
    )

    relative_strength_narrative_text = ft.Text("", size=13, selectable=True)
    relative_strength_message_body = ft.Column([relative_strength_narrative_text], spacing=8, tight=True)

    statistical_edge_narrative_text = ft.Text("", size=13, selectable=True)
    statistical_edge_message_body = ft.Column([statistical_edge_narrative_text], spacing=8, tight=True)

    thesis_consensus_badge = ft.Container(visible=False)
    thesis_text = ft.Text("", size=13, selectable=True)
    thesis_message_body = ft.Column([thesis_consensus_badge, thesis_text], spacing=8, tight=True)

    # -- per-agent status pills + under-the-hood (raw prompt/response) panels --
    quant_status = _status_pill()
    financial_status = _status_pill()
    news_status = _status_pill()
    macro_status = _status_pill()
    catalyst_status = _status_pill()
    risk_status = _status_pill()
    strategy_status = _status_pill()
    relative_strength_status = _status_pill()
    statistical_edge_status = _status_pill()
    thesis_status = _status_pill()

    quant_hood, quant_hood_fill = _hood()
    financial_hood, financial_hood_fill = _hood()
    news_hood, news_hood_fill = _hood()
    macro_hood, macro_hood_fill = _hood()
    catalyst_hood, catalyst_hood_fill = _hood()
    risk_hood, risk_hood_fill = _hood()
    strategy_hood, strategy_hood_fill = _hood()
    relative_strength_hood, relative_strength_hood_fill = _hood()
    statistical_edge_hood, statistical_edge_hood_fill = _hood()
    thesis_hood, thesis_hood_fill = _hood()

    conversation_card = ft.Column(
        [
            _card(
                _section_title("Agent conversation", ft.Icons.FORUM_OUTLINED),
                _agent_message(
                    "Quant Interpreter", ft.Icons.QUERY_STATS_ROUNDED, ft.Colors.INDIGO,
                    quant_status, quant_message_body, quant_hood,
                ),
                ft.Divider(),
                _agent_message(
                    "Financial Research", ft.Icons.ACCOUNT_BALANCE_OUTLINED, ft.Colors.GREEN_700,
                    financial_status, financial_message_body, financial_hood,
                ),
                ft.Divider(),
                _agent_message(
                    "News Research", ft.Icons.NEWSPAPER, ft.Colors.BLUE_700,
                    news_status, news_message_body, news_hood,
                ),
                ft.Divider(),
                _agent_message(
                    "Macro Research", ft.Icons.PUBLIC, ft.Colors.AMBER_800,
                    macro_status, macro_message_body, macro_hood,
                ),
                ft.Divider(),
                _agent_message(
                    "Catalyst Research", ft.Icons.EVENT_NOTE_OUTLINED, ft.Colors.CYAN_800,
                    catalyst_status, catalyst_message_body, catalyst_hood,
                ),
                ft.Divider(),
                _agent_message(
                    "Risk Challenger", ft.Icons.SHIELD_OUTLINED, ft.Colors.DEEP_ORANGE,
                    risk_status, risk_message_body, risk_hood,
                ),
                ft.Divider(),
                _agent_message(
                    "Options Strategist", ft.Icons.LIGHTBULB_OUTLINE, ft.Colors.TEAL,
                    strategy_status, strategy_message_body, strategy_hood,
                ),
                ft.Divider(),
                _agent_message(
                    "Relative Strength Research", ft.Icons.SHOW_CHART, ft.Colors.LIME_800,
                    relative_strength_status, relative_strength_message_body, relative_strength_hood,
                ),
                ft.Divider(),
                _agent_message(
                    "Statistical Edge Research", ft.Icons.FUNCTIONS, ft.Colors.BROWN_600,
                    statistical_edge_status, statistical_edge_message_body, statistical_edge_hood,
                ),
                ft.Divider(),
                _agent_message(
                    "Investment Thesis", ft.Icons.AUTO_AWESOME_OUTLINED, ft.Colors.PURPLE,
                    thesis_status, thesis_message_body, thesis_hood,
                ),
                spacing=16,
            ),
        ],
        visible=False,
    )

    def reset_agents_tab() -> None:
        thesis_error_banner.visible = False
        pipeline_warnings_banner.visible = False
        thesis_button.text = "Generate investment thesis"
        final_output_card.visible = False
        conversation_card.visible = False
        # A fresh analysis invalidates any thesis captured for the PDF export.
        report_state["thesis"] = None
        download_pdf_button.disabled = True
        download_status.visible = False
        download_pdf_link.visible = False

    # A domain_score dict -> a compact "Domain: 82/100 (conf 90%)" text row,
    # appended under each research agent's message body so its own
    # independent domain judgment is visible in the conversation, not just
    # in the quant-vs-agents comparison card (see _render_final).
    def _domain_score_line(domain_score: dict | None) -> ft.Control:
        if not domain_score:
            return ft.Container(height=0)
        label = domain_score.get("domain", "").replace("_", " ").title()
        score = float(domain_score.get("score") or 0.0)
        confidence = float(domain_score.get("confidence") or 0.0)
        tone = trade_quality_tone(score)
        color, _ = _tone_colors(tone)
        return ft.Row(
            [
                ft.Container(width=8, height=8, bgcolor=color, border_radius=4),
                ft.Text(
                    f"{label} domain score: {score:.0f}/100 (confidence {confidence:.0f}%)",
                    size=11,
                    color=ft.Colors.ON_SURFACE_VARIANT,
                ),
            ],
            spacing=6,
        )

    # -- per-agent renderers: populate a message body from one agent's output --
    def _apply_quant(quant: dict) -> None:
        quant_trade_quality = quant.get("quant_trade_quality") or {}
        quant_score = quant_trade_quality.get("composite_score") or 0.0
        _fill_pill(quant_score_badge, f"SCORE {quant_score:.0f}/100", trade_quality_tone(quant_score))
        quant_narrative_text.value = quant.get("narrative", "")
        quant_factors_row.controls = [_chip(factor) for factor in quant.get("key_factors", [])]
        quant_message_body.controls = [
            quant_score_badge,
            quant_narrative_text,
            quant_factors_row,
            _domain_score_line(quant.get("technical_domain_score")),
        ]

    def _apply_financial(financial: dict) -> None:
        _fill_pill(
            financial_health_badge,
            f"HEALTH: {financial.get('company_health', '—').upper()}",
            company_health_tone(financial.get("company_health", "")),
        )
        financial_chips_row.controls = [
            _toned_chip(f"Growth: {financial.get('growth', '—')}", growth_tone(financial.get("growth", ""))),
            _toned_chip(
                f"Profitability: {financial.get('profitability', '—')}",
                profitability_tone(financial.get("profitability", "")),
            ),
            _toned_chip(
                f"Cash flow: {financial.get('cash_flow', '—')}", cash_flow_tone(financial.get("cash_flow", ""))
            ),
        ]
        financial_analyst_text.value = f"Analyst consensus: {financial.get('analyst_consensus', '—')}"
        financial_narrative_text.value = financial.get("narrative", "")
        financial_message_body.controls = [
            financial_health_badge, financial_chips_row, financial_analyst_text, financial_narrative_text,
            _domain_score_line(financial.get("domain_score")),
        ]

    def _apply_news(news: dict) -> None:
        news_tone = trend_tone(news.get("sentiment", ""))
        news_sentiment_badge.content = ft.Text(
            news.get("sentiment", "—").upper(), size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE
        )
        news_sentiment_badge.bgcolor = _tone_colors(news_tone)[0]
        news_sentiment_badge.border_radius = 20
        news_sentiment_badge.padding = ft.padding.symmetric(vertical=3, horizontal=10)
        news_sentiment_badge.visible = True
        news_summary_text.value = news.get("summary", "")
        catalysts = news.get("catalysts", [])
        news_catalysts_column.controls = (
            [ft.Text("Catalysts", size=11, weight=ft.FontWeight.BOLD), _bullet_list(catalysts)]
            if catalysts
            else []
        )
        risks = news.get("risks", [])
        news_risks_column.controls = (
            [ft.Text("Risks", size=11, weight=ft.FontWeight.BOLD), _bullet_list(risks)] if risks else []
        )
        news_message_body.controls = [
            news_sentiment_badge, news_summary_text, news_catalysts_column, news_risks_column,
            _domain_score_line(news.get("domain_score")),
        ]

    def _apply_macro(macro: dict) -> None:
        macro_tone = macro_regime_tone(macro.get("regime", ""))
        macro_regime_badge.content = ft.Text(
            macro.get("regime", "—").upper().replace("_", " "),
            size=11,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.WHITE,
        )
        macro_regime_badge.bgcolor = _tone_colors(macro_tone)[0]
        macro_regime_badge.border_radius = 20
        macro_regime_badge.padding = ft.padding.symmetric(vertical=3, horizontal=10)
        macro_regime_badge.visible = True
        macro_summary_text.value = macro.get("summary", "")
        macro_outlook_text.value = macro.get("outlook", "")
        macro_message_body.controls = [
            macro_regime_badge, macro_summary_text, macro_outlook_text,
            _domain_score_line(macro.get("domain_score")),
        ]

    def _apply_catalyst(catalyst: dict) -> None:
        net_bias = catalyst.get("net_bias", "—")
        catalyst_bias_badge.content = ft.Text(
            f"NET: {net_bias.upper()}", size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE
        )
        catalyst_bias_badge.bgcolor = _tone_colors(consensus_tone(net_bias))[0]
        catalyst_bias_badge.border_radius = 20
        catalyst_bias_badge.padding = ft.padding.symmetric(vertical=3, horizontal=10)
        catalyst_bias_badge.visible = True
        catalyst_summary_text.value = catalyst.get("summary", "")
        catalyst_items_column.controls = [
            _catalyst_entry(item) for item in catalyst.get("catalysts", [])
        ] or [ft.Text("No discrete catalysts identified.", size=12, italic=True,
                      color=ft.Colors.ON_SURFACE_VARIANT)]
        catalyst_message_body.controls = [
            catalyst_bias_badge, catalyst_summary_text, catalyst_items_column
        ]

    def _apply_risk(risk: dict) -> None:
        risk_tone = risk_level_tone(risk.get("risk_level", ""))
        risk_badge.content = ft.Text(
            risk.get("risk_level", "—").upper(), size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE
        )
        risk_badge.bgcolor = _tone_colors(risk_tone)[0]
        risk_badge.border_radius = 20
        risk_badge.padding = ft.padding.symmetric(vertical=3, horizontal=10)
        risk_badge.visible = True
        risk_concerns_column.controls = [_bullet_list(risk.get("concerns", []))]
        risk_sizing_text.value = risk.get("position_sizing_note", "")
        risk_message_body.controls = [
            risk_badge, risk_concerns_column, risk_sizing_text, _domain_score_line(risk.get("domain_score")),
        ]

    def _apply_strategy(strategy: dict) -> None:
        strategy_name_text.value = strategy.get("strategy", "")
        strategy_rationale_text.value = strategy.get("rationale", "")
        strategy_message_body.controls = [
            strategy_name_text, strategy_rationale_text, _domain_score_line(strategy.get("domain_score")),
        ]

    def _apply_relative_strength(finding: dict) -> None:
        relative_strength_narrative_text.value = finding.get("narrative", "")
        relative_strength_message_body.controls = [
            relative_strength_narrative_text, _domain_score_line(finding.get("domain_score")),
        ]

    def _apply_statistical_edge(finding: dict) -> None:
        statistical_edge_narrative_text.value = finding.get("narrative", "")
        statistical_edge_message_body.controls = [
            statistical_edge_narrative_text, _domain_score_line(finding.get("domain_score")),
        ]

    def _apply_thesis(investment_thesis: dict) -> None:
        consensus = investment_thesis.get("consensus", "—")
        thesis_consensus_badge.content = ft.Text(
            consensus.upper(), size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE
        )
        thesis_consensus_badge.bgcolor = _tone_colors(consensus_tone(consensus))[0]
        thesis_consensus_badge.border_radius = 20
        thesis_consensus_badge.padding = ft.padding.symmetric(vertical=3, horizontal=10)
        thesis_consensus_badge.visible = True
        thesis_text.value = investment_thesis.get("thesis", "")
        thesis_message_body.controls = [thesis_consensus_badge, thesis_text]

    # Agent id (as emitted by the orchestrator) → (status pill, message body,
    # apply-output fn, under-the-hood fill fn).
    _AGENTS: dict[str, tuple] = {
        "quant_interpreter": (quant_status, quant_message_body, _apply_quant, quant_hood_fill),
        "financial_research": (financial_status, financial_message_body, _apply_financial, financial_hood_fill),
        "news_research": (news_status, news_message_body, _apply_news, news_hood_fill),
        "macro_research": (macro_status, macro_message_body, _apply_macro, macro_hood_fill),
        "catalyst_research": (catalyst_status, catalyst_message_body, _apply_catalyst, catalyst_hood_fill),
        "risk_challenger": (risk_status, risk_message_body, _apply_risk, risk_hood_fill),
        "options_strategy": (strategy_status, strategy_message_body, _apply_strategy, strategy_hood_fill),
        "relative_strength_research": (
            relative_strength_status, relative_strength_message_body,
            _apply_relative_strength, relative_strength_hood_fill,
        ),
        "statistical_edge_research": (
            statistical_edge_status, statistical_edge_message_body,
            _apply_statistical_edge, statistical_edge_hood_fill,
        ),
        "investment_thesis": (thesis_status, thesis_message_body, _apply_thesis, thesis_hood_fill),
    }

    def _reset_agent(agent_id: str) -> None:
        status, body, _apply, hood_fill = _AGENTS[agent_id]
        _set_status(status, "queued")
        body.controls = [
            ft.Text("Waiting to run…", size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)
        ]
        hood_fill(None)

    def _handle_agent_event(data: dict) -> None:
        agent_id = data.get("agent", "")
        if agent_id not in _AGENTS:
            return
        status, body, apply, hood_fill = _AGENTS[agent_id]
        phase = data.get("phase")
        if phase == "started":
            _set_status(status, "running")
            body.controls = [
                ft.Text("Running…", size=12, italic=True, color=ft.Colors.ON_SURFACE_VARIANT)
            ]
        elif phase == "completed":
            _set_status(status, "done")
            apply(data.get("output") or {})
            hood_fill(data.get("exchange"))
        elif phase == "skipped":
            _set_status(status, "skipped")
            body.controls = [_skipped_message(data.get("detail") or "Skipped.")]
            hood_fill(None)
        elif phase == "failed":
            _set_status(status, "failed")
            body.controls = [
                _skipped_message(f"Failed — {data.get('detail') or 'agent error'}.")
            ]
            hood_fill(data.get("exchange"))

    def _render_final(result: dict) -> None:
        investment_thesis = result.get("investment_thesis") or {}
        warnings = result.get("pipeline_warnings") or []

        if warnings:
            pipeline_warnings_column.controls = [
                ft.Text(warning, color=ft.Colors.AMBER_800, size=13, selectable=True)
                for warning in warnings
            ]
            pipeline_warnings_banner.visible = True

        # -- final output verdict --
        final_action_badge.content = ft.Text(
            last_recommendation["action"], size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE
        )
        final_action_badge.bgcolor = _tone_colors(recommendation_tone(last_recommendation["action"]))[0]
        final_action_badge.border_radius = 20
        final_action_badge.padding = ft.padding.symmetric(vertical=6, horizontal=14)
        final_action_badge.visible = True

        consensus = investment_thesis.get("consensus", "—")
        final_consensus_badge.content = ft.Text(
            f"AGENTS: {consensus.upper()}", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE
        )
        final_consensus_badge.bgcolor = _tone_colors(consensus_tone(consensus))[0]
        final_consensus_badge.border_radius = 20
        final_consensus_badge.padding = ft.padding.symmetric(vertical=6, horizontal=14)
        final_consensus_badge.visible = True

        final_confidence_text.value = f"{last_recommendation['confidence']:.0%} confidence"
        final_output_card.visible = True

        # -- Trade Quality Score: quant (from the /analyze run) vs. agents --
        quant_trade_quality = (result.get("quant_interpretation") or {}).get("quant_trade_quality")
        agent_trade_quality = result.get("agent_trade_quality")
        analysis_result = report_state.get("analysis") or {}
        quant_source_trade_quality = quant_trade_quality or analysis_result.get("trade_quality")
        agreement = trade_quality_agreement_summary(quant_source_trade_quality, agent_trade_quality)
        trade_quality_comparison_column.controls = [
            ft.Text(agreement, size=12, color=ft.Colors.ON_SURFACE_VARIANT) if agreement else ft.Container(height=0),
            ft.ResponsiveRow(
                [
                    ft.Column(
                        [ft.Text("Quant", size=11, weight=ft.FontWeight.W_600), _trade_quality_panel(quant_source_trade_quality)],
                        col={"xs": 12, "sm": 6}, spacing=6, tight=True,
                    ),
                    ft.Column(
                        [ft.Text("Agents", size=11, weight=ft.FontWeight.W_600), _trade_quality_panel(agent_trade_quality)],
                        col={"xs": 12, "sm": 6}, spacing=6, tight=True,
                    ),
                ],
                spacing=12, run_spacing=12,
            ),
        ]

        # Retain the payload and unlock the PDF export now that a full run exists.
        report_state["thesis"] = result
        download_pdf_button.disabled = False
        thesis_button.text = "Regenerate investment thesis"

    def _show_stream_error(message: str) -> None:
        thesis_error_banner.content.controls[1].value = message
        thesis_error_banner.visible = True

    def _stream_worker() -> None:
        got_result = False
        try:
            for item in client.stream_thesis(
                current_run_id["value"],
                regenerate=True,
                provider=provider_dropdown.value or "auto",
                api_key=(api_key_field.value or "").strip() or None,
            ):
                event, data = item["event"], item["data"]
                if event == "agent":
                    _handle_agent_event(data)
                elif event == "result":
                    got_result = True
                    _render_final(data)
                elif event == "error":
                    _show_stream_error(data.get("detail", "thesis generation failed"))
                page.update()
        except ApiError as exc:
            _show_stream_error(str(exc))
        finally:
            if not got_result and not thesis_error_banner.visible:
                _show_stream_error("Thesis stream ended without a result.")
            thesis_progress.visible = False
            thesis_button.disabled = False
            page.update()

    def generate_thesis(_: ft.ControlEvent) -> None:
        if current_run_id["value"] is None:
            return
        thesis_error_banner.visible = False
        pipeline_warnings_banner.visible = False
        final_output_card.visible = False
        download_status.visible = False
        download_pdf_link.visible = False
        thesis_progress.visible = True
        thesis_button.disabled = True
        # Reset every agent to a queued state and reveal the live transcript.
        for agent_id in _AGENTS:
            _reset_agent(agent_id)
        conversation_card.visible = True
        page.update()

        page.run_thread(_stream_worker)

    thesis_button.on_click = generate_thesis

    agents_ready_container = ft.Column(
        [
            _card(
                _section_title("Investment thesis pipeline", ft.Icons.AUTO_AWESOME_OUTLINED),
                ft.Text(
                    "Runs Quant Interpreter, Risk Challenger, Options Strategist, and "
                    "Investment Thesis over the analysis above.",
                    size=12,
                    color=ft.Colors.ON_SURFACE_VARIANT,
                ),
                ft.Row(
                    [provider_dropdown, api_key_field],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.END,
                    wrap=True,
                ),
                ft.Text(
                    "The API key is sent only for this request; it is never stored or logged.",
                    size=11,
                    italic=True,
                    color=ft.Colors.ON_SURFACE_VARIANT,
                ),
                ft.Row([thesis_button, thesis_progress], spacing=10),
                thesis_error_banner,
                pipeline_warnings_banner,
            ),
            final_output_card,
            conversation_card,
        ],
        spacing=16,
        visible=False,
    )

    agents_tab = ft.Container(
        padding=20,
        content=ft.Column(
            [agents_no_run_placeholder, agents_ready_container],
            spacing=16,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        ),
        expand=True,
    )

    # ---- history tab ---------------------------------------------------
    history_symbol_field = ft.TextField(
        label="Filter by symbol (optional)", width=220, border_radius=10, text_size=14
    )
    history_limit_field = ft.TextField(
        label="Limit", value="20", width=100, border_radius=10, text_size=14
    )
    refresh_button = ft.OutlinedButton(
        "Refresh",
        icon=ft.Icons.REFRESH,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
    )
    history_error_banner = ft.Container(
        visible=False,
        bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.RED),
        border_radius=10,
        padding=12,
        content=ft.Row(
            [
                ft.Icon(ft.Icons.ERROR_OUTLINE, color=ft.Colors.RED_700, size=18),
                ft.Text("", color=ft.Colors.RED_700, size=13, expand=True),
            ],
            spacing=8,
        ),
    )

    runs_table = ft.DataTable(
        columns=[ft.DataColumn(ft.Text(name, weight=ft.FontWeight.W_600, size=12)) for name in RUN_COLUMNS],
        rows=[],
        heading_row_color=ft.Colors.with_opacity(0.06, ft.Colors.ON_SURFACE),
        border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
        border_radius=12,
        column_spacing=28,
        data_row_color={ft.ControlState.HOVERED: ft.Colors.with_opacity(0.05, ft.Colors.PRIMARY)},
        heading_row_height=40,
        data_row_min_height=40,
    )
    runs_empty_state = ft.Container(
        padding=24,
        alignment=ft.alignment.center,
        content=ft.Column(
            [
                ft.Icon(ft.Icons.HISTORY, size=28, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text("No runs loaded yet — click Refresh", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=6,
        ),
    )
    runs_table.visible = False

    def load_runs(_: ft.ControlEvent) -> None:
        history_error_banner.visible = False
        try:
            limit = int(history_limit_field.value or 20)
        except ValueError:
            history_error_banner.content.controls[1].value = "Limit must be an integer."
            history_error_banner.visible = True
            page.update()
            return

        symbol = (history_symbol_field.value or "").strip().upper() or None
        try:
            runs = client.list_runs(symbol=symbol, limit=limit)
        except ApiError as exc:
            history_error_banner.content.controls[1].value = str(exc)
            history_error_banner.visible = True
            page.update()
            return

        rows = runs_to_rows(runs)
        runs_table.rows = [
            ft.DataRow(cells=[ft.DataCell(ft.Text(value, size=12)) for value in row]) for row in rows
        ]
        runs_table.visible = bool(rows)
        runs_empty_state.visible = not rows
        page.update()

    refresh_button.on_click = load_runs

    history_tab = ft.Container(
        padding=20,
        content=ft.Column(
            [
                _card(
                    _section_title("Recent runs", ft.Icons.HISTORY),
                    ft.Row(
                        [history_symbol_field, history_limit_field, refresh_button],
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.END,
                    ),
                    history_error_banner,
                    runs_empty_state,
                    ft.Row([runs_table], scroll=ft.ScrollMode.AUTO),
                ),
            ],
            spacing=16,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        ),
        expand=True,
    )

    # ---- logs tab: live tail of the backend's in-process log buffer -----
    _LOG_LEVEL_COLORS = {
        "DEBUG": ft.Colors.GREY_600,
        "INFO": ft.Colors.BLUE_700,
        "WARNING": ft.Colors.AMBER_800,
        "ERROR": ft.Colors.RED_700,
        "CRITICAL": ft.Colors.RED_900,
    }
    _LOG_LEVELS = ["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def _short_logger_name(name: str) -> str:
        """'agentic_options_reporter.data.financial.fmp' -> 'financial.fmp'
        — trims the package prefix and keeps just enough of the dotted path
        to tell providers/modules apart at a glance."""
        trimmed = name.removeprefix("agentic_options_reporter.")
        parts = trimmed.split(".")
        return ".".join(parts[-2:]) if len(parts) > 2 else trimmed

    def _log_time(timestamp: str) -> str:
        # ISO datetime, e.g. "2026-07-07T14:03:21.512000" -> "14:03:21".
        return timestamp[11:19] if len(timestamp) >= 19 else timestamp

    def _log_row(entry: dict) -> ft.Container:
        level = entry.get("level", "INFO")
        color = _LOG_LEVEL_COLORS.get(level, ft.Colors.GREY_700)
        return ft.Container(
            content=ft.Row(
                [
                    ft.Container(
                        content=ft.Text(level, size=9, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                        bgcolor=color,
                        border_radius=4,
                        padding=ft.padding.symmetric(horizontal=6, vertical=2),
                        width=68,
                        alignment=ft.alignment.center,
                    ),
                    ft.Text(
                        _log_time(entry.get("timestamp", "")),
                        size=11,
                        color=ft.Colors.ON_SURFACE_VARIANT,
                        width=64,
                    ),
                    ft.Container(
                        content=ft.Text(_short_logger_name(entry.get("logger", "")), size=10, color=ft.Colors.ON_SURFACE_VARIANT),
                        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                        border_radius=4,
                        padding=ft.padding.symmetric(horizontal=6, vertical=2),
                    ),
                    ft.Text(entry.get("message", ""), size=12, selectable=True, expand=True),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            padding=ft.padding.symmetric(horizontal=10, vertical=5),
            border=ft.border.only(bottom=ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT)),
        )

    log_entries: list[dict] = []
    logs_last_seq = {"value": 0}
    logs_auto_refresh = {"value": True}
    logs_level_filter = {"value": "ALL"}
    logs_polling = {"value": False}

    logs_listview = ft.ListView(spacing=0, auto_scroll=True, expand=True, visible=False)
    logs_empty_state = ft.Container(
        padding=24,
        alignment=ft.alignment.center,
        content=ft.Column(
            [
                ft.Icon(ft.Icons.TERMINAL, size=28, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.Text("No log entries yet — run an analysis to see activity here", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=6,
        ),
    )
    logs_error_banner = ft.Container(
        visible=False,
        bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.RED),
        border_radius=10,
        padding=12,
        content=ft.Row(
            [
                ft.Icon(ft.Icons.ERROR_OUTLINE, color=ft.Colors.RED_700, size=18),
                ft.Text("", color=ft.Colors.RED_700, size=13, expand=True),
            ],
            spacing=8,
        ),
    )
    logs_status_text = ft.Text("", size=11, color=ft.Colors.ON_SURFACE_VARIANT)

    def _render_logs() -> None:
        wanted = logs_level_filter["value"]
        shown = log_entries if wanted == "ALL" else [e for e in log_entries if e.get("level") == wanted]
        # Cap what's actually rendered — the buffer can hold thousands of
        # entries, but only the most recent ones are useful on screen.
        logs_listview.controls = [_log_row(entry) for entry in shown[-500:]]
        logs_listview.visible = bool(shown)
        logs_empty_state.visible = not shown
        logs_status_text.value = (
            f"{len(log_entries)} entries buffered · showing {min(len(shown), 500)}"
            if log_entries else ""
        )

    def fetch_logs() -> None:
        try:
            new_entries = client.get_logs(since_seq=logs_last_seq["value"], limit=1000)
        except ApiError as exc:
            logs_error_banner.content.controls[1].value = str(exc)
            logs_error_banner.visible = True
            return
        logs_error_banner.visible = False
        if not new_entries:
            return
        log_entries.extend(new_entries)
        logs_last_seq["value"] = max(e["seq"] for e in new_entries)
        # Keep the client-side buffer bounded to match the server's ring buffer.
        overflow = len(log_entries) - 2000
        if overflow > 0:
            del log_entries[:overflow]
        _render_logs()

    def refresh_logs_click(_: ft.ControlEvent) -> None:
        fetch_logs()
        page.update()

    def clear_logs_click(_: ft.ControlEvent) -> None:
        log_entries.clear()
        logs_listview.controls = []
        logs_listview.visible = False
        logs_empty_state.visible = True
        logs_status_text.value = "Cleared locally — new activity will still stream in."
        page.update()

    def _on_level_filter_change(e: ft.ControlEvent) -> None:
        logs_level_filter["value"] = e.control.value or "ALL"
        _render_logs()
        page.update()

    def _logs_poll_loop() -> None:
        # Only one poller at a time: the guard lets the auto-refresh switch
        # be flipped on/off repeatedly without stacking background threads.
        if logs_polling["value"]:
            return
        logs_polling["value"] = True
        try:
            while logs_auto_refresh["value"]:
                fetch_logs()
                page.update()
                time.sleep(2)
        finally:
            logs_polling["value"] = False

    def _on_auto_refresh_change(e: ft.ControlEvent) -> None:
        logs_auto_refresh["value"] = e.control.value
        if logs_auto_refresh["value"]:
            page.run_thread(_logs_poll_loop)

    auto_refresh_switch = ft.Switch(label="Live tail", value=True, on_change=_on_auto_refresh_change)
    level_filter_dropdown = ft.Dropdown(
        label="Level",
        value="ALL",
        width=140,
        border_radius=10,
        text_size=13,
        options=[ft.dropdown.Option(level) for level in _LOG_LEVELS],
        on_change=_on_level_filter_change,
    )
    logs_refresh_button = ft.OutlinedButton(
        "Refresh", icon=ft.Icons.REFRESH,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
        on_click=refresh_logs_click,
    )
    logs_clear_button = ft.OutlinedButton(
        "Clear", icon=ft.Icons.CLEAR_ALL,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10)),
        on_click=clear_logs_click,
    )

    logs_tab = ft.Container(
        padding=20,
        content=ft.Column(
            [
                _card(
                    _section_title("Logs", ft.Icons.TERMINAL),
                    ft.Text(
                        "Live activity from the backend pipeline — provider requests, "
                        "failover/merge decisions, analysis stages, and the agent thesis pipeline.",
                        size=12,
                        color=ft.Colors.ON_SURFACE_VARIANT,
                    ),
                    ft.Row(
                        [auto_refresh_switch, level_filter_dropdown, logs_refresh_button, logs_clear_button],
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        wrap=True,
                    ),
                    logs_error_banner,
                    logs_status_text,
                ),
                ft.Container(
                    content=ft.Column([logs_empty_state, logs_listview], spacing=0, expand=True),
                    bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                    border_radius=12,
                    padding=4,
                    expand=True,
                ),
            ],
            spacing=16,
            expand=True,
        ),
        expand=True,
    )

    # Start the live tail immediately so the tab has content without
    # requiring a manual click, matching auto_refresh_switch's default-on state.
    page.run_thread(_logs_poll_loop)

    # ---- app bar with theme toggle --------------------------------------
    theme_icon_button = ft.IconButton(
        icon=ft.Icons.DARK_MODE_OUTLINED,
        tooltip="Toggle dark mode",
    )

    def toggle_theme(_: ft.ControlEvent) -> None:
        page.theme_mode = (
            ft.ThemeMode.DARK if page.theme_mode == ft.ThemeMode.LIGHT else ft.ThemeMode.LIGHT
        )
        theme_icon_button.icon = (
            ft.Icons.LIGHT_MODE_OUTLINED
            if page.theme_mode == ft.ThemeMode.DARK
            else ft.Icons.DARK_MODE_OUTLINED
        )
        page.update()

    theme_icon_button.on_click = toggle_theme

    page.appbar = ft.AppBar(
        leading=ft.Container(
            content=ft.Icon(ft.Icons.SHOW_CHART_ROUNDED, color=ft.Colors.WHITE),
            bgcolor=_SEED_COLOR,
            border_radius=10,
            padding=6,
            margin=ft.margin.only(left=8),
        ),
        leading_width=44,
        title=ft.Text("AgenticOptionsReporter", weight=ft.FontWeight.BOLD, size=18),
        center_title=False,
        bgcolor=ft.Colors.SURFACE,
        actions=[theme_icon_button, ft.Container(width=8)],
    )

    page.add(
        ft.Tabs(
            selected_index=0,
            expand=True,
            tabs=[
                ft.Tab(text="Analyze", icon=ft.Icons.SEARCH, content=analyze_tab),
                ft.Tab(text="Agents", icon=ft.Icons.FORUM_OUTLINED, content=agents_tab),
                ft.Tab(text="History", icon=ft.Icons.HISTORY, content=history_tab),
                ft.Tab(text="Logs", icon=ft.Icons.TERMINAL, content=logs_tab),
            ],
        )
    )


def make_main(base_url: str, reports_dir: str | None = None):
    client = ApiClient(base_url=base_url)

    def main(page: ft.Page) -> None:
        build_view(page, client, reports_dir)

    return main


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="agentic-options-reporter-ui",
        description="Flet front end for the AgenticOptionsReporter API.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("AOR_API_BASE_URL", DEFAULT_BASE_URL),
        help="Base URL of the running API (default: %(default)s)",
    )
    parser.add_argument(
        "--web", action="store_true", help="Serve in a browser instead of a desktop window"
    )
    parser.add_argument("--port", type=int, default=0, help="Port to serve on when --web is set")
    args = parser.parse_args(argv)

    # In the browser build the PDF export can't use a native save dialog, so it
    # writes reports here and Flet serves them from the web root for download.
    reports_dir = os.path.join(tempfile.gettempdir(), "aor_reports")
    os.makedirs(reports_dir, exist_ok=True)

    app_kwargs: dict[str, object] = {}
    if args.web:
        app_kwargs["assets_dir"] = reports_dir

    ft.app(
        target=make_main(args.base_url, reports_dir),
        view=ft.AppView.WEB_BROWSER if args.web else ft.AppView.FLET_APP,
        port=args.port,
        **app_kwargs,
    )


if __name__ == "__main__":
    run()
