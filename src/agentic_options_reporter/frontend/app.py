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

import flet as ft

from agentic_options_reporter.api_client import DEFAULT_BASE_URL, ApiClient, ApiError
from agentic_options_reporter.frontend.formatting import (
    CANDIDATE_COLUMNS,
    RUN_COLUMNS,
    candidates_to_rows,
    consensus_tone,
    format_indicator_summary,
    format_recommendation,
    format_trend_summary,
    format_volume_summary,
    macro_regime_tone,
    recommendation_tone,
    risk_level_tone,
    runs_to_rows,
    trend_tone,
)

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


def _card(*controls: ft.Control, padding: int = 20, spacing: int = 12) -> ft.Card:
    return ft.Card(
        elevation=1,
        content=ft.Container(
            padding=padding,
            border_radius=16,
            content=ft.Column(list(controls), spacing=spacing, tight=True),
        ),
    )


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


def build_view(page: ft.Page, client: ApiClient) -> None:
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

    current_run_id: dict[str, int | None] = {"value": None}
    last_recommendation: dict[str, object] = {"action": "—", "confidence": 0.0}

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

    # ---- results: recommendation card -------------------------------
    action_badge = ft.Container(
        content=ft.Text("—", size=13, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
        bgcolor=ft.Colors.GREY_400,
        border_radius=20,
        padding=ft.padding.symmetric(vertical=6, horizontal=14),
    )
    confidence_bar = ft.ProgressBar(value=0, width=160, border_radius=6, bgcolor=ft.Colors.GREY_200)
    confidence_text = ft.Text("0%", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
    rationale_text = ft.Text("", size=13, color=ft.Colors.ON_SURFACE_VARIANT, selectable=True)

    recommendation_card = _card(
        _section_title("Recommendation", ft.Icons.INSIGHTS_OUTLINED),
        ft.Row([action_badge, ft.Column([confidence_bar, confidence_text], spacing=2)], spacing=16),
        rationale_text,
    )

    # ---- results: stat cards -----------------------------------------
    trend_icon = ft.Icon(ft.Icons.TRENDING_FLAT, size=22, color=ft.Colors.GREY_700)
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

    results_column = ft.Column(
        [recommendation_card, stat_row, candidates_card],
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
            )
        except ApiError as exc:
            progress.visible = False
            analyze_button.disabled = False
            set_error(str(exc))
            page.update()
            return

        _render_result(result)
        progress.visible = False
        analyze_button.disabled = False
        results_placeholder.visible = False
        results_column.visible = True
        page.update()

    def _render_result(result: dict) -> None:
        current_run_id["value"] = result["run_id"]
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
        rationale_text.value = format_recommendation(recommendation)

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

    analyze_button.on_click = run_analysis

    analyze_tab = ft.Container(
        padding=20,
        content=ft.Column(
            [
                _card(
                    _section_title("Run analysis", ft.Icons.SEARCH),
                    ft.ResponsiveRow(
                        [
                            ft.Column([symbol_field], col={"xs": 12, "sm": 3}),
                            ft.Column([lookback_field], col={"xs": 12, "sm": 3}),
                            ft.Column([expiration_field], col={"xs": 12, "sm": 4}),
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
    final_output_card = ft.Column(
        [
            _card(
                _section_title("Final output", ft.Icons.FLAG_OUTLINED),
                ft.Row(
                    [final_action_badge, final_consensus_badge, final_confidence_text],
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ),
        ],
        visible=False,
    )

    # -- agent conversation: sequential message transcript --
    quant_narrative_text = ft.Text("", size=13, selectable=True)
    quant_factors_row = ft.Row([], wrap=True, spacing=6)
    quant_message_body = ft.Column([quant_narrative_text, quant_factors_row], spacing=8, tight=True)

    financial_chips_row = ft.Row([], wrap=True, spacing=6)
    financial_analyst_text = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT, italic=True)
    financial_narrative_text = ft.Text("", size=13, selectable=True)
    financial_message_body = ft.Column(
        [financial_chips_row, financial_analyst_text, financial_narrative_text], spacing=6, tight=True
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

    thesis_consensus_badge = ft.Container(visible=False)
    thesis_text = ft.Text("", size=13, selectable=True)
    thesis_message_body = ft.Column([thesis_consensus_badge, thesis_text], spacing=8, tight=True)

    conversation_card = ft.Column(
        [
            _card(
                _section_title("Agent conversation", ft.Icons.FORUM_OUTLINED),
                _agent_message("Quant Interpreter", ft.Icons.QUERY_STATS_ROUNDED, ft.Colors.INDIGO, quant_message_body),
                ft.Divider(),
                _agent_message(
                    "Financial Research", ft.Icons.ACCOUNT_BALANCE_OUTLINED, ft.Colors.GREEN_700, financial_message_body
                ),
                ft.Divider(),
                _agent_message("News Research", ft.Icons.NEWSPAPER, ft.Colors.BLUE_700, news_message_body),
                ft.Divider(),
                _agent_message("Macro Research", ft.Icons.PUBLIC, ft.Colors.AMBER_800, macro_message_body),
                ft.Divider(),
                _agent_message("Risk Challenger", ft.Icons.SHIELD_OUTLINED, ft.Colors.DEEP_ORANGE, risk_message_body),
                ft.Divider(),
                _agent_message(
                    "Options Strategist", ft.Icons.LIGHTBULB_OUTLINE, ft.Colors.TEAL, strategy_message_body
                ),
                ft.Divider(),
                _agent_message(
                    "Investment Thesis", ft.Icons.AUTO_AWESOME_OUTLINED, ft.Colors.PURPLE, thesis_message_body
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

    def generate_thesis(_: ft.ControlEvent) -> None:
        if current_run_id["value"] is None:
            return
        thesis_error_banner.visible = False
        pipeline_warnings_banner.visible = False
        thesis_progress.visible = True
        thesis_button.disabled = True
        page.update()

        try:
            result = client.generate_thesis(
                current_run_id["value"],
                regenerate=True,
                provider=provider_dropdown.value or "auto",
                api_key=(api_key_field.value or "").strip() or None,
            )
        except ApiError as exc:
            thesis_progress.visible = False
            thesis_button.disabled = False
            thesis_error_banner.content.controls[1].value = str(exc)
            thesis_error_banner.visible = True
            page.update()
            return

        quant = result["quant_interpretation"]
        financial = result.get("financial_research")
        news = result.get("news_research")
        macro = result.get("macro_research")
        risk = result.get("risk_assessment")
        strategy = result.get("strategy_suggestion")
        investment_thesis = result["investment_thesis"]
        warnings = result.get("pipeline_warnings") or []

        if warnings:
            pipeline_warnings_column.controls = [
                ft.Text(warning, color=ft.Colors.AMBER_800, size=13, selectable=True)
                for warning in warnings
            ]
            pipeline_warnings_banner.visible = True

        def _skip_reason(agent_name: str, not_configured_text: str) -> str:
            if any(warning.startswith(f"{agent_name}:") for warning in warnings):
                return "Skipped — provider failed during the run (see warning above)."
            return not_configured_text

        # -- final output verdict --
        final_action_badge.content = ft.Text(
            last_recommendation["action"], size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE
        )
        final_action_badge.bgcolor = _tone_colors(recommendation_tone(last_recommendation["action"]))[0]
        final_action_badge.border_radius = 20
        final_action_badge.padding = ft.padding.symmetric(vertical=6, horizontal=14)
        final_action_badge.visible = True

        consensus = investment_thesis.get("consensus", "—")
        consensus_color = _tone_colors(consensus_tone(consensus))[0]
        final_consensus_badge.content = ft.Text(
            f"AGENTS: {consensus.upper()}", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE
        )
        final_consensus_badge.bgcolor = consensus_color
        final_consensus_badge.border_radius = 20
        final_consensus_badge.padding = ft.padding.symmetric(vertical=6, horizontal=14)
        final_consensus_badge.visible = True

        final_confidence_text.value = f"{last_recommendation['confidence']:.0%} confidence"
        final_output_card.visible = True

        # -- conversation transcript --
        quant_narrative_text.value = quant.get("narrative", "")
        quant_factors_row.controls = [_chip(factor) for factor in quant.get("key_factors", [])]

        if financial is not None:
            financial_chips_row.controls = [
                _chip(f"Health: {financial.get('company_health', '—')}"),
                _chip(f"Growth: {financial.get('growth', '—')}"),
                _chip(f"Profitability: {financial.get('profitability', '—')}"),
                _chip(f"Cash flow: {financial.get('cash_flow', '—')}"),
            ]
            financial_analyst_text.value = f"Analyst consensus: {financial.get('analyst_consensus', '—')}"
            financial_narrative_text.value = financial.get("narrative", "")
            financial_message_body.controls = [
                financial_chips_row, financial_analyst_text, financial_narrative_text
            ]
        else:
            financial_chips_row.controls = []
            financial_analyst_text.value = ""
            financial_narrative_text.value = ""
            financial_message_body.controls = [
                _skipped_message(
                    _skip_reason("financial_research", "Skipped — no financial data provider configured.")
                )
            ]

        if news is not None:
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
                news_sentiment_badge, news_summary_text, news_catalysts_column, news_risks_column
            ]
        else:
            news_sentiment_badge.visible = False
            news_summary_text.value = ""
            news_catalysts_column.controls = []
            news_risks_column.controls = []
            news_message_body.controls = [
                _skipped_message(_skip_reason("news_research", "Skipped — no news data provider configured."))
            ]

        if macro is not None:
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
            macro_message_body.controls = [macro_regime_badge, macro_summary_text, macro_outlook_text]
        else:
            macro_regime_badge.visible = False
            macro_summary_text.value = ""
            macro_outlook_text.value = ""
            macro_message_body.controls = [
                _skipped_message(_skip_reason("macro_research", "Skipped — no macro data provider configured."))
            ]

        if risk is not None:
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
        else:
            risk_badge.visible = False
            risk_concerns_column.controls = [_skipped_message("Skipped — no candidate contract to assess.")]
            risk_sizing_text.value = ""

        if strategy is not None:
            strategy_name_text.value = strategy.get("strategy", "")
            strategy_rationale_text.value = strategy.get("rationale", "")
            strategy_message_body.controls = [strategy_name_text, strategy_rationale_text]
        else:
            strategy_message_body.controls = [
                _skipped_message("Skipped — no candidate contract to build a strategy around.")
            ]

        thesis_consensus_badge.content = ft.Text(
            consensus.upper(), size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE
        )
        thesis_consensus_badge.bgcolor = consensus_color
        thesis_consensus_badge.border_radius = 20
        thesis_consensus_badge.padding = ft.padding.symmetric(vertical=3, horizontal=10)
        thesis_consensus_badge.visible = True
        thesis_text.value = investment_thesis.get("thesis", "")

        conversation_card.visible = True

        thesis_progress.visible = False
        thesis_button.disabled = False
        thesis_button.text = "Regenerate investment thesis"
        page.update()

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
            ],
        )
    )


def make_main(base_url: str):
    client = ApiClient(base_url=base_url)

    def main(page: ft.Page) -> None:
        build_view(page, client)

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

    ft.app(
        target=make_main(args.base_url),
        view=ft.AppView.WEB_BROWSER if args.web else ft.AppView.FLET_APP,
        port=args.port,
    )


if __name__ == "__main__":
    run()
