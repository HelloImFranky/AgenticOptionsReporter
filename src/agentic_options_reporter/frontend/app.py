"""Flet front end for AgenticOptionsReporter.

A thin UI over `agentic_options_reporter.api_client.ApiClient` — the same
HTTP client the CLI uses. All analysis logic runs server-side; this module
only requests data (per specs/api.yaml) and renders it. Display formatting
lives in formatting.py so it can be unit tested without a Flet runtime.
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
    format_indicator_summary,
    format_recommendation,
    format_trend_summary,
    format_volume_summary,
    runs_to_rows,
)


def build_view(page: ft.Page, client: ApiClient) -> None:
    page.title = "AgenticOptionsReporter"
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 20

    symbol_field = ft.TextField(label="Symbol", value="AAPL", width=140)
    lookback_field = ft.TextField(label="Lookback days", value="365", width=140)
    expiration_field = ft.TextField(label="Expiration (YYYY-MM-DD, optional)", width=220)

    status_text = ft.Text("", color=ft.Colors.RED)
    recommendation_text = ft.Text("", size=16, weight=ft.FontWeight.BOLD, selectable=True)
    trend_text = ft.Text("")
    volume_text = ft.Text("")
    indicators_text = ft.Text("")
    progress = ft.ProgressRing(visible=False, width=20, height=20)

    candidates_table = ft.DataTable(
        columns=[ft.DataColumn(ft.Text(name)) for name in CANDIDATE_COLUMNS],
        rows=[],
    )
    runs_table = ft.DataTable(
        columns=[ft.DataColumn(ft.Text(name)) for name in RUN_COLUMNS],
        rows=[],
    )

    def set_error(message: str) -> None:
        status_text.value = message
        page.update()

    def run_analysis(_: ft.ControlEvent) -> None:
        status_text.value = ""
        progress.visible = True
        page.update()

        try:
            lookback_days = int(lookback_field.value or 365)
        except ValueError:
            progress.visible = False
            set_error("Lookback days must be an integer.")
            return

        symbol = (symbol_field.value or "").strip().upper()
        if not symbol:
            progress.visible = False
            set_error("Symbol is required.")
            return

        try:
            result = client.analyze(
                symbol,
                lookback_days=lookback_days,
                expiration=(expiration_field.value or "").strip() or None,
            )
        except ApiError as exc:
            progress.visible = False
            set_error(str(exc))
            return

        recommendation_text.value = format_recommendation(result["recommendation"])
        trend_text.value = format_trend_summary(result["trend"])
        volume_text.value = format_volume_summary(result["volume"])
        indicators_text.value = format_indicator_summary(result["indicators"])
        candidates_table.rows = [
            ft.DataRow(cells=[ft.DataCell(ft.Text(value)) for value in row])
            for row in candidates_to_rows(result["candidates"])
        ]
        progress.visible = False
        page.update()

    def load_runs(_: ft.ControlEvent) -> None:
        status_text.value = ""
        symbol = (symbol_field.value or "").strip().upper() or None
        try:
            runs = client.list_runs(symbol=symbol, limit=20)
        except ApiError as exc:
            set_error(str(exc))
            return
        runs_table.rows = [
            ft.DataRow(cells=[ft.DataCell(ft.Text(value)) for value in row])
            for row in runs_to_rows(runs)
        ]
        page.update()

    analyze_button = ft.ElevatedButton("Analyze", on_click=run_analysis)
    runs_button = ft.OutlinedButton("Load recent runs", on_click=load_runs)

    page.add(
        ft.Text("AgenticOptionsReporter", size=24, weight=ft.FontWeight.BOLD),
        ft.Row([symbol_field, lookback_field, expiration_field, analyze_button, progress]),
        status_text,
        ft.Divider(),
        recommendation_text,
        trend_text,
        volume_text,
        indicators_text,
        ft.Text("Candidates", size=18, weight=ft.FontWeight.BOLD),
        candidates_table,
        ft.Divider(),
        ft.Row([ft.Text("Recent runs", size=18, weight=ft.FontWeight.BOLD), runs_button]),
        runs_table,
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
