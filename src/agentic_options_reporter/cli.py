"""Command-line client for the AgenticOptionsReporter API.

Talks to a running instance of `agentic_options_reporter.main:app` over
HTTP via `api_client.ApiClient`, with `argparse` handling command-line
parsing. Contract mirrors specs/api.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from agentic_options_reporter.api_client import DEFAULT_BASE_URL, ApiClient, ApiError


def cmd_health(client: ApiClient, args: argparse.Namespace) -> Any:
    return client.health()


def cmd_analyze(client: ApiClient, args: argparse.Namespace) -> Any:
    return client.analyze(args.symbol, lookback_days=args.lookback_days, expiration=args.expiration)


def cmd_runs(client: ApiClient, args: argparse.Namespace) -> Any:
    return client.list_runs(symbol=args.symbol, limit=args.limit)


def cmd_run(client: ApiClient, args: argparse.Namespace) -> Any:
    return client.get_run(args.run_id)


def cmd_thesis(client: ApiClient, args: argparse.Namespace) -> Any:
    if args.fetch_only:
        return client.get_thesis(args.run_id)
    return client.generate_thesis(
        args.run_id, regenerate=args.regenerate, provider=args.provider, api_key=args.api_key
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-options-reporter",
        description="CLI client for the AgenticOptionsReporter API.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the running API (default: {DEFAULT_BASE_URL})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    health_parser = subparsers.add_parser("health", help="Check API liveness")
    health_parser.set_defaults(func=cmd_health)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Run the analysis workflow for a symbol"
    )
    analyze_parser.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
    analyze_parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Days of price history to fetch (default: 365)",
    )
    analyze_parser.add_argument(
        "--expiration",
        default=None,
        help="Option expiration date (YYYY-MM-DD); defaults to the nearest expiration",
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    runs_parser = subparsers.add_parser("runs", help="List recent analysis runs")
    runs_parser.add_argument("--symbol", default=None, help="Filter by symbol")
    runs_parser.add_argument(
        "--limit", type=int, default=20, help="Max number of runs to return (default: 20)"
    )
    runs_parser.set_defaults(func=cmd_runs)

    run_parser = subparsers.add_parser("run", help="Fetch a specific analysis run")
    run_parser.add_argument("run_id", type=int, help="Run ID")
    run_parser.set_defaults(func=cmd_run)

    thesis_parser = subparsers.add_parser(
        "thesis",
        help="Generate (or fetch) the investment-thesis agent pipeline output for a run",
    )
    thesis_parser.add_argument("run_id", type=int, help="Run ID")
    thesis_parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Discard and regenerate an existing thesis instead of erroring",
    )
    thesis_parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Only fetch a previously generated thesis; never generate a new one",
    )
    thesis_parser.add_argument(
        "--provider",
        default="auto",
        help=(
            "LLM provider to use: auto (fails over across every configured "
            "provider), anthropic, openai, groq, gemini, deepseek, or "
            "openrouter (default: auto)"
        ),
    )
    thesis_parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the chosen provider; omit to use the server's configured key",
    )
    thesis_parser.set_defaults(func=cmd_thesis)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    client = ApiClient(base_url=args.base_url)

    try:
        result = args.func(client, args)
    except ApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
