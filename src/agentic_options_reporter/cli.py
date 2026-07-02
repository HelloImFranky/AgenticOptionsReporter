"""Command-line client for the AgenticOptionsReporter API.

Talks to a running instance of `agentic_options_reporter.main:app` over
HTTP using `requests` (not shell `curl`), with `argparse` handling
command-line parsing. Contract mirrors specs/api.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 30


class ApiError(RuntimeError):
    """Raised when the API cannot be reached or returns an error status."""


def _request(
    method: str,
    base_url: str,
    path: str,
    params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        response = requests.request(method, url, params=params, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Request to {url} failed: {exc}") from exc

    if not response.ok:
        raise ApiError(f"{method} {url} returned {response.status_code}: {response.text}")
    return response.json()


def cmd_health(args: argparse.Namespace) -> Any:
    return _request("GET", args.base_url, "/health")


def cmd_analyze(args: argparse.Namespace) -> Any:
    params: dict[str, Any] = {"lookback_days": args.lookback_days}
    if args.expiration:
        params["expiration"] = args.expiration
    return _request("GET", args.base_url, f"/analyze/{args.symbol}", params=params)


def cmd_runs(args: argparse.Namespace) -> Any:
    params: dict[str, Any] = {"limit": args.limit}
    if args.symbol:
        params["symbol"] = args.symbol
    return _request("GET", args.base_url, "/runs", params=params)


def cmd_run(args: argparse.Namespace) -> Any:
    return _request("GET", args.base_url, f"/runs/{args.run_id}")


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = args.func(args)
    except ApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
