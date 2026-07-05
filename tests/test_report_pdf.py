"""Unit tests for the PDF report builder.

These assert the builder returns a valid, non-trivial PDF for a full payload
and degrades gracefully when the thesis or individual agents are missing —
without needing a Flet runtime or a real PDF viewer.
"""

from reportlab.lib.styles import getSampleStyleSheet

from agentic_options_reporter.frontend.report_pdf import build_report_pdf, score_breakdown_flowables

_FULL_REPORT = {
    "symbol": "AAPL",
    "generated_at": "2026-07-04T12:30:00",
    "recommendation": {
        "action": "BUY",
        "confidence": 0.732,
        "contract_symbol": "AAPL260116C00150000",
        "rationale": "Strong trend alignment with supportive volume & liquidity.",
    },
    "trend": {"direction": "bullish", "strength": "strong", "adx": 31.2},
    "volume": {"relative_volume": 1.8, "flags": ["above_average"]},
    "indicators": {"sma_20": 195.1, "sma_50": 188.4, "rsi_14": 61.2, "atr_14": 3.4},
    "candidates": [
        {
            "contract_symbol": "AAPL260116C00150000",
            "option_type": "call",
            "strike": 150.0,
            "expiration": "2026-01-16",
            "score": 82.4,
            "delta": 0.612,
            "probability_of_profit": 0.58,
            "score_breakdown": {
                "trend_alignment": 0.82,
                "volume_confirmation": 0.76,
                "support_resistance_proximity": 0.61,
                "liquidity": 0.74,
                "risk_reward": 0.69,
            },
        }
    ],
    "thesis": {
        "quant_interpretation": {
            "narrative": "The score is driven mostly by trend & volume.",
            "key_factors": ["trend alignment", "elevated volume"],
            "overall_score": 82.4,
        },
        "financial_research": {
            "company_health": "strong",
            "growth": "accelerating",
            "profitability": "high",
            "cash_flow": "positive",
            "analyst_consensus": "overweight",
            "narrative": "Balance sheet is robust with expanding margins.",
        },
        "news_research": {
            "sentiment": "bullish",
            "summary": "Coverage skews positive into earnings.",
            "catalysts": ["product launch"],
            "risks": ["valuation stretch"],
        },
        "macro_research": {
            "regime": "risk_on",
            "outlook": "Supportive liquidity backdrop.",
            "summary": "Rates stable, credit spreads tight.",
        },
        "catalyst_research": {
            "net_bias": "bullish",
            "summary": "Two near-term catalysts identified.",
            "catalysts": [
                {
                    "title": "Q1 earnings",
                    "category": "earnings",
                    "horizon": "near_term",
                    "direction": "bullish",
                    "detail": "Consensus looks beatable.",
                }
            ],
        },
        "risk_assessment": {
            "risk_level": "medium",
            "concerns": ["theta decay", "gap risk into earnings"],
            "position_sizing_note": "Size to 1% of book.",
        },
        "strategy_suggestion": {
            "strategy": "Long call",
            "rationale": "Directional conviction with defined risk.",
        },
        "investment_thesis": {
            "consensus": "bullish",
            "thesis": "The setup favours a measured long-call position.",
        },
        "pipeline_warnings": [],
    },
}


def _is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def test_build_full_report_returns_pdf_bytes():
    data = build_report_pdf(_FULL_REPORT)
    assert _is_pdf(data)
    assert len(data) > 1500  # a real multi-section document, not an empty shell


def test_build_report_without_thesis():
    payload = {k: v for k, v in _FULL_REPORT.items() if k != "thesis"}
    data = build_report_pdf(payload)
    assert _is_pdf(data)


def test_build_report_with_skipped_agents():
    thesis = dict(_FULL_REPORT["thesis"])
    for skipped in ("financial_research", "news_research", "macro_research",
                    "catalyst_research", "risk_assessment", "strategy_suggestion"):
        thesis[skipped] = None
    thesis["pipeline_warnings"] = ["news_research: provider timed out"]
    payload = {**_FULL_REPORT, "thesis": thesis}
    data = build_report_pdf(payload)
    assert _is_pdf(data)


def test_build_report_minimal_payload():
    data = build_report_pdf({"symbol": "MSFT"})
    assert _is_pdf(data)


def test_build_report_escapes_markup_in_text():
    payload = {
        "symbol": "T&T",
        "thesis": {
            "quant_interpretation": {
                "narrative": "Spread <5% & liquidity > threshold",
                "key_factors": ["a & b", "x < y"],
                "overall_score": 40.0,
            },
            "investment_thesis": {"consensus": "neutral", "thesis": "Hold & wait <for> signal"},
        },
    }
    data = build_report_pdf(payload)
    assert _is_pdf(data)


def test_score_breakdown_flowables_render_for_candidate_payload():
    flowables = score_breakdown_flowables(
        {"score_breakdown": {"trend_alignment": 0.82, "liquidity": 0.74}},
        styles={"body": getSampleStyleSheet()["BodyText"]},
    )
    assert len(flowables) > 0
