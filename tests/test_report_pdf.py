"""Unit tests for the PDF report builder.

These assert the builder returns a valid, non-trivial PDF for a full payload
and degrades gracefully when the thesis or individual agents are missing —
without needing a Flet runtime or a real PDF viewer.
"""

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph

from agentic_options_reporter.frontend.report_pdf import (
    _recommendation_block,
    _styles,
    build_report_pdf,
    score_breakdown_flowables,
)


def _paragraph_texts(flowables) -> str:
    """Concatenate the raw text of every Paragraph in a flowable list, for
    asserting on what the block renders."""
    return " ".join(f.getPlainText() for f in flowables if isinstance(f, Paragraph))

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


def test_recommendation_block_replaces_factor_dump_with_summary():
    """When a score breakdown is present, the block captions the chart with a
    plain-language summary and drops the deterministic factor-dump rationale
    (now redundant with the visualization)."""
    rec = {
        "action": "BUY", "confidence": 0.82, "contract_symbol": "AAPL_C",
        "rationale": "AAPL_C scored 82.4/100 (trend_alignment=1.00, liquidity=0.00).",
    }
    candidates = [{"contract_symbol": "AAPL_C", "score_breakdown": {"trend_alignment": 1.0, "liquidity": 0.0}}]
    text = _paragraph_texts(_recommendation_block(rec, candidates, _styles()))

    assert "led by trend alignment" in text          # the summary caption
    assert "scored 82.4/100" not in text             # the raw factor-dump is gone
    assert "trend_alignment=1.00" not in text


def test_recommendation_block_keeps_rationale_without_breakdown():
    """AVOID / no-candidate: nothing to visualize, so the rationale stays."""
    rec = {"action": "AVOID", "confidence": 0.0, "contract_symbol": None,
           "rationale": "No liquid, scoreable candidates were found in the option chain."}
    text = _paragraph_texts(_recommendation_block(rec, [], _styles()))
    assert "No liquid, scoreable candidates" in text


def test_build_report_with_zero_and_tiny_score_breakdown_factors():
    """Regression: a factor at 0.0 (or a sliver like 0.05) made the meter
    bar a zero/sub-padding-width cell, and reportlab raised 'negative
    availWidth' at build time. The report must build for any ratio in
    [0, 1]."""
    payload = {
        "symbol": "AAPL",
        "recommendation": {
            "action": "BUY", "confidence": 0.7,
            "contract_symbol": "AAPL260116C00150000", "rationale": "x",
        },
        "candidates": [
            {
                "contract_symbol": "AAPL260116C00150000",
                "option_type": "call", "strike": 150.0, "expiration": "2026-01-16",
                "score": 80.0, "delta": 0.6, "probability_of_profit": 0.55,
                "score_breakdown": {
                    "trend_alignment": 1.0,        # full bar
                    "volume_confirmation": 0.5,    # half
                    "liquidity": 0.0,              # empty — the crash case
                    "support_resistance_proximity": 0.05,  # tiny sliver
                },
            }
        ],
    }
    data = build_report_pdf(payload)
    assert _is_pdf(data)


def test_score_breakdown_flowables_handle_extreme_ratios():
    styles = {
        "body": getSampleStyleSheet()["BodyText"],
        "cell": getSampleStyleSheet()["BodyText"],
        "cellhead": getSampleStyleSheet()["BodyText"],
    }
    # Building the whole doc is what actually wraps the flowables (and would
    # have raised); do it with only-zero and only-full breakdowns.
    for breakdown in ({"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 1.0}):
        assert score_breakdown_flowables({"score_breakdown": breakdown}, styles)
        payload = {
            "symbol": "T",
            "recommendation": {"action": "BUY", "confidence": 0.5, "contract_symbol": "C", "rationale": ""},
            "candidates": [{"contract_symbol": "C", "score_breakdown": breakdown}],
        }
        assert _is_pdf(build_report_pdf(payload))
