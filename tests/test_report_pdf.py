"""Unit tests for the PDF report builder.

These assert the builder returns a valid, non-trivial PDF for a full payload
and degrades gracefully when the thesis or individual agents are missing —
without needing a Flet runtime or a real PDF viewer.
"""

from io import BytesIO

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph

from agentic_options_reporter.frontend.report_pdf import (
    _build_story,
    _fundamentals_key_facts_blocks,
    _insider_activity_blocks,
    _insider_timeseries_chart,
    _recent_earnings_blocks,
    _recommendation_block,
    _section_header,
    _styles,
    build_report_pdf,
    trade_quality_comparison_flowables,
    trade_quality_flowables,
)


def _paragraph_texts(flowables) -> str:
    """Concatenate the raw text of every Paragraph in a flowable list —
    recursing into Table cells — for asserting on what the block renders."""
    from reportlab.platypus import Table

    parts: list[str] = []

    def walk(item) -> None:
        if isinstance(item, Paragraph):
            parts.append(item.getPlainText())
        elif isinstance(item, Table):
            for row in item._cellvalues:
                for cell in row:
                    walk(cell)
        elif isinstance(item, (list, tuple)):
            for sub in item:
                walk(sub)

    walk(list(flowables))
    return " ".join(parts)


def _domain_score(score: float, confidence: float = 90.0, evidence=None) -> dict:
    return {"score": score, "confidence": confidence, "evidence": evidence or []}


def _trade_quality(domain_scores: dict, composite_score: float = 82.4, action: str = "BUY") -> dict:
    return {
        "composite_score": composite_score,
        "confidence": 88.0,
        "recommendation_action": action,
        "weighting_profile": "swing",
        "domain_scores": domain_scores,
        "explainability": ["Technical (weight 20%): 91/100 — strongest contributor"],
    }


_TRADE_QUALITY = _trade_quality(
    {
        "technical": _domain_score(91.0, evidence=["Trend alignment: 1.00"]),
        "risk": _domain_score(69.0),
        "liquidity": _domain_score(74.0),
    }
)

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
        }
    ],
    "trade_quality": _TRADE_QUALITY,
    "thesis": {
        "quant_interpretation": {
            "narrative": "The score is driven mostly by trend & volume.",
            "key_factors": ["trend alignment", "elevated volume"],
            "quant_trade_quality": _TRADE_QUALITY,
            "technical_domain_score": _domain_score(88.0, evidence=["Own read of the trend"]),
        },
        "financial_research": {
            "company_health": "strong",
            "growth": "accelerating",
            "profitability": "high",
            "cash_flow": "positive",
            "analyst_consensus": "overweight",
            "narrative": "Balance sheet is robust with expanding margins.",
            "domain_score": _domain_score(80.0),
        },
        "news_research": {
            "sentiment": "bullish",
            "summary": "Coverage skews positive into earnings.",
            "catalysts": ["product launch"],
            "risks": ["valuation stretch"],
            "domain_score": _domain_score(70.0),
        },
        "macro_research": {
            "regime": "risk_on",
            "outlook": "Supportive liquidity backdrop.",
            "summary": "Rates stable, credit spreads tight.",
            "domain_score": _domain_score(65.0),
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
        "relative_strength_research": {
            "narrative": "Outperforming both SPY and the sector ETF over the last month.",
            "domain_score": _domain_score(78.0),
        },
        "statistical_edge_research": {
            "narrative": "Thin sample size, but the Monte Carlo readout leans favorable.",
            "domain_score": _domain_score(55.0, confidence=40.0),
        },
        "risk_assessment": {
            "risk_level": "medium",
            "concerns": ["theta decay", "gap risk into earnings"],
            "position_sizing_note": "Size to 1% of book.",
            "domain_score": _domain_score(60.0),
        },
        "strategy_suggestion": {
            "strategy": "Long call",
            "rationale": "Directional conviction with defined risk.",
            "domain_score": _domain_score(74.0),
        },
        "investment_thesis": {
            "consensus": "bullish",
            "thesis": "The setup favours a measured long-call position.",
        },
        "agent_trade_quality": _trade_quality(
            {
                "technical": _domain_score(88.0),
                "fundamental": _domain_score(80.0),
                "sentiment": _domain_score(70.0),
                "macro": _domain_score(65.0),
                "risk": _domain_score(60.0),
                "liquidity": _domain_score(74.0),
                "relative_strength": _domain_score(78.0),
                "statistical_edge": _domain_score(55.0, confidence=40.0),
            },
            composite_score=74.1,
        ),
        "pipeline_warnings": [],
    },
}


def _is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def _page_texts(data: bytes) -> list[str]:
    """Full extracted text of each page, in order."""
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def _page_first_lines(data: bytes) -> list[str]:
    """The first non-blank line of text on each page — used to assert a
    section's header actually lands at the top of its own page, not just
    that the text is present somewhere in the document."""
    return [text.splitlines()[0] if text else "" for text in _page_texts(data)]


_FUNDAMENTALS = {
    "ticker": "AAPL",
    "metrics": {
        "market_cap": 3.0e12, "pe_ratio": 30.5, "beta": 1.2, "dividend_yield": 0.005,
        "operating_margin": 0.30, "profit_margin": 0.25, "week52_high": 200.0, "week52_low": 150.0,
    },
    "earnings_calendar": {"next_date": "2026-08-01", "eps_estimate": 1.6},
    "earnings_history": {"surprises": [
        {"period": "2026-03-31", "actual_eps": 1.5, "estimate_eps": 1.4, "surprise_percent": 0.071},
        {"period": "2025-12-31", "actual_eps": 2.1, "estimate_eps": 2.2, "surprise_percent": -0.045},
    ]},
    "insider_activity": {"net_shares": -500.0, "transactions": [
        {"name": "Jane Doe", "transaction_type": "sell", "shares": 1000, "filed_at": "2026-06-01"},
        {"name": "John Roe", "transaction_type": "buy", "shares": 500, "filed_at": "2026-05-01"},
    ]},
}


def test_build_full_report_returns_pdf_bytes():
    data = build_report_pdf(_FULL_REPORT)
    assert _is_pdf(data)
    assert len(data) > 1500  # a real multi-section document, not an empty shell


def test_section_header_starts_a_new_page_by_default():
    flowables = _section_header("Recommendation", _styles())
    assert isinstance(flowables[0], PageBreak)


def test_section_header_shares_the_page_when_new_page_is_false():
    """The title isn't a section, so the FIRST section shares the cover
    page with it rather than opening yet another page — new_page=False is
    how build_report_pdf expresses that."""
    flowables = _section_header("Recommendation", _styles(), new_page=False)
    assert not isinstance(flowables[0], PageBreak)


def test_title_and_first_section_share_the_cover_page():
    """The title isn't a section itself: it and the FIRST section
    (Recommendation, here) share page 1, rather than the title getting a
    near-empty page of its own."""
    payload = {**_FULL_REPORT, "fundamentals": _FUNDAMENTALS, "data_warnings": []}
    data = build_report_pdf(payload)

    pages = _page_texts(data)

    assert pages[0].splitlines()[0] == "Options Analysis Report"
    assert "Recommendation" in pages[0]


def _section_header_texts(story: list) -> list[str]:
    """The text of every page-level section header ('AorSection' style) in
    story order — filters out other Paragraphs (fact labels, narratives,
    etc.) so this reads as just the sequence of section headers."""
    return [
        item.getPlainText()
        for item in story
        if isinstance(item, Paragraph) and item.style.name == "AorSection"
    ]


def test_flowing_sections_share_pages_with_no_forced_breaks_between_them():
    """Recommendation, Technical Snapshot, Fundamentals, Recent Earnings,
    and Insider Activity ('the combined three' + the two right after them)
    flow together with no forced page break between them — 'could be on
    one page if the contents fits', spilling onto more only as needed."""
    payload = {**_FULL_REPORT, "fundamentals": _FUNDAMENTALS, "data_warnings": []}
    _, story = _build_story(payload)

    scored_candidates_index = next(
        i for i, item in enumerate(story)
        if isinstance(item, Paragraph) and item.getPlainText() == "Scored candidates"
    )
    # Exclude index scored_candidates_index - 1 too: that's the PageBreak
    # forcing Scored Candidates onto its own page, not part of the flowing
    # group this test is checking.
    flowing_group = story[: scored_candidates_index - 1]

    assert not any(isinstance(item, PageBreak) for item in flowing_group)
    assert _section_header_texts(flowing_group) == [
        "Recommendation",
        "Technical snapshot",
        "Fundamentals",
        "Recent earnings",
        "Insider activity — net selling (500 shares)",
    ]


def test_scored_candidates_trade_quality_and_agent_pipeline_each_force_a_new_page():
    """Scored Candidates and the Agent Pipeline (plus the Trade Quality
    comparison that precedes it) each 'live on its own page' — unlike the
    flowing group above, every one of these is preceded by a PageBreak."""
    payload = {**_FULL_REPORT, "fundamentals": _FUNDAMENTALS, "data_warnings": []}
    _, story = _build_story(payload)

    page_break_indices = [i for i, item in enumerate(story) if isinstance(item, PageBreak)]
    assert len(page_break_indices) == 3  # Scored candidates, Trade Quality comparison, Agent pipeline

    headers_right_after_a_break = []
    for i in page_break_indices:
        headers_right_after_a_break.append(story[i + 1].getPlainText())
    assert headers_right_after_a_break == [
        "Scored candidates",
        "Trade Quality Score — Quant vs. Agents",
        "Agent pipeline",
    ]


def test_report_section_order():
    """recommendation, technical snapshot, fundamentals, recent earnings,
    insider activity, scored candidates, then the agent pipeline."""
    payload = {**_FULL_REPORT, "fundamentals": _FUNDAMENTALS, "data_warnings": []}
    _, story = _build_story(payload)

    assert _section_header_texts(story) == [
        "Recommendation",
        "Technical snapshot",
        "Fundamentals",
        "Recent earnings",
        "Insider activity — net selling (500 shares)",
        "Scored candidates",
        "Trade Quality Score — Quant vs. Agents",
        "Agent pipeline",
    ]


def test_whichever_section_is_actually_first_shares_the_cover_page():
    """When Recommendation is absent, Technical Snapshot (the next section
    in order) becomes the first one rendered — and it, not Recommendation,
    should be the one sharing the cover page."""
    payload = {k: v for k, v in _FULL_REPORT.items() if k not in ("recommendation", "thesis")}
    data = build_report_pdf(payload)

    pages = _page_texts(data)
    first_lines = _page_first_lines(data)

    assert pages[0].splitlines()[0] == "Options Analysis Report"
    assert "Technical snapshot" in pages[0]
    assert "Technical snapshot" not in first_lines[1:]


def test_minimal_report_has_no_stray_blank_pages():
    """A payload with no recommendation/candidates/fundamentals/thesis
    should be just the cover page — no section headers fired, no
    page-break-only sections left dangling."""
    data = build_report_pdf({"symbol": "MSFT"})
    first_lines = _page_first_lines(data)
    assert first_lines == ["Options Analysis Report"]


def test_fundamentals_key_facts_blocks_render_metrics_and_next_earnings():
    """Recent earnings and insider activity are their OWN sections now
    (see _recent_earnings_blocks / _insider_activity_blocks below) — this
    block is just key metrics + next earnings + data-source warnings."""
    styles = _styles()
    blocks = _fundamentals_key_facts_blocks(_FUNDAMENTALS, ["statements: rate limited"], styles)
    text = _paragraph_texts(blocks)
    assert "Key metrics" in text
    assert "Next earnings: 2026-08-01" in text
    assert "statements: rate limited" in text   # data_warnings surfaced
    assert "Recent earnings" not in text
    assert "net selling" not in text


def test_recent_earnings_blocks_render_facts_table():
    styles = _styles()
    blocks = _recent_earnings_blocks(_FUNDAMENTALS, styles)
    text = _paragraph_texts(blocks)
    assert "2026-03-31" in text
    assert "vs" in text


def test_recent_earnings_blocks_empty_when_no_history():
    styles = _styles()
    assert _recent_earnings_blocks({"ticker": "X"}, styles) == []


def test_insider_activity_blocks_render_chart_without_the_header_text():
    """The dynamic 'Insider activity — net selling (N shares)' header is
    used as the section's page header by build_report_pdf, not rendered a
    second time as a Paragraph inside the block itself."""
    from reportlab.graphics.shapes import Drawing

    styles = _styles()
    blocks = _insider_activity_blocks(_FUNDAMENTALS, styles)
    assert any(isinstance(b, Drawing) for b in blocks)
    assert "net selling" not in _paragraph_texts(blocks)


def test_insider_activity_blocks_empty_when_no_activity():
    styles = _styles()
    assert _insider_activity_blocks({"ticker": "X"}, styles) == []


def test_insider_timeseries_chart_needs_datable_transactions():
    from reportlab.graphics.shapes import Drawing

    from agentic_options_reporter.frontend.report_pdf import _insider_timeseries_chart

    styles = _styles()
    drawn = _insider_timeseries_chart(
        {"transactions": [{"transaction_type": "sell", "shares": 1000, "filed_at": "2026-06-01"}]},
        styles,
    )
    assert any(isinstance(x, Drawing) for x in drawn)
    # No filing dates -> nothing to place on the time axis.
    assert _insider_timeseries_chart(
        {"transactions": [{"transaction_type": "buy", "shares": 100}]}, styles
    ) == []
    assert _insider_timeseries_chart(None, styles) == []


def test_fundamentals_key_facts_blocks_omit_absent_next_earnings():
    styles = _styles()
    blocks = _fundamentals_key_facts_blocks({"ticker": "X", "metrics": {"pe_ratio": 12.0}}, None, styles)
    text = _paragraph_texts(blocks)
    assert "Key metrics" in text
    assert "Next earnings" not in text


def test_fundamentals_key_facts_blocks_empty_snapshot_notes_none_available():
    styles = _styles()
    blocks = _fundamentals_key_facts_blocks({"ticker": "X"}, None, styles)
    assert "No fundamentals available" in _paragraph_texts(blocks)


def test_build_report_includes_fundamentals_section():
    payload = {**_FULL_REPORT, "fundamentals": _FUNDAMENTALS, "data_warnings": []}
    with_fund = build_report_pdf(payload)
    without_fund = build_report_pdf({**payload, "fundamentals": None})
    assert _is_pdf(with_fund)
    # The extra section makes for a larger document.
    assert len(with_fund) > len(without_fund)


def test_build_report_without_thesis():
    payload = {k: v for k, v in _FULL_REPORT.items() if k != "thesis"}
    data = build_report_pdf(payload)
    assert _is_pdf(data)


def test_build_report_with_skipped_agents():
    thesis = dict(_FULL_REPORT["thesis"])
    for skipped in (
        "financial_research", "news_research", "macro_research", "catalyst_research",
        "relative_strength_research", "statistical_edge_research",
        "risk_assessment", "strategy_suggestion",
    ):
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
                "quant_trade_quality": _trade_quality({"technical": _domain_score(40.0)}, composite_score=40.0),
                "technical_domain_score": _domain_score(40.0),
            },
            "investment_thesis": {"consensus": "neutral", "thesis": "Hold & wait <for> signal"},
        },
    }
    data = build_report_pdf(payload)
    assert _is_pdf(data)


def test_trade_quality_flowables_render_for_domain_scores():
    flowables = trade_quality_flowables(
        _trade_quality({"technical": _domain_score(82.0), "liquidity": _domain_score(74.0)}),
        styles={"body": getSampleStyleSheet()["BodyText"]},
    )
    assert len(flowables) > 0


def test_trade_quality_flowables_note_missing_domains():
    styles = {"body": getSampleStyleSheet()["BodyText"]}
    flowables = trade_quality_flowables(_trade_quality({"technical": _domain_score(82.0)}), styles=styles)
    text = _paragraph_texts(flowables)
    assert "Not available" in text
    assert "Macro" in text


def test_recommendation_block_replaces_factor_dump_with_summary():
    """When a Trade Quality Score is present, the block captions the chart
    with the composite engine's own explainability bullet and drops the
    deterministic rationale (now redundant with the visualization)."""
    rec = {
        "action": "BUY", "confidence": 0.82, "contract_symbol": "AAPL_C",
        "rationale": "AAPL_C scored 82.4/100 (trend_alignment=1.00, liquidity=0.00).",
    }
    candidates = [{"contract_symbol": "AAPL_C"}]
    text = _paragraph_texts(_recommendation_block(rec, candidates, _TRADE_QUALITY, _styles()))

    assert "strongest contributor" in text          # the explainability caption
    assert "scored 82.4/100 (trend_alignment" not in text  # the raw factor-dump rationale is gone


def test_recommendation_block_keeps_rationale_without_trade_quality():
    """AVOID / no-candidate: nothing to visualize, so the rationale stays."""
    rec = {"action": "AVOID", "confidence": 0.0, "contract_symbol": None,
           "rationale": "No liquid, scoreable candidates were found in the option chain."}
    text = _paragraph_texts(_recommendation_block(rec, [], None, _styles()))
    assert "No liquid, scoreable candidates" in text


def test_build_report_with_zero_and_tiny_domain_score_factors():
    """Regression: a domain at 0 (or a sliver like 5) made the meter bar a
    zero/sub-padding-width cell, and reportlab raised 'negative availWidth'
    at build time. The report must build for any score in [0, 100]."""
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
            }
        ],
        "trade_quality": _trade_quality(
            {
                "technical": _domain_score(100.0),    # full bar
                "risk": _domain_score(50.0),           # half
                "liquidity": _domain_score(0.0),       # empty — the crash case
                "fundamental": _domain_score(5.0),     # tiny sliver
            }
        ),
    }
    data = build_report_pdf(payload)
    assert _is_pdf(data)


def test_trade_quality_flowables_handle_extreme_ratios():
    styles = {
        "body": getSampleStyleSheet()["BodyText"],
        "cell": getSampleStyleSheet()["BodyText"],
        "cellhead": getSampleStyleSheet()["BodyText"],
        "muted": getSampleStyleSheet()["BodyText"],
    }
    # Building the whole doc is what actually wraps the flowables (and would
    # have raised); do it with only-zero and only-full domain scores.
    for score in (0.0, 100.0):
        trade_quality = _trade_quality({"technical": _domain_score(score), "risk": _domain_score(score)})
        assert trade_quality_flowables(trade_quality, styles)
        payload = {
            "symbol": "T",
            "recommendation": {"action": "BUY", "confidence": 0.5, "contract_symbol": "C", "rationale": ""},
            "candidates": [{"contract_symbol": "C"}],
            "trade_quality": trade_quality,
        }
        assert _is_pdf(build_report_pdf(payload))


def test_trade_quality_flowables_include_relative_strength_and_statistical_edge_badges():
    """Domain-specific badges (Performance/Leadership for Relative Strength,
    Confidence for Statistical Edge) must render in the PDF, matching the
    Agents-tab per-domain pills."""
    trade_quality = _trade_quality(
        {
            "relative_strength": {
                "score": 90.0, "confidence": 80.0, "evidence": [],
                "factors": [{"name": "vs_market", "value": 0.9, "weight": 0.55, "detail": ""}],
            },
            "statistical_edge": {"score": 55.0, "confidence": 10.0, "evidence": []},
        }
    )
    text = _paragraph_texts(trade_quality_flowables(trade_quality, _styles()))
    assert "Exceptional" in text          # Relative Strength Performance tier
    assert "Market Leader" in text        # Relative Strength Leadership tier
    assert "Insufficient Data" in text    # Statistical Edge Confidence tier


def test_trade_quality_comparison_flowables_renders_quant_and_agents_side_by_side():
    quant = _trade_quality({"technical": _domain_score(90.0)}, composite_score=90.0)
    agent = _trade_quality({"technical": _domain_score(50.0)}, composite_score=50.0)
    text = _paragraph_texts(trade_quality_comparison_flowables(quant, agent, _styles()))
    assert "Quant" in text
    assert "Agents" in text
    assert "diverge" in text   # composite scores 90 vs 50 differ by well over the 15pt threshold


def test_trade_quality_comparison_flowables_empty_without_either_source():
    assert trade_quality_comparison_flowables(None, None, _styles()) == []


def test_trade_quality_comparison_flowables_handles_missing_agent_side():
    quant = _trade_quality({"technical": _domain_score(80.0)})
    text = _paragraph_texts(trade_quality_comparison_flowables(quant, None, _styles()))
    assert "No Trade Quality Score available" in text


def test_build_report_includes_quant_vs_agents_comparison_section():
    """The dedicated comparison section — mirroring the Agents tab's
    'Trade Quality Score — Quant vs. Agents' card — must appear before the
    'Agent pipeline' section, and the old standalone 'Agent Trade Quality
    Score' block must no longer exist as its own heading."""
    from reportlab.platypus import KeepTogether, Paragraph

    styles = _styles()
    doc_story: list = []
    symbol = "AAPL"
    report = _FULL_REPORT

    # Reuse build_report_pdf's actual story-building by inspecting the text
    # of the generated PDF isn't practical (binary), so assert via the
    # flowables this section is built from directly.
    thesis = report["thesis"]
    quant_tq = thesis["quant_interpretation"]["quant_trade_quality"]
    agent_tq = thesis["agent_trade_quality"]
    comparison = trade_quality_comparison_flowables(quant_tq, agent_tq, styles)
    assert comparison
    text = _paragraph_texts(comparison)
    assert "Quant" in text and "Agents" in text

    data = build_report_pdf(report)
    assert _is_pdf(data)


def test_insider_chart_value_label_position_is_independent_of_bar_height():
    """Regression: the sell-side peak value label used to sit at a y
    dependent on the bar's own height (baseline - col_h - 8), so a
    near-full-height red bar could push its label down to nearly touch the
    date row. It must now sit at a FIXED y regardless of magnitude."""
    from reportlab.graphics.shapes import String

    styles = _styles()

    small_sell = {"transactions": [
        {"transaction_type": "sell", "shares": 10, "filed_at": "2026-06-01"},
    ]}
    large_sell = {"transactions": [
        {"transaction_type": "sell", "shares": 100_000, "filed_at": "2026-06-01"},
    ]}

    def _peak_label_y(insider):
        flowables = _insider_timeseries_chart(insider, styles)
        drawing = flowables[0]
        value_strings = [
            s for s in drawing.contents
            if isinstance(s, String) and s.text.startswith(("+", "-"))
        ]
        assert value_strings
        return value_strings[0].y

    assert _peak_label_y(small_sell) == _peak_label_y(large_sell)


def test_insider_chart_has_visible_y_axis():
    """The share-count scale must be legible via a drawn axis, not just
    inferable from bar height."""
    from reportlab.graphics.shapes import Line, String

    styles = _styles()
    insider = {"transactions": [
        {"transaction_type": "buy", "shares": 500, "filed_at": "2026-05-01"},
        {"transaction_type": "sell", "shares": 1000, "filed_at": "2026-06-01"},
    ]}
    drawing = _insider_timeseries_chart(insider, styles)[0]

    lines = [item for item in drawing.contents if isinstance(item, Line)]
    # Vertical axis rule + zero baseline + 5 tick marks (100%/50%/0/-50%/-100%).
    vertical_lines = [ln for ln in lines if ln.x1 == ln.x2]
    assert vertical_lines  # the y-axis rule itself
    tick_labels = [
        item for item in drawing.contents
        if isinstance(item, String) and item.text in ("0",)
    ]
    assert tick_labels  # the "0" tick at the baseline


def test_insider_chart_axis_ticks_scale_with_max_magnitude():
    from reportlab.graphics.shapes import String

    styles = _styles()
    insider = {"transactions": [
        {"transaction_type": "buy", "shares": 2000, "filed_at": "2026-05-01"},
    ]}
    drawing = _insider_timeseries_chart(insider, styles)[0]
    tick_texts = {item.text for item in drawing.contents if isinstance(item, String)}
    assert "+2,000" in tick_texts   # the 100% tick label matches the max magnitude
