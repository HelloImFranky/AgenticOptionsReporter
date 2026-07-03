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


def test_format_recommendation_with_contract():
    text = format_recommendation(
        {
            "action": "BUY",
            "contract_symbol": "AAPL260116C00150000",
            "confidence": 0.732,
            "rationale": "strong trend alignment",
        }
    )
    assert "BUY" in text
    assert "73%" in text
    assert "AAPL260116C00150000" in text
    assert "strong trend alignment" in text


def test_format_recommendation_without_contract():
    text = format_recommendation(
        {"action": "AVOID", "contract_symbol": None, "confidence": 0.0, "rationale": "no candidates"}
    )
    assert "AVOID" in text
    assert "—" in text


def test_format_trend_summary():
    text = format_trend_summary({"direction": "bullish", "strength": "strong", "adx": 42.567})
    assert "Bullish" in text
    assert "strong" in text
    assert "42.6" in text


def test_format_volume_summary_with_flags():
    text = format_volume_summary({"relative_volume": 1.8, "flags": ["high_volume"]})
    assert "1.80x" in text
    assert "high_volume" in text


def test_format_volume_summary_no_flags():
    text = format_volume_summary({"relative_volume": 1.0, "flags": []})
    assert "none" in text


def test_format_indicator_summary():
    text = format_indicator_summary(
        {"sma_20": 101.234, "sma_50": 98.5, "rsi_14": 55.1, "atr_14": 2.345}
    )
    assert "101.23" in text
    assert "98.50" in text
    assert "55.1" in text
    assert "2.35" in text


def test_candidates_to_rows_shapes_match_columns():
    rows = candidates_to_rows(
        [
            {
                "contract_symbol": "AAPL260116C00150000",
                "option_type": "call",
                "strike": 150.0,
                "expiration": "2026-01-16",
                "score": 78.456,
                "delta": 0.567,
                "breakeven": 152.5,
                "max_loss": 250.0,
                "max_gain": None,
                "probability_of_profit": 0.612,
            }
        ]
    )
    assert len(rows) == 1
    assert len(rows[0]) == len(CANDIDATE_COLUMNS)
    assert rows[0][0] == "AAPL260116C00150000"
    assert rows[0][1] == "CALL"
    assert rows[0][8] == "unlimited"
    assert rows[0][9] == "61%"


def test_candidates_to_rows_with_capped_max_gain():
    rows = candidates_to_rows(
        [
            {
                "contract_symbol": "AAPL260116P00150000",
                "option_type": "put",
                "strike": 150.0,
                "expiration": "2026-01-16",
                "score": 40.0,
                "delta": -0.3,
                "breakeven": 147.5,
                "max_loss": 250.0,
                "max_gain": 14750.0,
                "probability_of_profit": 0.4,
            }
        ]
    )
    assert rows[0][8] == "14750.00"


def test_runs_to_rows_shapes_match_columns():
    rows = runs_to_rows(
        [
            {
                "run_id": 1,
                "symbol": "AAPL",
                "generated_at": "2026-07-03T12:00:00",
                "recommendation_action": "BUY",
                "recommendation_confidence": 0.65,
            }
        ]
    )
    assert len(rows) == 1
    assert len(rows[0]) == len(RUN_COLUMNS)
    assert rows[0] == ["1", "AAPL", "2026-07-03T12:00:00", "BUY", "65%"]


def test_empty_lists_produce_no_rows():
    assert candidates_to_rows([]) == []
    assert runs_to_rows([]) == []
