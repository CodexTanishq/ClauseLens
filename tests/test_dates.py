"""Tests for Indian day-first date parsing, duration extraction, and tenure calculation."""

from __future__ import annotations

try:
    import pytest
except ImportError:
    pytest = None

from docanalyzer.product.dates import compute_tenure, parse_date, parse_duration


def test_parse_date_indian_day_first():
    """Verify standard Indian date formats parse with day-first convention."""
    res = parse_date("05/10/2024")
    assert res is not None
    # 5th October 2024, not 10th May 2024
    assert res["day"] == 5
    assert res["month"] == 10
    assert res["year"] == 2024


def test_parse_date_ambiguity_flag():
    """Verify ambiguity flag is set when both day and month <= 12 and no month name is written."""
    # 04/05/2024 could be 4th May or 5th April
    ambig = parse_date("04/05/2024")
    assert ambig is not None
    assert ambig["ambiguous"] is True

    # 15/05/2024 cannot be ambiguous because day=15 > 12
    unambig = parse_date("15/05/2024")
    assert unambig is not None
    assert unambig["ambiguous"] is False

    # Text month name removes ambiguity
    text_month = parse_date("4th May 2024")
    assert text_month is not None
    assert text_month["ambiguous"] is False


def test_parse_duration():
    """Verify extraction of duration in months from words and numbers."""
    assert parse_duration("eleven (11) months") == 11.0
    assert parse_duration("11 months") == 11.0
    assert parse_duration("one year") == 12.0
    assert parse_duration("3 years") == 36.0
    assert parse_duration("24 months") == 24.0


def test_compute_tenure_from_dates():
    """Verify tenure calculation when start and end dates are supplied."""
    tenure = compute_tenure(
        effective_text="1st May 2024",
        expiration_text="31st March 2025",
    )
    assert tenure["source"] == "from dates"
    assert tenure["months"] is not None
    assert 10.5 <= tenure["months"] <= 11.5


def test_compute_tenure_from_duration_text():
    """Verify fallback to duration text when full dates are not available."""
    tenure = compute_tenure(
        effective_text="1st May 2024",
        expiration_text=None,
        duration_text="The license shall continue for a term of eleven (11) months.",
    )
    assert tenure["source"] == "from duration text"
    assert tenure["months"] == 11.0
