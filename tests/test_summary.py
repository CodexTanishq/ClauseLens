"""Tests for template-based contract summary generation."""

from __future__ import annotations

try:
    import pytest
except ImportError:
    pytest = None

from docanalyzer.product.schema import DocumentInfo, Evidence, Field, TenureInfo
from docanalyzer.product.summary import generate_summary


def test_summary_derivation():
    """Verify summary is constructed strictly from provided fields without hallucination."""
    doc_info = DocumentInfo(
        type="rental_lease",
        type_confidence=0.9,
        title="Residential Agreement",
        n_chars=1200,
    )
    parties_text = "Acme Realty and John Doe"
    eff = Field(
        name="Effective Date",
        raw_text="1st May 2024",
        evidence=Evidence(0, 12, "1st May 2024"),
        reliability="high",
    )
    exp = Field(
        name="Expiration Date",
        raw_text="31st March 2025",
        evidence=Evidence(20, 35, "31st March 2025"),
        reliability="high",
    )
    dates = {"Effective Date": eff, "Expiration Date": exp}
    tenure = TenureInfo(months=11.0, days=334, source="from dates")
    gov_law = Field(
        name="Governing Law",
        raw_text="laws of India",
        evidence=Evidence(50, 63, "laws of India"),
        reliability="high",
    )

    summary = generate_summary(
        doc_info=doc_info,
        parties_text=parties_text,
        dates=dates,
        tenure=tenure,
        gov_law=gov_law,
        checks_count=2,
    )

    assert "lease or rental agreement" in summary
    assert "Acme Realty and John Doe" in summary
    assert "1st May 2024" in summary
    assert "31st March 2025" in summary
    assert "laws of India" in summary
    assert "2 clauses" in summary or "flagged 2" in summary

    # Sentence count should be between 3 and 5
    sentences = [s.strip() for s in summary.split(".") if s.strip()]
    assert 3 <= len(sentences) <= 5
