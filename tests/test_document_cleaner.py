"""Tests for document extraction, text cleaning rules, and QA validation utilities."""

from __future__ import annotations

try:
    import pytest
except ImportError:
    pytest = None

from docanalyzer.document_cleaner import clean_text, extract_jurisdiction_hint, validate_answer


def test_dehyphenation():
    """Verify words broken across a line break with a hyphen are joined cleanly."""
    text = "The dis-\ntributor shall deliver the pro-\nducts on time."
    cleaned, report = clean_text(text)
    assert "distributor" in cleaned
    assert "products" in cleaned
    assert "dis-" not in cleaned
    assert report.hyphenated_words_joined_step2 == 2


def test_page_header_footer_removal():
    """Verify repeated header/footer lines across 3+ pages are stripped."""
    page_1 = "CONFIDENTIAL HEADER\nFirst body line of contract.\nMiddle clause text.\nPage 1 of 3"
    page_2 = "CONFIDENTIAL HEADER\nSecond body line of terms.\nMiddle clause text.\nPage 2 of 3"
    page_3 = "CONFIDENTIAL HEADER\nThird body line of signature.\nMiddle clause text.\nPage 3 of 3"
    raw_doc = "\f".join([page_1, page_2, page_3])

    cleaned, report = clean_text(raw_doc)
    assert "CONFIDENTIAL HEADER" not in cleaned
    assert "Page 1 of 3" not in cleaned
    assert "First body line of contract" in cleaned
    assert "Second body line of terms" in cleaned
    assert report.page_headers_footers_removed_step3 > 0


def test_clause_numbering_kept():
    """Verify structured clause headers and numbers retain their distinct lines."""
    text = (
        "1. DEFINITIONS\n"
        "In this Agreement words shall have the following meanings.\n"
        "1.1 'Company' means Acme Inc.\n"
        "(a) Any subsidiary thereof;\n"
        "Section 2. GOVERNING LAW\n"
        "The laws of India shall apply."
    )
    cleaned, _ = clean_text(text)
    lines = cleaned.split("\n")
    # Verify clause numbers are at the start of their respective lines
    assert any(l.startswith("1. DEFINITIONS") for l in lines)
    assert any(l.startswith("1.1 'Company'") for l in lines)
    assert any(l.startswith("(a) Any subsidiary") for l in lines)
    assert any("Section 2. GOVERNING LAW" in l for l in lines)


def test_validate_answer():
    """Verify context[start:start+len(text)] strict matching."""
    context = "The Effective Date is 1st January 2025 as agreed."
    answer = "1st January 2025"
    start = context.find(answer)

    assert validate_answer(context, answer, start) is True
    assert validate_answer(context, answer, start + 1) is False
    assert validate_answer(context, "Wrong Text", start) is False


def test_jurisdiction_hint():
    """Verify detection of Indian vs US markers and governing law."""
    india_text = (
        "This agreement executed in Mumbai, Maharashtra under the Indian Contract Act. "
        "The stamp duty is paid. Total rent Rs. 50,000. "
        "This contract is governed by the laws of India and subject to Mumbai courts."
    )
    hint = extract_jurisdiction_hint(india_text)
    assert hint["label"] == "india-likely"
    assert hint["india_hits"] >= 3
    assert "India" in hint["governing_law"]
