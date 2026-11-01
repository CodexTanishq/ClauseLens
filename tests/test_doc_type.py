"""Tests for rule-based document type classification."""

from __future__ import annotations

try:
    import pytest
except ImportError:
    pytest = None

from docanalyzer.product.doc_type import classify_document_type


def test_classify_rental_lease():
    text = (
        "RESIDENTIAL LEASE AGREEMENT\n"
        "This lease is made between the Landlord and Tenant for the premises. "
        "The Tenant agrees to pay rent and provide a security deposit."
    )
    doc_type, conf = classify_document_type(text, title="RESIDENTIAL LEASE AGREEMENT")
    assert doc_type == "rental_lease"
    assert conf > 0.4


def test_classify_loan():
    text = (
        "FACILITY AGREEMENT\n"
        "The Borrower agrees to borrow the Principal amount from the Lender. "
        "Repayment shall be made via monthly EMI with interest."
    )
    doc_type, conf = classify_document_type(text, title="FACILITY AGREEMENT")
    assert doc_type == "loan"
    assert conf > 0.4


def test_classify_employment():
    text = (
        "OFFER OF EMPLOYMENT\n"
        "We are pleased to appoint you as Software Engineer. "
        "Your salary and CTC are detailed below. The probation period is 6 months."
    )
    doc_type, conf = classify_document_type(text, title="OFFER OF EMPLOYMENT")
    assert doc_type == "employment"
    assert conf > 0.4


def test_classify_other():
    text = "General nondescript memo discussing quarterly business topics without specific terms."
    doc_type, conf = classify_document_type(text, title="Memorandum")
    assert doc_type == "other"
    assert conf == 0.0
