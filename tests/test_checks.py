"""Tests for all active contract checks and planned check specifications."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import pytest
except ImportError:
    pytest = None

from docanalyzer.product.checks import (
    check_e1_post_employment_noncompete,
    check_e2_bond,
    check_e3_benefits_not_mentioned,
    check_g1_governing_law,
    check_g2_auto_renewal,
    check_g3_early_termination,
    check_g4_date_sanity,
    check_g5_liability,
    check_g6_signature_block,
    check_g7_assignment,
    check_l1_lender_can_change,
    check_l2_acceleration,
    check_r1_rental_registration,
    check_r2_rental_stamp,
    run_checks,
)
from docanalyzer.product.schema import Evidence, Field


def _make_field(name: str, text: str, parsed: Any = None) -> Field:
    return Field(
        name=name,
        raw_text=text,
        evidence=Evidence(0, len(text), text),
        parsed=parsed,
        score=10.0,
        reliability="high",
        status="found",
    )


# G1: Governing Law
def test_check_g1_foreign_law():
    foreign = _make_field("Governing Law", "laws of the State of Delaware")
    indian = _make_field("Governing Law", "laws of India and courts in Mumbai")

    assert check_g1_governing_law(foreign, "text") is not None
    assert check_g1_governing_law(indian, "text") is None


# G2: Auto Renewal
def test_check_g2_auto_renewal():
    renewal = _make_field("Renewal Term", "The agreement shall automatically renew for 1 year.")
    short_notice = _make_field("Notice Period To Terminate Renewal", "15 days")
    long_notice = _make_field("Notice Period To Terminate Renewal", "60 days")

    assert check_g2_auto_renewal(renewal, short_notice, "text") is not None
    assert check_g2_auto_renewal(renewal, long_notice, "text") is None


# G3: Early Termination
def test_check_g3_early_termination():
    found = _make_field("Termination For Convenience", "Either party may terminate upon 30 days notice.")
    assert check_g3_early_termination(found) is not None
    assert check_g3_early_termination(None) is None


# G4: Date Sanity
def test_check_g4_date_sanity():
    eff_ok = _make_field("Effective Date", "01/05/2024", parsed={"iso": "2024-05-01"})
    exp_ok = _make_field("Expiration Date", "31/12/2024", parsed={"iso": "2024-12-31"})
    exp_bad = _make_field("Expiration Date", "01/01/2024", parsed={"iso": "2024-01-01"})

    assert check_g4_date_sanity({"Effective Date": eff_ok, "Expiration Date": exp_bad}) is not None
    assert check_g4_date_sanity({"Effective Date": eff_ok, "Expiration Date": exp_ok}) is None


# G5: Liability
def test_check_g5_liability():
    cap = _make_field("Cap On Liability", "Liability is capped at $10,000.")
    assert check_g5_liability(cap, None) is not None
    assert check_g5_liability(None, None) is None


# G6: Signature Block
def test_check_g6_signature():
    assert check_g6_signature_block("Draft memo without closing words.") is not None
    assert check_g6_signature_block("In witness whereof, the parties have signed.") is None


# G7: Assignment
def test_check_g7_assignment():
    assign = _make_field("Anti-Assignment", "Neither party may assign this agreement.")
    assert check_g7_assignment(assign) is not None
    assert check_g7_assignment(None) is None


# R1: Rental Registration
def test_check_r1_rental_registration():
    # Long lease without registration
    assert check_r1_rental_registration(tenure_months=24.0, text="Plain contract.") is not None
    # Long lease with registration
    assert check_r1_rental_registration(tenure_months=24.0, text="Agreement shall be registered.") is None
    # 11-month lease
    assert check_r1_rental_registration(tenure_months=11.0, text="Plain contract.") is None


# R2: Rental Stamp
def test_check_r2_rental_stamp():
    assert check_r2_rental_stamp("Agreement with no mention of tax.") is not None
    assert check_r2_rental_stamp("Stamp duty has been duly paid.") is None


# E1: Employment Non-Compete
def test_check_e1_non_compete():
    text_bad = "Employee agrees to a non-compete covenant following the end of employment."
    text_ok = "Employee agrees to devote full working hours to the Company."
    assert check_e1_post_employment_noncompete(None, text_bad) is not None
    assert check_e1_post_employment_noncompete(None, text_ok) is None


# E2: Employment Bond
def test_check_e2_bond():
    assert check_e2_bond("Candidate shall execute an employment bond of Rs. 1,00,000.") is not None
    assert check_e2_bond("Standard compensation agreement.") is None


# E3: Benefits Not Mentioned
def test_check_e3_benefits():
    assert check_e3_benefits_not_mentioned("Compensation is purely fixed monthly sum.") is not None
    assert check_e3_benefits_not_mentioned("Employee is eligible for Provident Fund and ESI.") is None


# L1: Loan Lender Discretion
def test_check_l1_lender_discretion():
    assert check_l1_lender_can_change("Lender may at any time revise interest rates at its sole discretion.") is not None
    assert check_l1_lender_can_change("Interest is fixed at 8.5% for the tenure.") is None


# L2: Loan Acceleration
def test_check_l2_acceleration():
    assert check_l2_acceleration("Upon event of default, all dues shall become immediately due and payable.") is not None
    assert check_l2_acceleration("Payments follow the standard monthly schedule.") is None


# Planned Checks
def test_planned_checks_not_checked():
    """Verify planned checks are registered with status 'not_checked'."""
    checks = run_checks("rental_lease", {}, {}, None, "Sample text")
    planned = [c for c in checks if c.status == "not_checked"]
    assert len(planned) > 0
    for p in planned:
        assert p.status == "not_checked"
