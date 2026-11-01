"""Tests for Report schema serialization, deserialization, and round-trip integrity."""

from __future__ import annotations

import json

try:
    import pytest
except ImportError:
    pytest = None

from docanalyzer.product.schema import (
    Check,
    DocumentInfo,
    Evidence,
    Field,
    MetaInfo,
    Report,
    TenureInfo,
)


def test_report_schema_round_trip():
    """Verify Report serializes to dict/json and deserializes back faithfully."""
    doc_info = DocumentInfo(
        type="loan",
        type_confidence=0.85,
        title="Term Loan Agreement",
        n_chars=5400,
    )
    parties = [{"name": "First Bank and borrower", "evidence": {"start_char": 10, "end_char": 30, "text": "First Bank"}}]
    field_eff = Field(
        name="Effective Date",
        raw_text="10th June 2024",
        evidence=Evidence(start_char=50, end_char=64, text="10th June 2024"),
        score=8.5,
        reliability="high",
        status="found",
    )
    dates = {"Effective Date": field_eff}
    tenure = TenureInfo(months=36.0, days=1095, source="from dates", note="Standard loan term")
    key_terms = {
        "Governing Law": Field(
            name="Governing Law",
            raw_text="laws of India",
            evidence=Evidence(start_char=100, end_char=113, text="laws of India"),
            score=9.2,
            reliability="high",
        )
    }
    check = Check(
        id="G1",
        title="Governing Law Check",
        severity="info",
        why="Jurisdiction check",
        message="Local law applied",
        status="active",
        evidence=Evidence(start_char=100, end_char=113, text="laws of India"),
    )
    meta = MetaInfo(
        option_name="CUAD (US contracts)",
        model_id="roberta-base-squad2",
        app_version="0.1.0",
        created_at="2026-11-01T12:00:00Z",
    )

    report = Report(
        document=doc_info,
        parties=parties,
        dates=dates,
        tenure=tenure,
        key_terms=key_terms,
        checks=[check],
        summary="Test summary sentence.",
        meta=meta,
    )

    # Serialize to JSON and parse back
    report_dict = report.to_dict()
    json_str = json.dumps(report_dict)
    loaded_dict = json.loads(json_str)

    restored_report = Report.from_dict(loaded_dict)

    assert restored_report.document.title == "Term Loan Agreement"
    assert restored_report.document.type == "loan"
    assert restored_report.tenure.months == 36.0
    assert restored_report.dates["Effective Date"].raw_text == "10th June 2024"
    assert restored_report.checks[0].id == "G1"
    assert restored_report.checks[0].evidence.text == "laws of India"
