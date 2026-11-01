"""Field specifications, CUAD question category mappings, and planned field definitions."""

from __future__ import annotations

from typing import Any, Dict, List

COMMON_AVAILABLE_FIELDS: List[str] = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Termination For Convenience",
    "Anti-Assignment",
    "Cap On Liability",
    "Uncapped Liability",
]

EMPLOYMENT_AVAILABLE_FIELDS: List[str] = COMMON_AVAILABLE_FIELDS + [
    "Non-Compete",
    "No-Solicit Of Employees",
    "No-Solicit Of Customers",
    "Liquidated Damages",
    "Ip Ownership Assignment",
]

PLANNED_FIELDS: Dict[str, List[str]] = {
    "rental_lease": [
        "Rent Amount",
        "Rent Due Date",
        "Security Deposit",
        "Deposit Refund Terms",
        "Lock-in Period",
        "Notice Period",
        "Rent Escalation",
        "Maintenance Responsibility",
        "Sub-letting",
        "Permitted Use",
    ],
    "employment": [
        "Job Title",
        "Salary",
        "Variable Pay",
        "Probation Period",
        "Notice Period (Employee)",
        "Notice Period (Employer)",
        "Working Hours",
        "Leave",
        "Bond Or Training Cost",
        "Confidentiality",
    ],
    "loan": [
        "Loan Amount",
        "Interest Rate",
        "Rate Type",
        "Loan Tenure",
        "EMI Amount",
        "Processing Fee",
        "Prepayment Charges",
        "Penal Charges",
        "Security Or Collateral",
        "Guarantor",
        "Events Of Default",
        "Right To Change Rate",
    ],
}


def get_field_specs_for_type(doc_type: str) -> Dict[str, Any]:
    """Retrieve available CUAD questions and planned fields for a document type."""
    if doc_type == "employment":
        available = EMPLOYMENT_AVAILABLE_FIELDS
    else:
        available = COMMON_AVAILABLE_FIELDS

    planned = PLANNED_FIELDS.get(doc_type, [])

    return {
        "doc_type": doc_type,
        "available_questions": available,
        "planned_fields": planned,
    }

