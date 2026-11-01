"""Template-based document summary generator using solely verified extracted fields."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from docanalyzer.product.schema import DocumentInfo, Field, TenureInfo


def generate_summary(
    doc_info: DocumentInfo,
    parties_text: str,
    dates: Dict[str, Optional[Field]],
    tenure: TenureInfo,
    gov_law: Optional[Field],
    checks_count: int,
) -> str:
    """Compose a factual 3 to 5 sentence summary from extracted fields without hallucinations."""
    sentences: List[str] = []

    # Sentence 1: Document classification and title
    doc_type_names = {
        "rental_lease": "lease or rental agreement",
        "loan": "loan or credit facility agreement",
        "employment": "employment agreement or appointment letter",
        "other": "commercial contract",
    }
    type_name = doc_type_names.get(doc_info.type, "commercial contract")
    title_snippet = f" titled '{doc_info.title}'" if doc_info.title else ""
    sentences.append(f"This document appears to be a {type_name}{title_snippet}.")

    # Sentence 2: Parties
    if parties_text.strip():
        sentences.append(f"The identified parties to the contract include: {parties_text.strip()}.")
    else:
        sentences.append("Contracting parties could not be clearly verified from the text.")

    # Sentence 3: Tenure and effective dates
    eff = dates.get("Effective Date")
    exp = dates.get("Expiration Date")
    if eff and eff.raw_text.strip() and exp and exp.raw_text.strip():
        tenure_desc = f" ({tenure.months:.0f} months)" if tenure.months else ""
        low_tag = " (unverified/low confidence)" if eff.reliability == "low" or exp.reliability == "low" else ""
        sentences.append(
            f"The agreement takes effect on {eff.raw_text.strip()} and expires on {exp.raw_text.strip()}{tenure_desc}{low_tag}."
        )
    elif eff and eff.raw_text.strip():
        sentences.append(f"The agreement states an effective date of {eff.raw_text.strip()}.")
    elif tenure.months:
        sentences.append(f"The duration is specified as approximately {tenure.months:.0f} months.")

    # Sentence 4: Governing law
    if gov_law and gov_law.raw_text.strip():
        low_tag = " (low confidence)" if gov_law.reliability == "low" else ""
        sentences.append(f"Disputes and governing law are designated under {gov_law.raw_text.strip()}{low_tag}.")

    # Sentence 5: Flagged issues count
    if checks_count > 0:
        sentences.append(f"Analysis flagged {checks_count} clauses or structural items worth verifying.")

    return " ".join(sentences)

