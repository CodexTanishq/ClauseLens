"""Automated contract checks, rule evaluation, and evidence extraction."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dateutil import parser as dt_parser

from docanalyzer.product.schema import Check, Evidence, Field

logger = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).resolve().parent / "checks_catalog.yaml"

INDIA_GEO_MARKERS = [
    r"\bindia\b", r"\bindian\b", r"\bindian\s+contract\s+act\b",
    r"\bmumbai\b", r"\bdelhi\b", r"\bbengaluru\b", r"\bbangalore\b",
    r"\bchennai\b", r"\bkolkata\b", r"\bhyderabad\b", r"\bpune\b",
    r"\bgurgaon\b", r"\bgurugram\b", r"\bnoida\b", r"\bahmedabad\b",
    r"\bmaharashtra\b", r"\bkarnataka\b", r"\btamil\s+nadu\b",
    r"\btelangana\b", r"\bgujarat\b", r"\buttar\s+pradesh\b",
    r"\bwest\s+bengal\b", r"\bharyana\b", r"\bkerala\b",
]


def load_catalog() -> Dict[str, Any]:
    """Load check definitions from checks_catalog.yaml."""
    if not CATALOG_PATH.exists():
        return {"active_checks": [], "planned_checks": []}
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_sentence_evidence(text: str, match_span: Tuple[int, int]) -> Evidence:
    """Find the surrounding sentence boundaries for a character span."""
    start_pos, end_pos = match_span
    # Search backwards for sentence start
    s_start = max(0, text.rfind("\n", 0, start_pos) + 1)
    period_start = text.rfind(". ", 0, start_pos)
    if period_start != -1 and period_start + 2 > s_start:
        s_start = period_start + 2

    # Search forward for sentence end
    s_end = text.find(". ", end_pos)
    newline_end = text.find("\n", end_pos)
    candidates = [c for c in [s_end, newline_end, len(text)] if c != -1]
    s_end = min(candidates) if candidates else len(text)
    if s_end < len(text) and text[s_end] == ".":
        s_end += 1

    return Evidence(
        start_char=s_start,
        end_char=s_end,
        text=text[s_start:s_end].strip(),
    )


# -----------------------------------------------------------------------------
# Active Check Functions
# -----------------------------------------------------------------------------

def check_g1_governing_law(gov_field: Optional[Field], text: str) -> Optional[Check]:
    if not gov_field or not gov_field.raw_text.strip():
        return None
    law_text = gov_field.raw_text.lower()
    has_india = any(re.search(p, law_text, re.IGNORECASE) for p in INDIA_GEO_MARKERS)
    if not has_india:
        return Check(
            id="G1",
            title="Foreign or non-local governing law",
            severity="check",
            why="Disputes follow this law and these courts. A foreign law or a far-away court makes a dispute expensive.",
            message=f"Governing law cited: '{gov_field.raw_text[:80]}'. It does not appear to reference Indian jurisdiction.",
            status="active",
            evidence=gov_field.evidence,
        )
    return None


def check_g2_auto_renewal(
    renewal_field: Optional[Field],
    notice_field: Optional[Field],
    text: str,
) -> Optional[Check]:
    r_text = (renewal_field.raw_text if renewal_field else "").lower()
    m = re.search(r"\b(auto|automatic|automatically|renew)\b", r_text)
    if not m and not re.search(r"\bautomatically\s+renew\b", text[:5000], re.IGNORECASE):
        return None

    notice_days = None
    if notice_field and notice_field.raw_text:
        d_match = re.search(r"(\d+)\s*days?", notice_field.raw_text, re.IGNORECASE)
        if d_match:
            notice_days = int(d_match.group(1))

    if notice_days is None or notice_days < 30:
        notice_desc = f"{notice_days} days" if notice_days is not None else "unstated or missing"
        ev = renewal_field.evidence if renewal_field else extract_sentence_evidence(text, (0, min(100, len(text))))
        return Check(
            id="G2",
            title="Automatic renewal clause",
            severity="check",
            why="The contract may renew by itself unless you tell them in time.",
            message=f"The contract contains automatic renewal language. Notice required to prevent renewal: {notice_desc}.",
            status="active",
            evidence=ev,
        )
    return None


def check_g3_early_termination(term_field: Optional[Field]) -> Optional[Check]:
    if term_field and term_field.raw_text.strip():
        return Check(
            id="G3",
            title="Early termination for convenience",
            severity="info",
            why="Allows either party or one party to end the agreement early without showing a breach.",
            message=f"Early termination rights found: '{term_field.raw_text[:120]}'.",
            status="active",
            evidence=term_field.evidence,
        )
    return None


def check_g4_date_sanity(dates: Dict[str, Optional[Field]]) -> Optional[Check]:
    eff_field = dates.get("Effective Date")
    exp_field = dates.get("Expiration Date")
    agr_field = dates.get("Agreement Date")

    if not eff_field or not exp_field or not eff_field.parsed or not exp_field.parsed:
        return None

    try:
        d_eff = dt_parser.parse(eff_field.parsed["iso"]).date()
        d_exp = dt_parser.parse(exp_field.parsed["iso"]).date()

        if d_exp < d_eff:
            return Check(
                id="G4",
                title="Inconsistent or backward dates",
                severity="important",
                why="Dates that run backwards are often typing mistakes and can cause disputes.",
                message=f"Expiration date ({d_exp}) occurs prior to the effective date ({d_eff}).",
                status="active",
                evidence=exp_field.evidence,
            )

        if agr_field and agr_field.parsed:
            d_agr = dt_parser.parse(agr_field.parsed["iso"]).date()
            if d_exp < d_agr:
                return Check(
                    id="G4",
                    title="Inconsistent or backward dates",
                    severity="important",
                    why="Dates that run backwards are often typing mistakes and can cause disputes.",
                    message=f"Expiration date ({d_exp}) occurs prior to agreement date ({d_agr}).",
                    status="active",
                    evidence=exp_field.evidence,
                )
    except Exception as e:
        logger.debug("Date sanity check error: %s", e)

    return None


def check_g5_liability(cap_field: Optional[Field], uncapped_field: Optional[Field]) -> Optional[Check]:
    if cap_field and cap_field.raw_text.strip():
        return Check(
            id="G5",
            title="Liability limitation clause",
            severity="info",
            why="Limits how much one or both parties can recover in damages.",
            message=f"Liability cap clause found: '{cap_field.raw_text[:120]}'.",
            status="active",
            evidence=cap_field.evidence,
        )
    if uncapped_field and uncapped_field.raw_text.strip() and uncapped_field.reliability != "low":
        return Check(
            id="G5",
            title="Uncapped liability terms",
            severity="check",
            why="Limits how much one or both parties can recover in damages.",
            message=f"Uncapped liability terms identified: '{uncapped_field.raw_text[:120]}'.",
            status="active",
            evidence=uncapped_field.evidence,
        )
    return None


def check_g6_signature_block(text: str) -> Optional[Check]:
    patterns = [
        r"\bwitness\b",
        r"\bsigned\b",
        r"\bsignature\b",
        r"in\s+witness\s+whereof",
        r"\bexecuted\b",
    ]
    has_sig = any(re.search(p, text, re.IGNORECASE) for p in patterns)
    if not has_sig:
        return Check(
            id="G6",
            title="Missing signing or execution block",
            severity="check",
            why="No signing text was found. This may be an unsigned draft, or the signature page is an image we cannot read.",
            message="No standard execution or signature block detected in the document.",
            status="active",
            evidence=None,
        )
    return None


def check_g7_assignment(anti_assign_field: Optional[Field]) -> Optional[Check]:
    if anti_assign_field and anti_assign_field.raw_text.strip():
        return Check(
            id="G7",
            title="Restriction on assignment",
            severity="info",
            why="Governs whether duties or rights can be transferred to a third party.",
            message=f"Assignment restrictions found: '{anti_assign_field.raw_text[:120]}'.",
            status="active",
            evidence=anti_assign_field.evidence,
        )
    return None


def check_r1_rental_registration(tenure_months: Optional[float], text: str) -> Optional[Check]:
    has_regist = bool(re.search(r"\bregist(?:er|ration|ered)\b", text, re.IGNORECASE))
    if tenure_months is not None and tenure_months > 12.0 and not has_regist:
        return Check(
            id="R1",
            title="Lease registration requirement",
            severity="important",
            why="Leases longer than a year normally have to be registered. Many rent agreements run 11 months for this reason.",
            message=f"Tenure ({tenure_months:.0f} months) exceeds 12 months, but no reference to registration was found.",
            status="active",
            evidence=None,
        )

    # Maharashtra specific check
    mentions_mh = bool(re.search(r"\b(maharashtra|mumbai|pune)\b", text, re.IGNORECASE))
    mentions_lic = bool(re.search(r"\bleave\s+and\s+licen[cs]e\b", text, re.IGNORECASE))
    if mentions_mh and mentions_lic and not has_regist:
        return Check(
            id="R1",
            title="Maharashtra Leave and License registration check",
            severity="check",
            why="In Maharashtra, all Leave and License agreements are required by law to be registered regardless of duration.",
            message="Agreement references Maharashtra leave and license, but no registration terms are mentioned.",
            status="active",
            evidence=None,
        )
    return None


def check_r2_rental_stamp(text: str) -> Optional[Check]:
    has_stamp = bool(re.search(r"\bstamp(?:\s+duty|\s+paper)?\b", text, re.IGNORECASE))
    if not has_stamp:
        return Check(
            id="R2",
            title="Stamp duty reference missing",
            severity="check",
            why="Rent agreements are normally made on stamp paper or e-stamped. The stamp duty differs by state.",
            message="No mention of stamp duty or stamp paper found in the document.",
            status="active",
            evidence=None,
        )
    return None


def check_e1_post_employment_noncompete(non_compete_field: Optional[Field], text: str) -> Optional[Check]:
    has_nc = False
    nc_ev = None
    if non_compete_field and non_compete_field.raw_text.strip() and non_compete_field.reliability != "low":
        has_nc = True
        nc_ev = non_compete_field.evidence
    else:
        m = re.search(r"\b(non-compete|not\s+to\s+join|not\s+to\s+work\s+for\s+a\s+competitor)\b", text, re.IGNORECASE)
        if m:
            has_nc = True
            nc_ev = extract_sentence_evidence(text, m.span())

    if has_nc:
        post_pats = [r"after\s+termination", r"post-employment", r"following\s+the\s+end\s+of\s+employment"]
        if any(re.search(p, text, re.IGNORECASE) for p in post_pats):
            return Check(
                id="E1",
                title="Post-employment non-compete clause",
                severity="important",
                why="Rules that stop you from working elsewhere after you leave are often not enforceable in India, but the clause can still be used to pressure you. Read it.",
                message="Post-employment restrictive covenant detected.",
                status="active",
                evidence=nc_ev,
            )
    return None


def check_e2_bond(text: str) -> Optional[Check]:
    bond_pats = [
        r"\bbond\b",
        r"\bliquidated\s+damages\b",
        r"reimburse[^\.\n;]{0,40}training",
        r"recover[^\.\n;]{0,40}training",
    ]
    for p in bond_pats:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            ev = extract_sentence_evidence(text, m.span())
            return Check(
                id="E2",
                title="Employment bond or training cost recovery",
                severity="important",
                why="You may have to pay money if you leave early.",
                message=f"Employment bond or cost recovery language detected: '{ev.text[:120]}'.",
                status="active",
                evidence=ev,
            )
    return None


def check_e3_benefits_not_mentioned(text: str) -> Optional[Check]:
    benefit_pats = [
        r"provident\s+fund", r"\bepf\b", r"\bgratuity\b",
        r"\besi\b", r"earned\s+leave", r"casual\s+leave", r"paid\s+leave",
    ]
    has_benefits = any(re.search(p, text, re.IGNORECASE) for p in benefit_pats)
    if not has_benefits:
        return Check(
            id="E3",
            title="Statutory employee benefits unstated",
            severity="check",
            why="Standard benefits are not mentioned. Ask where they are written down.",
            message="No mention of standard benefits (PF, Gratuity, ESI, or paid leave) found.",
            status="active",
            evidence=None,
        )
    return None


def check_l1_lender_can_change(text: str) -> Optional[Check]:
    pats = [
        r"\bsole\s+discretion\b",
        r"at\s+any\s+time[^\.\n;]{0,30}(?:change|revise|vary)",
        r"may\s+(?:revise|vary|modify)\s+(?:the\s+)?(?:rate|interest|terms)",
    ]
    for p in pats:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            ev = extract_sentence_evidence(text, m.span())
            return Check(
                id="L1",
                title="Unilateral term modification",
                severity="check",
                why="The lender may be able to change the cost of the loan on its own.",
                message=f"Lender unilateral modification clause found: '{ev.text[:120]}'.",
                status="active",
                evidence=ev,
            )
    return None


def check_l2_acceleration(text: str) -> Optional[Check]:
    pats = [
        r"immediately\s+due\s+and\s+payable",
        r"\baccelerate\b",
        r"on\s+demand",
        r"recall\s+the\s+loan",
    ]
    for p in pats:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            ev = extract_sentence_evidence(text, m.span())
            return Check(
                id="L2",
                title="Immediate loan recall or acceleration",
                severity="check",
                why="The lender may be able to demand the whole loan back at once.",
                message=f"Loan acceleration or immediate recall clause found: '{ev.text[:120]}'.",
                status="active",
                evidence=ev,
            )
    return None


# -----------------------------------------------------------------------------
# Main Runner
# -----------------------------------------------------------------------------

def run_checks(
    doc_type: str,
    fields: Dict[str, Optional[Field]],
    dates: Dict[str, Optional[Field]],
    tenure_months: Optional[float],
    text: str,
) -> List[Check]:
    """Execute all active checks applicable to doc_type and append planned checks."""
    catalog = load_catalog()
    checks: List[Check] = []

    # General checks (apply to all)
    c_g1 = check_g1_governing_law(fields.get("Governing Law"), text)
    if c_g1: checks.append(c_g1)

    c_g2 = check_g2_auto_renewal(fields.get("Renewal Term"), fields.get("Notice Period To Terminate Renewal"), text)
    if c_g2: checks.append(c_g2)

    c_g3 = check_g3_early_termination(fields.get("Termination For Convenience"))
    if c_g3: checks.append(c_g3)

    c_g4 = check_g4_date_sanity(dates)
    if c_g4: checks.append(c_g4)

    c_g5 = check_g5_liability(fields.get("Cap On Liability"), fields.get("Uncapped Liability"))
    if c_g5: checks.append(c_g5)

    c_g6 = check_g6_signature_block(text)
    if c_g6: checks.append(c_g6)

    c_g7 = check_g7_assignment(fields.get("Anti-Assignment"))
    if c_g7: checks.append(c_g7)

    # Document type specific checks
    if doc_type == "rental_lease":
        c_r1 = check_r1_rental_registration(tenure_months, text)
        if c_r1: checks.append(c_r1)
        c_r2 = check_r2_rental_stamp(text)
        if c_r2: checks.append(c_r2)

    elif doc_type == "employment":
        c_e1 = check_e1_post_employment_noncompete(fields.get("Non-Compete"), text)
        if c_e1: checks.append(c_e1)
        c_e2 = check_e2_bond(text)
        if c_e2: checks.append(c_e2)
        c_e3 = check_e3_benefits_not_mentioned(text)
        if c_e3: checks.append(c_e3)

    elif doc_type == "loan":
        c_l1 = check_l1_lender_can_change(text)
        if c_l1: checks.append(c_l1)
        c_l2 = check_l2_acceleration(text)
        if c_l2: checks.append(c_l2)

    # Append planned checks as not_checked
    for planned in catalog.get("planned_checks", []):
        applies = planned.get("applies_to", "all")
        if applies in ("all", doc_type):
            checks.append(Check(
                id=planned["id"],
                title=planned["title"],
                severity=planned.get("severity", "info"),
                why=planned.get("why", ""),
                message="Not checked yet (requires Indian fine-tuned model checkpoint).",
                status="not_checked",
                evidence=None,
            ))

    return checks

