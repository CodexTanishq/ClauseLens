"""Date parsing (day-first Indian conventions), duration extraction, and tenure calculation."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

from dateutil import parser as dt_parser

logger = logging.getLogger(__name__)

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
}

MONTH_NAMES = {
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "june",
    "july", "august", "september", "october", "november", "december",
}


def parse_date(text: str) -> Optional[Dict[str, Any]]:
    """Parse text into ISO date, defaulting to day-first (Indian format).

    Flags ambiguous dates when day and month are both <= 12 and no text month name is present.
    """
    if not text or not text.strip():
        return None

    cleaned = text.strip()
    has_alpha_month = any(m in cleaned.lower() for m in MONTH_NAMES)

    try:
        # Default dayfirst=True for Indian legal documents
        parsed_dt = dt_parser.parse(cleaned, dayfirst=True, fuzzy=True)

        is_ambiguous = False
        if not has_alpha_month:
            # Check if text contains digits like 05/06/2024 or 05-06-2024
            m = re.search(r"\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})\b", cleaned)
            if m:
                d1, d2 = int(m.group(1)), int(m.group(2))
                if 1 <= d1 <= 12 and 1 <= d2 <= 12:
                    is_ambiguous = True

        return {
            "iso": parsed_dt.date().isoformat(),
            "year": parsed_dt.year,
            "month": parsed_dt.month,
            "day": parsed_dt.day,
            "ambiguous": is_ambiguous,
            "has_month_name": has_alpha_month,
        }
    except Exception:
        return None


def parse_duration_words(text: str) -> Optional[int]:
    """Parse English number words up to 100 into integer."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    val = 0
    current = 0
    for tok in tokens:
        if tok in NUMBER_WORDS:
            num = NUMBER_WORDS[tok]
            if num == 100:
                current = (current if current else 1) * 100
            else:
                current += num
        elif tok == "and":
            continue
        else:
            if current:
                val += current
                current = 0
    val += current
    return val if val > 0 else None


def parse_duration(text: str) -> Optional[float]:
    """Extract duration in months from text (e.g., 'eleven (11) months', '1 year', '36 months')."""
    if not text or not text.strip():
        return None

    cleaned = text.lower()

    # Pattern 1: Digit months (e.g., '11 months', '11 (eleven) months')
    m_month_num = re.search(r"(\d+)\s*(?:\([a-zA-Z\s]+\))?\s*months?", cleaned)
    if m_month_num:
        return float(m_month_num.group(1))

    # Pattern 2: Word months (e.g., 'eleven (11) months', 'twenty-four months')
    m_month_words = re.search(r"([a-zA-Z\s\-]+)\s*(?:\(\d+\))?\s*months?", cleaned)
    if m_month_words:
        w_val = parse_duration_words(m_month_words.group(1))
        if w_val:
            return float(w_val)

    # Pattern 3: Digit years (e.g., '3 years', '1 year')
    m_year_num = re.search(r"(\d+)\s*(?:\([a-zA-Z\s]+\))?\s*years?", cleaned)
    if m_year_num:
        return float(int(m_year_num.group(1)) * 12)

    # Pattern 4: Word years (e.g., 'three years', 'one (1) year')
    m_year_words = re.search(r"([a-zA-Z\s\-]+)\s*(?:\(\d+\))?\s*years?", cleaned)
    if m_year_words:
        w_val = parse_duration_words(m_year_words.group(1))
        if w_val:
            return float(w_val * 12)

    # Pattern 5: Days
    m_days = re.search(r"(\d+)\s*days?", cleaned)
    if m_days:
        return round(float(m_days.group(1)) / 30.0, 1)

    return None


def compute_tenure(
    effective_text: Optional[str],
    expiration_text: Optional[str],
    duration_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate tenure months/days from parsed dates or duration clause text."""
    eff_parsed = parse_date(effective_text) if effective_text else None
    exp_parsed = parse_date(expiration_text) if expiration_text else None

    # Priority 1: Compute from effective and expiration dates
    if eff_parsed and exp_parsed:
        try:
            d_eff = dt_parser.parse(eff_parsed["iso"]).date()
            d_exp = dt_parser.parse(exp_parsed["iso"]).date()
            delta = d_exp - d_eff
            days = delta.days
            if days >= 0:
                months = round(days / 30.4375, 1)
                note = f"Calculated from {d_eff} to {d_exp} ({days} days)"
                return {
                    "months": months,
                    "days": days,
                    "source": "from dates",
                    "note": note,
                }
            else:
                return {
                    "months": None,
                    "days": days,
                    "source": "from dates",
                    "note": f"Date sanity warning: expiration ({d_exp}) precedes effective date ({d_eff}).",
                }
        except Exception as e:
            logger.warning("Error computing date delta: %s", e)

    # Priority 2: Extract duration from duration text or renewal text
    if duration_text:
        months = parse_duration(duration_text)
        if months is not None:
            return {
                "months": months,
                "days": int(months * 30.4375),
                "source": "from duration text",
                "note": f"Extracted from duration clause: '{duration_text.strip()[:60]}'",
            }

    return {
        "months": None,
        "days": None,
        "source": "unknown",
        "note": "Neither complete dates nor explicit duration clause available.",
    }

