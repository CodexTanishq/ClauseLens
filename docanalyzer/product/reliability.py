"""Category-level reliability scoring, human-readable confidence labels, and threshold mappings."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from docanalyzer.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_RELIABILITY_RATINGS: Dict[str, str] = {
    # Based on benchmark CUAD run:
    # High: EM >= 80
    "Document Name": "medium",      # EM ~78.4 -> medium
    "Parties": "high",              # EM ~81.4 -> high
    "Agreement Date": "high",       # EM ~89.2 -> high
    "Expiration Date": "high",      # EM ~91.0 -> high
    "Governing Law": "high",        # EM ~89.2 -> high
    "Anti-Assignment": "high",      # EM ~84.7 -> high
    "Renewal Term": "high",         # EM ~88.5 -> high
    "Termination For Convenience": "high", # EM ~82.8 -> high
    # Medium: 60 <= EM < 80
    "Effective Date": "medium",     # EM ~65.7 -> medium
    "Cap On Liability": "medium",   # EM ~79.5 -> medium
    "Notice Period To Terminate Renewal": "high", # EM ~81.2 -> high
    "Non-Compete": "medium",        # EM ~60.9 -> medium
    "No-Solicit Of Employees": "medium", # EM ~70.0 -> medium
    # Low: EM < 60
    "Uncapped Liability": "low",    # EM ~30.8 -> low
    "Ip Ownership Assignment": "low", # EM ~43.5 -> low
    "Liquidated Damages": "low",    # EM ~57.1 -> low
    "No-Solicit Of Customers": "low", # EM ~57.1 -> low
}

HUMAN_LABELS: Dict[str, str] = {
    "high": "usually right",
    "medium": "check carefully",
    "low": "often wrong",
}


def get_human_reliability_label(reliability: str) -> str:
    """Return friendly non-technical description of reliability."""
    return HUMAN_LABELS.get(reliability.lower(), "check carefully")


def load_reliability_json(path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load per-category reliability mappings from JSON, with automatic candidate discovery."""
    candidates = []
    if path:
        candidates.append(Path(path))

    candidates.extend([
        PROJECT_ROOT / "models" / "cuad_us_only" / "reliability.json",
        PROJECT_ROOT / "outputs" / "reliability.json",
    ])

    for c in candidates:
        if c.exists():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to parse reliability json from '%s': %s", c, e)

    # Fallback to default benchmark ratings
    return {
        cat: {
            "reliability": rating,
            "human_label": get_human_reliability_label(rating),
            "exact_match": 85.0 if rating == "high" else (70.0 if rating == "medium" else 45.0),
        }
        for cat, rating in DEFAULT_RELIABILITY_RATINGS.items()
    }


def write_reliability_json(
    per_category_csv: Path | str,
    out_path: Optional[Path | str] = None,
    high_threshold: float = 80.0,
    medium_threshold: float = 60.0,
) -> Path:
    """Generate reliability.json mapping from evaluated per_category_metrics.csv."""
    csv_file = Path(per_category_csv)
    target_out = Path(out_path) if out_path else csv_file.parent / "reliability.json"

    mapping: Dict[str, Any] = {}
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"]
            em = float(row["exact_match"])
            f1 = float(row["f1"])

            if em >= high_threshold:
                level = "high"
            elif em >= medium_threshold:
                level = "medium"
            else:
                level = "low"

            mapping[cat] = {
                "reliability": level,
                "human_label": get_human_reliability_label(level),
                "exact_match": em,
                "f1": f1,
            }

    with open(target_out, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    logger.info("Saved reliability mapping to '%s'", target_out)
    return target_out

