"""Rule-based contract type classification with weighted keyword scoring."""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

KEYWORDS: Dict[str, Dict[str, float]] = {
    "rental_lease": {
        r"\blease\b": 3.0,
        r"\blicen[cs]e\b": 2.5,
        r"\blandlord\b": 3.0,
        r"\btenant\b": 3.0,
        r"\blicensor\b": 3.0,
        r"\blicensee\b": 3.0,
        r"\bpremises\b": 2.0,
        r"\brent\b": 2.5,
        r"\bsecurity\s+deposit\b": 3.0,
        r"\bleave\s+and\s+licen[cs]e\b": 4.0,
    },
    "loan": {
        r"\bborrower\b": 3.0,
        r"\blender\b": 3.0,
        r"\bloan\b": 3.0,
        r"\binterest\b": 2.0,
        r"\brepayment\b": 2.5,
        r"\bemi\b": 3.0,
        r"\bprincipal\b": 2.0,
        r"\bfacility\b": 2.0,
        r"\bprepay\b": 2.5,
        r"\bcredit\s+facility\b": 3.5,
    },
    "employment": {
        r"\bemployer\b": 3.0,
        r"\bemployee\b": 3.0,
        r"\bsalary\b": 3.0,
        r"\bprobation\b": 3.0,
        r"\bnotice\s+period\b": 2.5,
        r"\bappointment\b": 2.5,
        r"\bdesignation\b": 2.0,
        r"\bctc\b": 3.0,
        r"\bemployment\s+agreement\b": 4.0,
        r"\boffer\s+of\s+employment\b": 4.0,
    },
}


def classify_document_type(
    text: str,
    title: str = "",
    min_confidence_score: float = 4.0,
) -> Tuple[str, float]:
    """Classify contract into rental_lease, loan, employment, or other using weighted keyword matches.

    Analyzes title and the first 3000 characters of the document text.
    Returns:
        Tuple of (doc_type, confidence) where confidence is in [0.0, 1.0].
    """
    excerpt = (title + " " + text[:3000]).lower()
    scores: Dict[str, float] = {"rental_lease": 0.0, "loan": 0.0, "employment": 0.0}

    for doc_type, patterns in KEYWORDS.items():
        total = 0.0
        for pattern, weight in patterns.items():
            matches = len(re.findall(pattern, excerpt, re.IGNORECASE))
            if matches > 0:
                total += weight * min(matches, 3)
        scores[doc_type] = total

    best_type, best_score = max(scores.items(), key=lambda x: x[1])

    if best_score < min_confidence_score:
        return "other", 0.0

    # Normalize confidence into [0.0, 1.0]
    confidence = min(1.0, round(best_score / 25.0, 2))
    return best_type, max(confidence, 0.40)


def evaluate_classifier(folder: Path, labels_csv: Path) -> None:
    """Evaluate classifier against ground truth CSV (columns: filename, label)."""
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ground_truth = {row["filename"]: row["label"].strip() for row in reader}

    all_labels = ["rental_lease", "loan", "employment", "other"]
    matrix = {true_lbl: {pred_lbl: 0 for pred_lbl in all_labels} for true_lbl in all_labels}
    correct = 0
    total = 0

    for fname, true_label in ground_truth.items():
        doc_path = folder / fname
        if not doc_path.exists():
            continue
        text = doc_path.read_text(encoding="utf-8", errors="replace")
        first_line = text.split("\n")[0] if "\n" in text else ""
        pred_label, conf = classify_document_type(text, title=first_line)

        matrix[true_label][pred_label] += 1
        total += 1
        if pred_label == true_label:
            correct += 1

    accuracy = (correct / total * 100) if total > 0 else 0.0
    print(f"\nClassification Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print("\nConfusion Matrix:")
    header = "True \\ Pred".ljust(15) + "".join(lbl.ljust(15) for lbl in all_labels)
    print(header)
    for true_lbl in all_labels:
        row = true_lbl.ljust(15) + "".join(str(matrix[true_lbl][p]).ljust(15) for p in all_labels)
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Document type classification CLI.")
    parser.add_argument("--folder", type=Path, help="Folder containing documents")
    parser.add_argument("--labels", type=Path, help="CSV file with filename,label columns")

    args = parser.parse_args()
    if args.folder and args.labels:
        evaluate_classifier(args.folder, args.labels)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

