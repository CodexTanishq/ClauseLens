"""Document extraction, text cleaning, jurisdiction hint extraction, and QA row building."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------

def extract_text(file_path: Path | str) -> Tuple[str, int]:
    """Extract raw text from PDF, DOCX, or TXT file.

    Returns:
        Tuple of (raw_text, page_count).
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".doc":
        raise ValueError(
            f"Legacy Word format (.doc) for '{path.name}' is not supported. "
            "Please convert it to modern .docx format before processing."
        )

    if suffix == ".txt":
        raw = _extract_txt(path)
        return raw, 1

    if suffix == ".pdf":
        return _extract_pdf(path)

    if suffix == ".docx":
        return _extract_docx(path)

    raise ValueError(f"Unsupported file format '{suffix}'. Supported formats: .pdf, .docx, .txt")


def _extract_txt(path: Path) -> str:
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1", errors="replace")


def _extract_pdf(path: Path) -> Tuple[str, int]:
    try:
        import pypdf
    except ImportError as err:
        raise ImportError(
            "pypdf is required to extract text from PDF files. Install via pip install pypdf."
        ) from err

    reader = pypdf.PdfReader(str(path))
    n_pages = len(reader.pages)
    page_texts: List[str] = []

    total_chars = 0
    for page in reader.pages:
        txt = page.extract_text() or ""
        total_chars += len(txt.strip())
        page_texts.append(txt)

    chars_per_page = (total_chars / n_pages) if n_pages > 0 else 0
    if n_pages > 0 and chars_per_page < 200:
        logger.warning(
            "Warning: PDF '%s' has an average of only %.1f characters per page. "
            "It is likely a scanned document. Scanned documents / OCR are not yet supported.",
            path.name,
            chars_per_page,
        )

    return "\f".join(page_texts), n_pages


def _extract_docx(path: Path) -> Tuple[str, int]:
    try:
        import docx
    except ImportError as err:
        raise ImportError(
            "python-docx is required to extract text from DOCX files. Install via pip install python-docx."
        ) from err

    doc = docx.Document(str(path))
    lines: List[str] = []

    # Preserve document order across paragraphs and tables
    for elem in doc.element.body:
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag == "p":
            text = elem.text or ""
            if text.strip():
                lines.append(text)
        elif tag == "tbl":
            # Extract table cells
            table_rows: List[str] = []
            for row in elem.findall(".//{*}tr"):
                row_cells = [c.text.strip() for c in row.findall(".//{*}tc") if c.text]
                if row_cells:
                    table_rows.append(" | ".join(row_cells))
            if table_rows:
                lines.append("\n".join(table_rows))

    raw_text = "\n\n".join(lines)
    return raw_text, 1


# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------

@dataclass
class CleaningReport:
    """Statistics detailing modifications applied during document cleaning."""

    n_chars_raw: int = 0
    n_chars_clean: int = 0
    lines_removed_step1: int = 0
    hyphenated_words_joined_step2: int = 0
    page_headers_footers_removed_step3: int = 0
    lines_joined_step4: int = 0
    top_removed_header_footer_lines: List[str] = field(default_factory=list)


def clean_text(
    text: str,
    enable_step1_normalize: bool = True,
    enable_step2_dehyphen: bool = True,
    enable_step3_page_artifacts: bool = True,
    enable_step4_line_merge: bool = True,
    enable_step5_whitespace: bool = True,
) -> Tuple[str, CleaningReport]:
    """Clean extracted document text in 5 configurable steps.

    Returns:
        Tuple of (cleaned_text, cleaning_report).
    """
    report = CleaningReport(n_chars_raw=len(text))
    current = text

    # Step 1: Remove null/control characters, normalize endings, non-breaking spaces
    if enable_step1_normalize:
        # Keep \n, \t, \f. Remove other control chars.
        current = current.replace("\r\n", "\n").replace("\r", "\n")
        current = current.replace("\u00a0", " ")
        cleaned_chars: List[str] = []
        for ch in current:
            code = ord(ch)
            if code in (10, 9, 12) or (code >= 32 and code != 127):
                cleaned_chars.append(ch)
        current = "".join(cleaned_chars)

    # Step 2: Join hyphenated words broken across a line break (e.g. "distri-\nbutor")
    if enable_step2_dehyphen:
        def _join_hyphen(match: re.Match[str]) -> str:
            report.hyphenated_words_joined_step2 += 1
            return match.group(1) + match.group(2)

        current = re.sub(r"([A-Za-z]+)-\n([A-Za-z]+)", _join_hyphen, current)

    # Step 3: Remove page numbers and repeated header/footer lines (>= 3 pages)
    if enable_step3_page_artifacts:
        pages = current.split("\f")
        if len(pages) >= 3:
            header_footer_candidates: Counter[str] = Counter()
            page_blocks: List[List[str]] = []

            for page in pages:
                lines = [line.strip() for line in page.split("\n") if line.strip()]
                page_blocks.append(lines)
                page_candidates = set()
                # First 3 lines
                for l in lines[:3]:
                    page_candidates.add(l)
                # Last 3 lines
                for l in lines[-3:]:
                    page_candidates.add(l)

                for l in page_candidates:
                    if len(l) > 3 and not re.match(r"^(page|\d+)", l, re.IGNORECASE):
                        header_footer_candidates[l] += 1

            min_threshold = max(2, int(0.40 * len(pages)))
            repeated_lines = {
                line for line, cnt in header_footer_candidates.items() if cnt >= min_threshold
            }
            report.top_removed_header_footer_lines = [
                f"{line} (x{cnt})"
                for line, cnt in header_footer_candidates.most_common(5)
                if cnt >= min_threshold
            ]

            filtered_pages: List[str] = []
            for lines in page_blocks:
                new_page_lines: List[str] = []
                for idx, line in enumerate(lines):
                    s = line.strip()
                    # Check page number patterns
                    if re.match(r"^(?:page\s*\d+(?:\s*of\s*\d+)?|\d+\s*of\s*\d+|\d+)$", s, re.IGNORECASE):
                        report.page_headers_footers_removed_step3 += 1
                        continue
                    # Check first/last 3 lines for repeated header/footer
                    if (idx < 3 or idx >= len(lines) - 3) and s in repeated_lines:
                        report.page_headers_footers_removed_step3 += 1
                        continue
                    new_page_lines.append(line)
                filtered_pages.append("\n".join(new_page_lines))
            current = "\f".join(filtered_pages)

    # Step 4: Join lines broken inside a sentence while keeping structured legal clauses
    if enable_step4_line_merge:
        clause_break_patterns = [
            r"^\s*(?:\d+(?:\.\d+)*\.?)\s+",        # "1.", "1.1", "1.1.2"
            r"^\s*\([a-zA-Z0-9]+\)\s+",             # "(a)", "(i)", "(1)"
            r"^\s*(?:Article|Section|Schedule|Exhibit|Clause)\b",
            r"^\s*[•\-\*\u2022]\s+",               # Bullets
            r"^[A-Z0-9\s,\-\.]{5,}$",              # ALL CAPS headings
        ]
        clause_re = re.compile("|".join(f"(?:{p})" for p in clause_break_patterns), re.IGNORECASE)

        lines = current.split("\n")
        merged_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                merged_lines.append("")
                continue

            if not merged_lines or not merged_lines[-1].strip():
                merged_lines.append(stripped)
                continue

            prev = merged_lines[-1].strip()

            # Preserve break if current line starts a new clause or heading
            if clause_re.match(stripped):
                merged_lines.append(stripped)
                continue

            # Preserve break if previous line ended with punctuation, colon, semicolon
            if prev[-1] in (".", ":", ";", "?", "!"):
                merged_lines.append(stripped)
                continue

            # Otherwise, join broken sentence
            merged_lines[-1] = prev + " " + stripped
            report.lines_joined_step4 += 1

        current = "\n".join(merged_lines)

    # Step 5: Collapse repeated spaces, max 2 consecutive newlines, convert \f to paragraph break
    if enable_step5_whitespace:
        current = current.replace("\f", "\n\n")
        current = re.sub(r"[ \t]+", " ", current)
        current = re.sub(r"\n{3,}", "\n\n", current)
        current = current.strip()

    report.n_chars_clean = len(current)
    return current, report


# -----------------------------------------------------------------------------
# Jurisdiction & Hint Extraction
# -----------------------------------------------------------------------------

def extract_jurisdiction_hint(text: str) -> Dict[str, Any]:
    """Detect India vs US markers and extract Governing Law clause text."""
    if not text.strip():
        return {
            "india_hits": 0,
            "us_hits": 0,
            "label": "no text",
            "governing_law": "",
        }

    india_patterns = [
        r"\bIndia\b",
        r"\bIndian\b",
        r"\bRs\.?\s?\d",
        r"₹",
        r"\bINR\b",
        r"Indian\s+Contract\s+Act",
        r"Stamp\s+Act",
        r"Companies\s+Act",
        r"Private\s+Limited",
        r"Pvt\.?\s?Ltd",
        r"\b(?:Mumbai|Delhi|Bengaluru|Bangalore|Chennai|Kolkata|Hyderabad|Pune|Gurgaon|Gurugram|Noida|Ahmedabad)\b",
        r"\b(?:Maharashtra|Karnataka|Tamil\s+Nadu|Telangana|Gujarat|Uttar\s+Pradesh|West\s+Bengal|Haryana|Kerala)\b",
    ]

    us_patterns = [
        r"United\s+States",
        r"\bU\.S\.",
        r"\bUSD\b",
        r"US\$",
        r"\$\s?\d",
        r"\bDelaware\b",
        r"\bNew\s+York\b",
        r"\bCalifornia\b",
        r"\bTexas\b",
        r"\bFlorida\b",
        r"\bNew\s+Jersey\b",
        r"\bIllinois\b",
        r"Securities\s+Act",
        r"Exchange\s+Act",
    ]

    india_hits = sum(len(re.findall(p, text, re.IGNORECASE)) for p in india_patterns)
    us_hits = sum(len(re.findall(p, text, re.IGNORECASE)) for p in us_patterns)

    if india_hits >= 3 and india_hits > us_hits:
        label = "india-likely"
    elif us_hits >= 3 and us_hits > india_hits:
        label = "us-likely"
    else:
        label = "unclear"

    # Governing law regex
    gov_re = re.search(
        r"governed\s+(?:by|under)\s+(?:the\s+)?laws?\s+of\s+(?:the\s+)?([^\.\n;]{2,60})",
        text,
        re.IGNORECASE,
    )
    governing_law = gov_re.group(1).strip() if gov_re else ""

    return {
        "india_hits": india_hits,
        "us_hits": us_hits,
        "label": label,
        "governing_law": governing_law,
    }


# -----------------------------------------------------------------------------
# QA Alignment & Validation Utilities
# -----------------------------------------------------------------------------

def find_answer_start(context: str, answer_text: str) -> int:
    """Return character start offset of answer_text in context, or -1 if not found."""
    return context.find(answer_text)


def validate_answer(context: str, text: str, start: int) -> bool:
    """Ensure context[start:start+len(text)] == text strictly."""
    if start < 0 or start + len(text) > len(context):
        return False
    return context[start : start + len(text)] == text


def build_qa_rows(document: Dict[str, Any], qa_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build CUAD-format QA rows with validated character offsets."""
    context = document["context"]
    doc_id = document.get("id", hashlib.sha256(context.encode("utf-8")).hexdigest()[:16])

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(qa_list):
        question = item["question"]
        answers_in = item.get("answers", [])
        validated_texts: List[str] = []
        validated_starts: List[int] = []

        for ans in answers_in:
            ans_text = ans.get("text", "")
            ans_start = ans.get("answer_start", -1)
            if ans_start == -1 or not validate_answer(context, ans_text, ans_start):
                # Try finding exact occurrence
                ans_start = find_answer_start(context, ans_text)

            if ans_start != -1 and validate_answer(context, ans_text, ans_start):
                validated_texts.append(ans_text)
                validated_starts.append(ans_start)
            else:
                logger.warning(
                    "Answer span '%s' could not be validated in document '%s'",
                    ans_text[:30],
                    doc_id,
                )

        row = {
            "id": f"{doc_id}_{idx}",
            "question": question,
            "context": context,
            "answers": {
                "text": validated_texts,
                "answer_start": validated_starts,
            },
        }
        rows.append(row)

    return rows


def show_sample(file_path: Path | str) -> None:
    """Print a before/after excerpt of a middle page for quality inspection."""
    path = Path(file_path)
    raw_text, n_pages = extract_text(path)
    cleaned_text, report = clean_text(raw_text)

    print(f"=== Quality Sample for {path.name} ({n_pages} pages) ===")
    print(f"Raw chars: {report.n_chars_raw} -> Clean chars: {report.n_chars_clean}")
    print(f"Removed headers/footers: {report.page_headers_footers_removed_step3}")
    print(f"Joined lines: {report.lines_joined_step4}")
    print("\n--- Cleaned Excerpt (first 500 chars) ---")
    print(cleaned_text[:500])
    print("=========================================\n")


# -----------------------------------------------------------------------------
# CLI Batch Processing
# -----------------------------------------------------------------------------

def process_folder(
    input_folder: Path,
    output_folder: Path,
    category: Optional[str] = None,
    no_clean: bool = False,
    force: bool = False,
) -> None:
    """Process all supported documents in input_folder and save frozen clean output."""
    if output_folder.exists() and not force:
        raise FileExistsError(
            f"Output folder '{output_folder}' already exists. Pass --force to overwrite."
        )
    output_folder.mkdir(parents=True, exist_ok=True)

    supported_suffixes = {".pdf", ".docx", ".txt"}
    files = [f for f in input_folder.iterdir() if f.suffix.lower() in supported_suffixes]

    records: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    for f in files:
        try:
            raw_text, n_pages = extract_text(f)
            if no_clean:
                cleaned, rpt = raw_text, CleaningReport(n_chars_raw=len(raw_text), n_chars_clean=len(raw_text))
            else:
                cleaned, rpt = clean_text(raw_text)

            doc_id = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]
            first_line = next((l.strip() for l in cleaned.split("\n") if l.strip()), "")
            title = first_line[:120]
            jurisdiction = extract_jurisdiction_hint(cleaned)

            record = {
                "id": doc_id,
                "title": title,
                "source_file": f.name,
                "source_format": f.suffix.lower(),
                "category": category,
                "n_pages": n_pages,
                "n_chars_raw": rpt.n_chars_raw,
                "n_chars": rpt.n_chars_clean,
                "text_sha256": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
                "context": cleaned,
                "cleaning_report": asdict(rpt),
                "jurisdiction_hint": jurisdiction,
            }
            records.append(record)

            # Write clean text file
            txt_out = output_folder / f"{f.stem}.clean.txt"
            txt_out.write_text(cleaned, encoding="utf-8")

            report_rows.append({
                "filename": f.name,
                "pages": n_pages,
                "raw_chars": rpt.n_chars_raw,
                "clean_chars": rpt.n_chars_clean,
                "headers_removed": rpt.page_headers_footers_removed_step3,
                "lines_joined": rpt.lines_joined_step4,
                "jurisdiction": jurisdiction["label"],
            })
        except Exception as e:
            logger.error("Failed processing file '%s': %s", f.name, e)

    # Save documents.json
    docs_json = output_folder / "documents.json"
    with open(docs_json, "w", encoding="utf-8") as out:
        json.dump(records, out, indent=2, ensure_ascii=False)

    # Save clean_report.csv
    report_csv = output_folder / "clean_report.csv"
    if report_rows:
        with open(report_csv, "w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)

    print(f"Processed {len(records)} documents into '{output_folder}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Document extraction and cleaning CLI for ClauseLens.")
    parser.add_argument("--input", type=Path, default=None, help="Input directory containing documents")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for cleaned records")
    parser.add_argument("--category", type=str, default=None, help="Document category label")
    parser.add_argument("--no-clean", action="store_true", help="Skip text cleaning rules")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing output folder")
    parser.add_argument("--sample", type=Path, default=None, help="Inspect a sample file and exit")

    args = parser.parse_args()
    if args.sample:
        show_sample(args.sample)
        return

    if not args.input or not args.output:
        parser.error("Both --input and --output are required when --sample is not provided.")

    process_folder(
        input_folder=args.input,
        output_folder=args.output,
        category=args.category,
        no_clean=args.no_clean,
        force=args.force,
    )


if __name__ == "__main__":
    main()
