"""ClauseLens: Production Contract Review Desk Streamlit Application."""

from __future__ import annotations

import html
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

from docanalyzer.config import (
    APP_NAME,
    APP_VERSION,
    MODEL_SOURCES,
    PROJECT_ROOT,
)
from docanalyzer.document_cleaner import clean_text, extract_text
from docanalyzer.inference import QAEngine, analyze_document, load_engine
from docanalyzer.product.reliability import get_human_reliability_label
from docanalyzer.product.schema import Report

# Set page configuration
st.set_page_config(
    page_title=f"{APP_NAME} - Contract Review Desk",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit default branding and decoration
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_engine(option_name: str) -> QAEngine:
    """Cache singleton QAEngine in memory for fast interactive queries."""
    return load_engine(option_name)


def load_categories() -> List[str]:
    """Load dataset question categories for multiselect."""
    cat_file = PROJECT_ROOT / "models" / "cuad_us_only" / "categories.json"
    if cat_file.exists():
        try:
            with open(cat_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return [
        "Parties",
        "Agreement Date",
        "Effective Date",
        "Expiration Date",
        "Governing Law",
        "Termination For Convenience",
        "Renewal Term",
        "Notice Period To Terminate Renewal",
        "Anti-Assignment",
        "Cap On Liability",
        "Non-Compete",
    ]


def build_interactive_html(report: Report, full_text: str) -> str:
    """Build self-contained HTML/CSS/JS review desk component."""
    css_path = Path(__file__).parent / "assets" / "style.css"
    js_path = Path(__file__).parent / "assets" / "viewer.js"

    css_content = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    js_content = js_path.read_text(encoding="utf-8") if js_path.exists() else ""

    # Build spans to highlight in contract text
    highlights: List[Dict[str, Any]] = []

    # Highlight dates and key fields
    for field_name, f_obj in {**report.dates, **report.key_terms}.items():
        if f_obj and f_obj.evidence and f_obj.raw_text.strip():
            highlights.append({
                "id": f"field_{field_name.replace(' ', '_')}",
                "start": f_obj.evidence.start_char,
                "end": f_obj.evidence.end_char,
                "title": field_name,
                "cls": "hl-date" if "Date" in field_name or "Term" in field_name else "hl-term",
            })

    # Highlight check evidence
    for chk in report.checks:
        if chk.status == "active" and chk.evidence and chk.evidence.text.strip():
            highlights.append({
                "id": f"chk_{chk.id}",
                "start": chk.evidence.start_char,
                "end": chk.evidence.end_char,
                "title": chk.title,
                "cls": "hl-flag",
            })

    # Sort non-overlapping highlights
    highlights = [h for h in highlights if h["start"] < h["end"] and h["end"] <= len(full_text)]
    highlights.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

    # Inject mark tags
    html_pieces = []
    last_idx = 0
    for h in highlights:
        if h["start"] >= last_idx:
            # Append preceding clean text
            html_pieces.append(html.escape(full_text[last_idx:h["start"]]))
            # Wrap marked text
            span_text = html.escape(full_text[h["start"]:h["end"]])
            mark_tag = (
                f'<mark class="cl-highlight {h["cls"]}" data-finding-id="{h["id"]}" '
                f'title="{html.escape(h["title"])}">{span_text}</mark>'
            )
            html_pieces.append(mark_tag)
            last_idx = h["end"]

    html_pieces.append(html.escape(full_text[last_idx:]))
    annotated_contract_html = "".join(html_pieces)

    # Build findings cards
    findings_html = []

    # Active checks
    active_checks = [c for c in report.checks if c.status == "active"]
    if active_checks:
        findings_html.append('<div class="findings-group-title">Contract Checks</div>')
        for c in active_checks:
            sev_class = f"severity-{c.severity}"
            sev_label = "Important to read" if c.severity == "important" else ("Worth checking" if c.severity == "check" else "Information")
            ev_html = f'<div class="finding-evidence">"{html.escape(c.evidence.text[:140])}..."</div>' if c.evidence else ""

            card = f"""
            <div class="finding-card" data-finding-id="chk_{c.id}">
              <div class="finding-header">
                <span class="finding-title">{html.escape(c.title)}</span>
                <span class="severity-pill {sev_class}">{sev_label}</span>
              </div>
              <div class="finding-why">{html.escape(c.why)}</div>
              {ev_html}
              <div class="finding-meta">{html.escape(c.message)}</div>
            </div>
            """
            findings_html.append(card)

    # Key terms and dates
    findings_html.append('<div class="findings-group-title">Extracted Clauses</div>')
    for name, f_obj in {**report.dates, **report.key_terms}.items():
        if f_obj and f_obj.raw_text.strip():
            rel_label = get_human_reliability_label(f_obj.reliability)
            card = f"""
            <div class="finding-card" data-finding-id="field_{name.replace(' ', '_')}">
              <div class="finding-header">
                <span class="finding-title">{html.escape(name)}</span>
                <span class="tag-item">{rel_label}</span>
              </div>
              <div class="finding-evidence">"{html.escape(f_obj.raw_text[:140])}"</div>
            </div>
            """
            findings_html.append(card)

    # Planned checks (collapsed)
    planned_checks = [c for c in report.checks if c.status == "not_checked"]
    if planned_checks:
        planned_items = "".join([f'<div class="planned-item">{html.escape(c.title)}</div>' for c in planned_checks])
        findings_html.append(f"""
        <details class="planned-group">
          <summary>Not checked yet ({len(planned_checks)} checks)</summary>
          {planned_items}
        </details>
        """)

    findings_col_html = "".join(findings_html)

    # Summary strip details
    parties_str = ", ".join([p["name"] for p in report.parties]) if report.parties else "Unspecified"
    tenure_str = f"{report.tenure.months:.0f} months" if report.tenure.months else "Unknown"

    component_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <style>{css_content}</style>
    </head>
    <body>
      <div class="desk-container">
        <div class="summary-strip">
          <div class="meta-group">
            <span class="tag-item">Document: <strong>{html.escape(report.document.title or report.document.type)}</strong></span>
            <span class="tag-item">Parties: <strong>{html.escape(parties_str[:60])}</strong></span>
            <span class="tag-item">Tenure: <strong>{html.escape(tenure_str)}</strong></span>
          </div>
          <div>
            <span class="tag-item">Model: {html.escape(report.meta.option_name)}</span>
          </div>
        </div>
        <div class="summary-text-box">
          {html.escape(report.summary)}
        </div>
        <div class="workspace-split">
          <div class="contract-pane">
            <div class="contract-text">{annotated_contract_html}</div>
          </div>
          <div class="findings-pane">
            {findings_col_html}
          </div>
        </div>
      </div>
      <script>{js_content}</script>
    </body>
    </html>
    """
    return component_html


def main() -> None:
    # Sidebar
    st.sidebar.title(f"{APP_NAME}")
    st.sidebar.caption(f"Version {APP_VERSION}")

    # Model option selector
    st.sidebar.subheader("Model Selection")
    options = list(MODEL_SOURCES.keys())
    selected_option = st.sidebar.selectbox("Active Model", options=options, index=0)

    if selected_option == "CUAD (US contracts)":
        st.sidebar.caption("Trained on US contracts. Accuracy on Indian contracts has not been measured yet.")
    elif not MODEL_SOURCES[selected_option]:
        st.sidebar.warning(f"{selected_option} is not available yet. Please select the CUAD model.")

    # Multiselect question categories
    all_categories = load_categories()
    default_cats = [c for c in ["Parties", "Agreement Date", "Effective Date", "Expiration Date", "Governing Law"] if c in all_categories]
    selected_categories = st.sidebar.multiselect("Question Categories", options=all_categories, default=default_cats)

    st.sidebar.markdown("---")
    st.sidebar.info("This tool helps you read a contract. It is not legal advice.")

    # Main view
    st.title("Contract Review Desk")
    st.write("Extract key clauses, calculate contract dates and tenure, and review potential legal risks.")

    # File upload or sample loading
    col_upload, col_sample = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload agreement (.pdf, .docx, or .txt)",
            type=["pdf", "docx", "txt", "doc"],
            help="Upload an agreement up to 20MB. Scanned PDFs are not currently supported.",
        )

    with col_sample:
        st.write("Or try a sample:")
        sample_choice = st.selectbox(
            "Select sample",
            ["None", "11-Month Rental Agreement", "Employment Offer Letter"],
            index=0,
            label_visibility="collapsed",
        )

    # Resolve document text
    contract_text = ""
    source_filename = ""

    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".doc"):
            st.error("Legacy Word (.doc) files are not supported. Please convert your file to .docx before uploading.")
            return

        try:
            bytes_data = uploaded_file.read()
            if uploaded_file.name.lower().endswith(".txt"):
                raw_text = bytes_data.decode("utf-8", errors="replace")
            elif uploaded_file.name.lower().endswith(".pdf"):
                import pypdf
                pdf_reader = pypdf.PdfReader(io.BytesIO(bytes_data))
                raw_text = "\f".join([page.extract_text() or "" for page in pdf_reader.pages])
            elif uploaded_file.name.lower().endswith(".docx"):
                import docx
                doc = docx.Document(io.BytesIO(bytes_data))
                raw_text = "\n\n".join([p.text for p in doc.paragraphs if p.text])
            else:
                raw_text = ""

            cleaned, _ = clean_text(raw_text)
            contract_text = cleaned
            source_filename = uploaded_file.name

        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

    elif sample_choice != "None":
        sample_map = {
            "11-Month Rental Agreement": PROJECT_ROOT / "app" / "samples" / "rental_sample.txt",
            "Employment Offer Letter": PROJECT_ROOT / "app" / "samples" / "employment_sample.txt",
        }
        sample_path = sample_map.get(sample_choice)
        if sample_path and sample_path.exists():
            contract_text = sample_path.read_text(encoding="utf-8")
            source_filename = sample_path.name

    if not contract_text.strip():
        st.info("Upload a contract file above or select a sample agreement to begin.")
        return

    # Document preview snippet
    with st.expander(f"Document Preview: {source_filename} ({len(contract_text)} characters)", expanded=False):
        st.text(contract_text[:1500] + ("..." if len(contract_text) > 1500 else ""))

    # Analyze Action
    if st.button("Analyze document", type="primary"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_progress(p: float, msg: str) -> None:
            progress_bar.progress(p)
            status_text.text(msg)

        try:
            with st.spinner("Analyzing contract clauses and rules..."):
                report = analyze_document(
                    contract_text,
                    option=selected_option,
                    progress_callback=on_progress,
                )

            st.session_state["latest_report"] = report
            status_text.empty()
            progress_bar.empty()

        except Exception as e:
            status_text.empty()
            progress_bar.empty()
            st.error(f"Analysis could not be completed: {e}")
            return

    # Render Report if available in session state
    if "latest_report" in st.session_state:
        report: Report = st.session_state["latest_report"]

        # Render self-contained review desk component
        desk_html = build_interactive_html(report, contract_text)
        components.html(desk_html, height=880, scrolling=False)

        # Download actions
        st.markdown("---")
        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            st.download_button(
                label="Download Report (JSON)",
                data=json.dumps(report.to_dict(), indent=2),
                file_name=f"{Path(source_filename).stem}_report.json",
                mime="application/json",
            )
        with col_d2:
            # Flatten findings into CSV
            csv_buffer = io.StringIO()
            csv_buffer.write("Category,Value,Evidence,Reliability\n")
            for name, f_obj in {**report.dates, **report.key_terms}.items():
                if f_obj and f_obj.raw_text.strip():
                    clean_val = f_obj.raw_text.replace('"', '""')
                    clean_ev = f_obj.evidence.text.replace('"', '""')
                    csv_buffer.write(f'"{name}","{clean_val}","{clean_ev}","{f_obj.reliability}"\n')

            st.download_button(
                label="Download Findings (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"{Path(source_filename).stem}_findings.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

