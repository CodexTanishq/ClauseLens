# ClauseLens

ClauseLens is a production-oriented contract and agreement analyzer. It performs extractive question answering using fine-tuned RoBERTa models to locate key clauses, compute dates and duration, flag potential risks, and cite exact textual evidence from agreements.

Direct Link to deployed app : https://clause-lens.streamlit.app/

---

## Architecture Overview

```
clauselens/
  docanalyzer/
    config.py               # Paths, settings, model registry
    document_cleaner.py     # PDF/DOCX/TXT text extraction and cleaning pipeline
    data_loading.py         # CUAD QA dataset loading, row deduplication, contract splits
    data_analysis.py        # Token percentiles and answer length profiling
    preprocessing.py        # Sliding-window chunking, offset alignment, negative filtering
    model.py                # Tokenizer/model loaders, training arguments factory
    train.py                # Fine-tuning CLI for Stage 1 (CUAD) and Stage 2 (Indian Docs)
    postprocessing.py       # Logits-to-text span extraction using token offset mappings
    evaluate.py             # SQuAD EM/F1 evaluation and error categorization CLI
    inference.py            # Unified single-document analysis pipeline
    product/
      schema.py             # Report, Field, Check, and Evidence dataclasses
      doc_type.py           # Rule-based agreement classifier (rental, loan, employment)
      dates.py              # Day-first date parsing, duration words, and tenure logic
      fields.py             # Available vs. planned field specifications
      reliability.py        # Category-level confidence ratings (high/medium/low)
      checks_catalog.yaml   # Registry of contract risk rules and explanations
      checks.py             # Active rule implementations
      summary.py            # Factual template-based summary generation
      calibrate.py          # Presence threshold calibration CLI
  models/
    cuad_us_only/           # CUAD model statistics and reliability metadata
    cuad_indian/            # Extension point for Stage 2 adaptation
    rag/                    # Extension point for future RAG engine
  app/
    streamlit_app.py        # Interactive document review desk
    DESIGN.md               # Visual design specifications and principles
    assets/                 # Stylesheet and scroll-to-highlight viewer script
    samples/                # Fictional sample agreements for testing
  tests/                    # Automated unit and integration test suite
```

---

## Setup and Installation

### 1. Requirements

Install runtime dependencies from the project root:

```powershell
pip install -r requirements.txt
```

For development, training, and evaluation:

```powershell
pip install -r requirements-train.txt
```

### 2. Model Weights Resolution

Model weights are **not** stored in Git. By default, ClauseLens searches:
1. `CUAD_MODEL_ID` environment variable (if specified, local folder or Hugging Face repository ID)
2. `../cuad_roberta_v2_results/checkpoint-7401`
3. `../cuad_roberta_v2_final`
4. `models/cuad_us_only/`

In cloud deployments, set:
```powershell
$env:CUAD_MODEL_ID = "<your-hf-username>/cuad-roberta-base"
```

---

## Usage Guide (PowerShell Commands)

### 1. Document Extraction & Cleaning

Clean and structure raw agreements from an input folder into frozen JSON and text:

```powershell
python -m docanalyzer.document_cleaner --input ./raw_contracts --output ./clean_contracts
```

### 2. Dataset Statistical Profiling

Profile contract token lengths, answer span percentiles, and generate recommended `max_answer_length`:

```powershell
python -m docanalyzer.data_analysis --output-dir models/cuad_us_only
```

### 3. Model Training CLI

Train on CUAD (or local contract JSON):

```powershell
# Quick smoke test (20 steps on 20 rows):
python -m docanalyzer.train --run-name smoke_test --smoke

# Full training run:
python -m docanalyzer.train --run-name cuad_run_v1 --epochs 3
```

### 4. Evaluation CLI

Evaluate model checkpoints on validation or test splits:

```powershell
# Evaluate all checkpoints in a run directory on validation to find the best epoch:
python -m docanalyzer.evaluate --run-dir outputs/cuad_run_v1 --all-checkpoints

# Evaluate a specific checkpoint on validation:
python -m docanalyzer.evaluate --checkpoint ../cuad_roberta_v2_results/checkpoint-7401 --split validation

# Evaluate on test split (explicit confirmation required to prevent leakage):
python -m docanalyzer.evaluate --checkpoint ../cuad_roberta_v2_results/checkpoint-7401 --split test --confirm-test
```

### 5. Running the Review Desk Application

Launch the interactive review desk interface:

```powershell
streamlit run app/streamlit_app.py
```

---

## Calibration Note & Assumptions

In `docanalyzer/product/calibrate.py`:
- The CUAD dataset contains answered rows for applicable categories.
- For calibration, positive pairs are taken from the validation split `(contract, category)` records.
- Negative (unanswerable) pairs are synthesized by pairing contracts with categories that do not occur in that contract's annotations.
- The resulting scores establish optimal F1 thresholds stored in `calibration.json`.

---

## Testing

Run unit and integration tests using pytest:

```powershell
pytest tests -v
```

