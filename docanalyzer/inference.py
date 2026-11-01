"""Inference engines, lazy model loader, single-document extractive QA, and Report builder."""

from __future__ import annotations

import abc
import argparse
import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from docanalyzer.config import (
    APP_VERSION,
    MODEL_SOURCES,
    PROJECT_ROOT,
    AppConfig,
    get_default_cuad_model_path,
)
from docanalyzer.document_cleaner import clean_text, extract_text
from docanalyzer.model import load_model, load_tokenizer
from docanalyzer.postprocessing import postprocess_qa_predictions
from docanalyzer.preprocessing import preprocess_eval
from docanalyzer.product.checks import run_checks
from docanalyzer.product.dates import compute_tenure, parse_date
from docanalyzer.product.doc_type import classify_document_type
from docanalyzer.product.fields import get_field_specs_for_type
from docanalyzer.product.reliability import load_reliability_json
from docanalyzer.product.schema import (
    DocumentInfo,
    Evidence,
    Field,
    MetaInfo,
    Report,
    TenureInfo,
)
from docanalyzer.product.summary import generate_summary

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    """Predicted answer span for a single question."""

    question: str
    text: str
    score: float
    start_char: int
    end_char: int
    found: bool = True


class QAEngine(abc.ABC):
    """Abstract Question Answering Engine interface."""

    @abc.abstractmethod
    def answer(
        self,
        document_text: str,
        questions: List[str],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[Answer]:
        """Answer questions against document_text."""
        raise NotImplementedError


class ExtractiveQAEngine(QAEngine):
    """Extractive QA Engine leveraging fine-tuned RoBERTa with sliding window tokenization."""

    def __init__(
        self,
        model_path_or_id: str | Path,
        device: str = "auto",
        max_answer_length: Optional[int] = None,
        n_best_size: int = 20,
        batch_size: int = 8,
        quantize: bool = False,
        min_score: Optional[float] = None,
    ) -> None:
        self.model_path_or_id = str(model_path_or_id)
        self.device_str = device
        self.n_best_size = n_best_size
        self.batch_size = batch_size
        self.quantize = quantize
        self.min_score = min_score

        # Determine max_answer_length
        if max_answer_length is not None:
            self.max_answer_length = max_answer_length
        else:
            self.max_answer_length = self._discover_max_answer_length()

        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None

    def _discover_max_answer_length(self) -> int:
        p = Path(self.model_path_or_id)
        candidates = [
            p / "train_stats.json",
            p.parent / "train_stats.json",
            PROJECT_ROOT / "models" / "cuad_us_only" / "train_stats.json",
        ]
        for c in candidates:
            if c.exists():
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        val = data.get("train_answer_stats", {}).get("recommended_max_answer_length")
                        if val:
                            return int(val)
                except Exception:
                    pass
        return 346  # Established default from CUAD training run

    def _ensure_loaded(self) -> None:
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.model_path_or_id)
        if self._model is None:
            if self.device_str == "auto":
                target_dev = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                target_dev = self.device_str

            model = load_model(self.model_path_or_id, device=target_dev)

            if self.quantize and target_dev == "cpu":
                logger.info("Applying dynamic int8 quantization on CPU...")
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

            model.eval()
            self._model = model

    def answer(
        self,
        document_text: str,
        questions: List[str],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[Answer]:
        self._ensure_loaded()
        cfg = AppConfig()

        results: List[Answer] = []
        total_questions = len(questions)

        target_device = next(self._model.parameters()).device

        for q_idx, question in enumerate(questions):
            if progress_callback:
                progress_callback(q_idx / total_questions, f"Extracting {question}...")

            single_example = {
                "id": [f"doc_{q_idx}"],
                "question": [question],
                "context": [document_text],
            }

            tokenized = preprocess_eval(
                single_example,
                self._tokenizer,
                max_length=cfg.tokens.max_length,
                stride=cfg.tokens.stride,
            )

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            all_start_logits = []
            all_end_logits = []

            for i in range(0, len(input_ids), self.batch_size):
                batch_features = [
                    {"input_ids": input_ids[j], "attention_mask": attention_mask[j]}
                    for j in range(i, min(i + self.batch_size, len(input_ids)))
                ]
                padded = self._tokenizer.pad(batch_features, padding=True, return_tensors="pt")
                b_input_ids = padded["input_ids"].to(target_device)
                b_attn = padded["attention_mask"].to(target_device)

                with torch.inference_mode():
                    outputs = self._model(input_ids=b_input_ids, attention_mask=b_attn)
                    s_logits = outputs.start_logits.detach().cpu().numpy()
                    e_logits = outputs.end_logits.detach().cpu().numpy()

                    for j in range(len(batch_features)):
                        seq_len = len(batch_features[j]["input_ids"])
                        all_start_logits.append(s_logits[j, :seq_len])
                        all_end_logits.append(e_logits[j, :seq_len])

            preds = postprocess_qa_predictions(
                examples=single_example,
                features=tokenized,
                raw_predictions=(all_start_logits, all_end_logits),
                n_best_size=self.n_best_size,
                max_answer_length=self.max_answer_length,
                return_details=True,
            )

            p = preds[0] if preds else {"prediction_text": "", "score": 0.0, "start_char": 0, "end_char": 0}
            score = float(p.get("score", 0.0))
            found = True if (self.min_score is None or score >= self.min_score) else False

            results.append(Answer(
                question=question,
                text=p["prediction_text"],
                score=score,
                start_char=int(p.get("start_char", 0)),
                end_char=int(p.get("end_char", 0)),
                found=found,
            ))

        if progress_callback:
            progress_callback(1.0, "Analysis complete.")

        return results


class RagEngine(QAEngine):
    """Retrieval-Augmented Generation (RAG) QA Engine. (Planned feature)."""

    def answer(
        self,
        document_text: str,
        questions: List[str],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[Answer]:
        """Planned RAG interface: embeds document chunks, retrieves top-k passages, and answers."""
        raise NotImplementedError(
            "RAG engine is planned for a future release and is not yet available."
        )


def load_engine(option_name: str) -> QAEngine:
    """Factory function loading QAEngine for the selected model source option."""
    if option_name not in MODEL_SOURCES:
        raise ValueError(f"Unknown model option '{option_name}'. Available: {list(MODEL_SOURCES.keys())}")

    model_target = MODEL_SOURCES[option_name]
    if not model_target:
        raise ValueError(
            f"The '{option_name}' engine is not available yet. "
            "Please select 'CUAD (US contracts)' to run analysis."
        )

    target_path = Path(model_target)
    if not target_path.exists() and not str(model_target).startswith(("hf-", "deepset", "distilbert")):
        raise FileNotFoundError(
            f"Model weights for '{option_name}' not found at '{model_target}'. "
            "Please ensure the checkpoint directory exists or set the CUAD_MODEL_ID environment variable."
        )

    return ExtractiveQAEngine(model_path_or_id=model_target)


# -----------------------------------------------------------------------------
# Document Analysis Pipeline
# -----------------------------------------------------------------------------

def analyze_document(
    path_or_text: Path | str,
    option: str = "CUAD (US contracts)",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Report:
    """Analyze a single contract document end-to-end and generate a structured Report."""
    # 1. Text extraction and cleaning
    if isinstance(path_or_text, Path) or (isinstance(path_or_text, str) and len(path_or_text) < 300 and Path(path_or_text).exists()):
        raw_text, _ = extract_text(path_or_text)
    else:
        raw_text = str(path_or_text)

    cleaned_text, _ = clean_text(raw_text)

    # 2. Document classification
    first_line = next((l.strip() for l in cleaned_text.split("\n") if l.strip()), "")
    doc_type, conf = classify_document_type(cleaned_text, title=first_line)

    doc_info = DocumentInfo(
        type=doc_type,
        type_confidence=conf,
        title=first_line[:120],
        n_chars=len(cleaned_text),
    )

    # 3. Questions selection and extractive inference
    field_specs = get_field_specs_for_type(doc_type)
    questions = field_specs["available_questions"]

    engine = load_engine(option)
    answers = engine.answer(cleaned_text, questions, progress_callback=progress_callback)

    # 4. Construct Field instances and attach reliability
    reliability_data = load_reliability_json()

    fields_dict: Dict[str, Optional[Field]] = {}
    dates_dict: Dict[str, Optional[Field]] = {
        "Agreement Date": None,
        "Effective Date": None,
        "Expiration Date": None,
        "Renewal Term": None,
        "Notice Period To Terminate Renewal": None,
    }

    parties_extracted: List[Dict[str, Any]] = []

    for ans in answers:
        q = ans.question
        rel_info = reliability_data.get(q, {})
        rel_level = rel_info.get("reliability", "medium")

        # Create Evidence and Field
        ev = Evidence(start_char=ans.start_char, end_char=ans.end_char, text=ans.text)
        f_obj = Field(
            name=q,
            raw_text=ans.text,
            evidence=ev,
            score=ans.score,
            reliability=rel_level,
            status="unverified",
        )

        # Date parsing
        if q in ("Agreement Date", "Effective Date", "Expiration Date"):
            if ans.text.strip():
                f_obj.parsed = parse_date(ans.text)
            dates_dict[q] = f_obj
        elif q in ("Renewal Term", "Notice Period To Terminate Renewal"):
            dates_dict[q] = f_obj
        elif q == "Parties":
            fields_dict[q] = f_obj
            if ans.text.strip():
                parties_extracted.append({"name": ans.text, "evidence": ev.to_dict()})
        else:
            fields_dict[q] = f_obj

    # 5. Compute Tenure
    eff_text = dates_dict["Effective Date"].raw_text if dates_dict["Effective Date"] else None
    exp_text = dates_dict["Expiration Date"].raw_text if dates_dict["Expiration Date"] else None
    duration_text = dates_dict["Renewal Term"].raw_text if dates_dict["Renewal Term"] else None
    tenure_calc = compute_tenure(eff_text, exp_text, duration_text)

    tenure_info = TenureInfo(
        months=tenure_calc.get("months"),
        days=tenure_calc.get("days"),
        source=tenure_calc.get("source", "unknown"),
        note=tenure_calc.get("note", ""),
    )

    # 6. Run automated checks
    all_fields_for_checks = {**fields_dict, **dates_dict}
    checks_list = run_checks(
        doc_type=doc_type,
        fields=all_fields_for_checks,
        dates=dates_dict,
        tenure_months=tenure_info.months,
        text=cleaned_text,
    )

    # 7. Generate summary
    parties_text = ", ".join([p["name"] for p in parties_extracted])
    gov_law = fields_dict.get("Governing Law")
    active_checks_count = sum(1 for c in checks_list if c.status == "active" and c.severity in ("important", "check"))

    summary_text = generate_summary(
        doc_info=doc_info,
        parties_text=parties_text,
        dates=dates_dict,
        tenure=tenure_info,
        gov_law=gov_law,
        checks_count=active_checks_count,
    )

    # 8. Meta
    meta = MetaInfo(
        option_name=option,
        model_id=str(getattr(engine, "model_path_or_id", option)),
        app_version=APP_VERSION,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    return Report(
        document=doc_info,
        parties=parties_extracted,
        dates=dates_dict,
        tenure=tenure_info,
        key_terms=fields_dict,
        checks=checks_list,
        summary=summary_text,
        meta=meta,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ClauseLens contract analysis CLI.")
    parser.add_argument("--file", required=True, type=Path, help="Path to contract file (.pdf, .docx, .txt)")
    parser.add_argument("--option", default="CUAD (US contracts)", help="Model option name")
    parser.add_argument("--json", type=Path, default=None, help="Save report to JSON file")

    args = parser.parse_args()
    report = analyze_document(args.file, option=args.option)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report written to '{args.json}'")
    else:
        print("\n=== Analysis Report Summary ===")
        print(f"Document: {report.document.title} ({report.document.type})")
        print(f"Summary:  {report.summary}")
        print(f"Active Checks: {len([c for c in report.checks if c.status == 'active'])}")


if __name__ == "__main__":
    main()
