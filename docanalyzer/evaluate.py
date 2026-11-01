"""Evaluation CLI, official SQuAD metric validation, category tables, and error diagnostics."""

from __future__ import annotations

import argparse
import csv
import datetime
import gc
import json
import logging
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForQuestionAnswering, DataCollatorWithPadding, Trainer, TrainingArguments

from docanalyzer.config import AppConfig, PROJECT_ROOT
from docanalyzer.data_loading import load_all
from docanalyzer.model import load_model, load_tokenizer
from docanalyzer.postprocessing import postprocess_qa_predictions
from docanalyzer.preprocessing import preprocess_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Official SQuAD Normalization & Scoring
# -----------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn: Any, prediction: str, ground_truths: List[str]) -> float:
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths) if scores_for_ground_truths else 0.0


# -----------------------------------------------------------------------------
# Evaluation Pipeline
# -----------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: Path | str,
    split_name: str,
    dataset: Optional[Dataset] = None,
    max_answer_length: Optional[int] = None,
    output_dir: Optional[Path] = None,
    batch_size: int = 8,
    save_logits: bool = True,
) -> Dict[str, Any]:
    """Run full evaluation on a dataset split, verifying exact metrics and producing error tables."""
    checkpoint_str = str(checkpoint_path)
    logger.info("Evaluating checkpoint '%s' on split '%s'...", checkpoint_str, split_name)

    cfg = AppConfig()

    # Determine max_answer_length from parameter or train_stats.json
    if max_answer_length is None:
        stats_candidates = [
            Path(checkpoint_path).parent / "train_stats.json",
            PROJECT_ROOT / "models" / "cuad_us_only" / "train_stats.json",
            PROJECT_ROOT / "outputs" / "train_stats.json",
        ]
        for sc in stats_candidates:
            if sc.exists():
                try:
                    with open(sc, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        max_answer_length = data.get("train_answer_stats", {}).get("recommended_max_answer_length")
                        break
                except Exception:
                    pass
        if max_answer_length is None:
            max_answer_length = 346  # Established default from CUAD training run

    # Load data if not supplied
    if dataset is None:
        ds_dict = load_all(cfg)
        dataset = ds_dict[split_name]

    tokenizer = load_tokenizer(checkpoint_str)
    model = load_model(checkpoint_str)

    # 1. Build eval features
    eval_features = dataset.map(
        lambda ex: preprocess_eval(ex, tokenizer, cfg.tokens.max_length, cfg.tokens.stride),
        batched=True,
        num_proc=1,
        remove_columns=[c for c in dataset.column_names if c != "id"],
        desc=f"Tokenizing {split_name} features",
    )

    # 2. Predict logits
    tmp_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "outputs" / "eval_tmp"),
        per_device_eval_batch_size=batch_size,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=tmp_args,
        eval_dataset=eval_features,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    logger.info("Predicting logits for %d features...", len(eval_features))
    predictions_raw = trainer.predict(eval_features)
    start_logits, end_logits = predictions_raw.predictions

    # Free model memory
    del model
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save logits if requested
    eval_out_dir = output_dir or (PROJECT_ROOT / "outputs" / "eval" / f"{Path(checkpoint_path).name}_{split_name}")
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    if save_logits:
        np.save(eval_out_dir / "start_logits.npy", start_logits)
        np.save(eval_out_dir / "end_logits.npy", end_logits)

    # 3. Postprocess predictions
    preds = postprocess_qa_predictions(
        examples=dataset,
        features=eval_features,
        raw_predictions=(start_logits, end_logits),
        n_best_size=cfg.tokens.n_best_size,
        max_answer_length=max_answer_length,
        return_details=True,
    )

    with open(eval_out_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

    # 4. Official SQuAD evaluation
    squad_metric = evaluate.load("squad")
    formatted_preds = [{"id": p["id"], "prediction_text": p["prediction_text"]} for p in preds]
    formatted_refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset]

    official_results = squad_metric.compute(predictions=formatted_preds, references=formatted_refs)
    off_em = float(official_results["exact_match"])
    off_f1 = float(official_results["f1"])

    # 5. Custom per-example scoring & consistency verification
    pred_map = {p["id"]: p["prediction_text"] for p in preds}
    row_details: List[Dict[str, Any]] = []

    sum_em = 0.0
    sum_f1 = 0.0

    for ex in dataset:
        ex_id = str(ex["id"])
        question = str(ex["question"])
        gold_answers = ex["answers"]["text"]
        pred_text = pred_map.get(ex_id, "")

        em_val = 100.0 if metric_max_over_ground_truths(exact_match_score, pred_text, gold_answers) else 0.0
        f1_val = 100.0 * metric_max_over_ground_truths(f1_score, pred_text, gold_answers)

        sum_em += em_val
        sum_f1 += f1_val

        # Best matching gold answer
        norm_pred = normalize_answer(pred_text)
        best_gold = ""
        best_gold_f1 = -1.0
        for g in gold_answers:
            cur_f1 = f1_score(pred_text, g)
            if cur_f1 > best_gold_f1:
                best_gold_f1 = cur_f1
                best_gold = g
        norm_gold = normalize_answer(best_gold)

        # Classify error type
        if em_val == 100.0:
            err_type = "exact"
        elif f1_val == 0.0:
            err_type = "no overlap"
        elif norm_gold in norm_pred:
            err_type = "prediction longer"
        elif norm_pred in norm_gold:
            err_type = "prediction shorter"
        else:
            err_type = "partly overlapping"

        # F1 bucket
        if f1_val == 0.0:
            f1_bucket = "0"
        elif f1_val <= 30.0:
            f1_bucket = ">0 to 30"
        elif f1_val <= 60.0:
            f1_bucket = "30 to 60"
        elif f1_val <= 90.0:
            f1_bucket = "60 to 90"
        else:
            f1_bucket = "90 to 100"

        row_details.append({
            "id": ex_id,
            "category": question,
            "em": em_val,
            "f1": f1_val,
            "error_type": err_type,
            "f1_bucket": f1_bucket,
            "pred_words": len(pred_text.split()),
            "gold_words": len(best_gold.split()),
        })

    custom_em = sum_em / len(dataset)
    custom_f1 = sum_f1 / len(dataset)

    # Metric consistency assertion (within 0.01 tolerance)
    if abs(custom_em - off_em) > 0.01 or abs(custom_f1 - off_f1) > 0.01:
        raise ValueError(
            f"Metric mismatch! Official: EM={off_em:.2f}, F1={off_f1:.2f} | "
            f"Custom: EM={custom_em:.2f}, F1={custom_f1:.2f}"
        )

    logger.info("Evaluation results validated: Exact Match = %.2f, F1 = %.2f", off_em, off_f1)

    # 6. Generate and save CSV tables
    # Category table
    cat_groups = collections.defaultdict(list)
    for r in row_details:
        cat_groups[r["category"]].append(r)

    cat_rows: List[Dict[str, Any]] = []
    for cat, items in cat_groups.items():
        n = len(items)
        c_em = sum(x["em"] for x in items) / n
        c_f1 = sum(x["f1"] for x in items) / n
        c_gw = sum(x["gold_words"] for x in items) / n
        c_pw = sum(x["pred_words"] for x in items) / n
        cat_rows.append({
            "category": cat,
            "rows": n,
            "exact_match": round(c_em, 1),
            "f1": round(c_f1, 1),
            "mean_gold_words": round(c_gw, 1),
            "mean_pred_words": round(c_pw, 1),
        })

    cat_rows.sort(key=lambda x: x["exact_match"])

    per_category_csv = eval_out_dir / "per_category_metrics.csv"
    with open(per_category_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cat_rows)

    # Write reliability.json from category CSV
    from docanalyzer.product.reliability import write_reliability_json
    write_reliability_json(per_category_csv, eval_out_dir / "reliability.json")

    # Error types summary
    err_counts = Counter(r["error_type"] for r in row_details)
    with open(eval_out_dir / "error_types.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["error_type", "count", "percentage"])
        for et, count in err_counts.most_common():
            writer.writerow([et, count, round(count / len(row_details) * 100, 1)])

    # F1 buckets summary
    bucket_counts = Counter(r["f1_bucket"] for r in row_details)
    with open(eval_out_dir / "f1_buckets.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["f1_bucket", "count", "percentage"])
        for b_name in ["0", ">0 to 30", "30 to 60", "60 to 90", "90 to 100"]:
            cnt = bucket_counts.get(b_name, 0)
            writer.writerow([b_name, cnt, round(cnt / len(row_details) * 100, 1)])

    summary = {
        "checkpoint": checkpoint_str,
        "split": split_name,
        "num_examples": len(dataset),
        "exact_match": off_em,
        "f1": off_f1,
        "max_answer_length": max_answer_length,
        "output_dir": str(eval_out_dir),
    }

    with open(eval_out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def evaluate_all_checkpoints(run_dir: Path, split_name: str = "validation") -> Path:
    """Evaluate all checkpoints in run_dir on validation and write best_checkpoint.json."""
    checkpoints = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
    if not checkpoints:
        raise ValueError(f"No checkpoint directories found in '{run_dir}'")

    logger.info("Found %d checkpoints to score in '%s'", len(checkpoints), run_dir)
    results: List[Dict[str, Any]] = []

    for ckpt in checkpoints:
        res = evaluate_checkpoint(ckpt, split_name=split_name)
        results.append(res)

    # Select best by EM (tie-breaker: F1)
    results.sort(key=lambda x: (x["exact_match"], x["f1"]), reverse=True)
    best = results[0]

    best_file = run_dir / "best_checkpoint.json"
    with open(best_file, "w", encoding="utf-8") as f:
        json.dump({
            "best_checkpoint": best["checkpoint"],
            "exact_match": best["exact_match"],
            "f1": best["f1"],
            "evaluated_split": split_name,
            "all_results": results,
        }, f, indent=2)

    logger.info("Best checkpoint: %s (EM=%.2f, F1=%.2f)", best["checkpoint"], best["exact_match"], best["f1"])
    return best_file


def main() -> None:
    parser = argparse.ArgumentParser(description="ClauseLens model evaluation and SQuAD metrics CLI.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path or repo ID of model checkpoint")
    parser.add_argument("--split", choices=["validation", "test"], default="validation", help="Dataset split to evaluate")
    parser.add_argument("--max-answer-length", type=int, default=None, help="Upper bound on answer token length")
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory containing checkpoints for --all-checkpoints")
    parser.add_argument("--all-checkpoints", action="store_true", help="Evaluate all checkpoints in run-dir")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows for debugging")
    parser.add_argument("--confirm-test", action="store_true", help="Explicit confirmation required to evaluate on test split")

    args = parser.parse_args()

    if args.split == "test" and not args.confirm_test:
        raise PermissionError(
            "Evaluating on the test split requires --confirm-test to prevent test-set data leakage."
        )

    if args.all-checkpoints or (args.run_dir and not args.checkpoint):
        if not args.run_dir:
            raise ValueError("--run-dir is required when using --all-checkpoints.")
        evaluate_all_checkpoints(args.run_dir, split_name=args.split)
        return

    if not args.checkpoint:
        raise ValueError("Either --checkpoint or --run-dir --all-checkpoints must be specified.")

    # Log test set evaluations to outputs/test_runs.log
    if args.split == "test":
        log_file = PROJECT_ROOT / "outputs" / "test_runs.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"[{now}] checkpoint={args.checkpoint} split=test limit={args.max_rows}\n")

    res = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        split_name=args.split,
        max_answer_length=args.max_answer_length,
    )
    print("\n=== Evaluation Results ===")
    print(f"Checkpoint: {res['checkpoint']}")
    print(f"Split: {res['split']} (Rows: {res['num_examples']})")
    print(f"Exact Match: {res['exact_match']:.2f}%")
    print(f"F1 Score:    {res['f1']:.2f}%")
    print(f"Reports saved to: {res['output_dir']}\n")


if __name__ == "__main__":
    main()

