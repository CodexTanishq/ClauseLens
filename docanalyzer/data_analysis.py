"""Statistical profiling of contract contexts, answer spans, token distributions, and category counts."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset

from docanalyzer.config import AppConfig, PROJECT_ROOT
from docanalyzer.data_loading import load_all
from docanalyzer.model import load_tokenizer

logger = logging.getLogger(__name__)


def compute_answer_token_lengths(ds: Dataset, tokenizer: Any) -> List[int]:
    """Compute token length for every answer in the dataset using the model tokenizer."""
    lengths: List[int] = []
    answers_list: List[Dict[str, Any]] = list(ds["answers"])

    for ans in answers_list:
        texts = ans.get("text", [])
        for t in texts:
            if t.strip():
                tokens = tokenizer(t, add_special_tokens=False)["input_ids"]
                lengths.append(len(tokens))

    return lengths


def compute_max_answer_length(train_ds: Dataset, tokenizer: Any) -> int:
    """Compute ceil(99th percentile of answer token length) exclusively from TRAIN split."""
    lengths = compute_answer_token_lengths(train_ds, tokenizer)
    if not lengths:
        return 30

    p99 = float(np.percentile(lengths, 99))
    max_len = int(math.ceil(p99))
    logger.info("Computed max_answer_length from %d train answers: 99th percentile = %d", len(lengths), max_len)
    return max_len


def analyze_dataset_stats(
    train_ds: Dataset,
    val_ds: Optional[Dataset],
    test_ds: Optional[Dataset],
    tokenizer: Any,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive dataset statistics, answer percentiles, and category counts."""
    train_ans_lengths = compute_answer_token_lengths(train_ds, tokenizer)

    ans_stats = {
        "count": len(train_ans_lengths),
        "median": float(np.median(train_ans_lengths)) if train_ans_lengths else 0.0,
        "p90": float(np.percentile(train_ans_lengths, 90)) if train_ans_lengths else 0.0,
        "p95": float(np.percentile(train_ans_lengths, 95)) if train_ans_lengths else 0.0,
        "p99": float(np.percentile(train_ans_lengths, 99)) if train_ans_lengths else 0.0,
        "max": int(np.max(train_ans_lengths)) if train_ans_lengths else 0,
        "recommended_max_answer_length": int(math.ceil(np.percentile(train_ans_lengths, 99))) if train_ans_lengths else 30,
    }

    # Questions / Categories breakdown
    questions: List[str] = list(train_ds["question"])
    category_counts = dict(Counter(questions).most_common())
    unique_categories = sorted(list(set(questions)))

    stats: Dict[str, Any] = {
        "train_rows": len(train_ds),
        "validation_rows": len(val_ds) if val_ds is not None else 0,
        "test_rows": len(test_ds) if test_ds is not None else 0,
        "train_answer_stats": ans_stats,
        "categories_count": len(unique_categories),
        "category_distribution": category_counts,
        "categories": unique_categories,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        stats_file = output_dir / "train_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved dataset statistics to '%s'", stats_file)

        # Also write categories.json for UI multiselect
        cats_file = output_dir / "categories.json"
        with open(cats_file, "w", encoding="utf-8") as f:
            json.dump(unique_categories, f, indent=2)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dataset statistics and calculate answer token length.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "cuad_us_only", help="Target stats dir")
    args = parser.parse_args()

    cfg = AppConfig()
    tokenizer = load_tokenizer(cfg.train.base_checkpoint)
    ds_dict = load_all(cfg)

    stats = analyze_dataset_stats(
        train_ds=ds_dict["train"],
        val_ds=ds_dict["validation"],
        test_ds=ds_dict["test"],
        tokenizer=tokenizer,
        output_dir=args.output_dir,
    )

    print("\n=== Dataset Analysis Summary ===")
    print(f"Train rows: {stats['train_rows']} | Val rows: {stats['validation_rows']} | Test rows: {stats['test_rows']}")
    print(f"Train answers analyzed: {stats['train_answer_stats']['count']}")
    print(f"Answer token length percentiles: Median={stats['train_answer_stats']['median']:.1f}, "
          f"90th={stats['train_answer_stats']['p90']:.1f}, "
          f"95th={stats['train_answer_stats']['p95']:.1f}, "
          f"99th={stats['train_answer_stats']['p99']:.1f}")
    print(f"Recommended max_answer_length: {stats['train_answer_stats']['recommended_max_answer_length']}")
    print(f"Unique categories: {stats['categories_count']}\n")


if __name__ == "__main__":
    main()

