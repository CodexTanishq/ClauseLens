"""Dataset loading, row deduplication/merging, and contract-level splitting."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

from docanalyzer.config import AppConfig

logger = logging.getLogger(__name__)


def assert_no_overlap(train_ds: Dataset, val_ds: Dataset, test_ds: Dataset) -> None:
    """Verify strictly that no contract context leaks across train, validation, or test."""
    train_contexts: Set[str] = set(train_ds["context"])
    val_contexts: Set[str] = set(val_ds["context"])
    test_contexts: Set[str] = set(test_ds["context"])

    train_val_leak = train_contexts.intersection(val_contexts)
    train_test_leak = train_contexts.intersection(test_contexts)
    val_test_leak = val_contexts.intersection(test_contexts)

    if train_val_leak:
        raise AssertionError(f"Data leakage detected! {len(train_val_leak)} contexts overlap between Train and Val.")
    if train_test_leak:
        raise AssertionError(f"Data leakage detected! {len(train_test_leak)} contexts overlap between Train and Test.")
    if val_test_leak:
        raise AssertionError(f"Data leakage detected! {len(val_test_leak)} contexts overlap between Val and Test.")

    logger.info("Contract boundary verification passed: 0 context overlap across all splits.")


def merge_rows(ds: Dataset) -> Dataset:
    """Merge duplicate (question, context) pairs into single rows with combined answers.

    Why this uses plain Python lists:
        Iterating over a Hugging Face Dataset row-by-row (e.g., `for row in ds:`)
        incurs significant overhead because PyArrow serializes/deserializes
        individual dictionary records on each index lookup. Extracting entire columns
        at once (`list(ds['col'])`) performs a fast zero-copy extraction into native Python,
        enabling high-speed hash map aggregation that runs in milliseconds.
    """
    ids: List[str] = list(ds["id"])
    questions: List[str] = list(ds["question"])
    contexts: List[str] = list(ds["context"])
    answers_list: List[Dict[str, Any]] = list(ds["answers"])

    grouped: OrderedDict[Tuple[str, str], Dict[str, Any]] = OrderedDict()

    for idx, q, ctx, ans in zip(ids, questions, contexts, answers_list):
        key = (q, ctx)
        if key not in grouped:
            grouped[key] = {
                "id": idx,
                "question": q,
                "context": ctx,
                "seen_answers": set(),
                "texts": [],
                "starts": [],
            }

        texts = ans.get("text", [])
        starts = ans.get("answer_start", [])

        for t, s in zip(texts, starts):
            item = (t, s)
            if item not in grouped[key]["seen_answers"]:
                grouped[key]["seen_answers"].add(item)
                grouped[key]["texts"].append(t)
                grouped[key]["starts"].append(s)

    merged_data: Dict[str, List[Any]] = {
        "id": [],
        "question": [],
        "context": [],
        "answers": [],
    }

    total_answers = 0
    for g in grouped.values():
        merged_data["id"].append(g["id"])
        merged_data["question"].append(g["question"])
        merged_data["context"].append(g["context"])
        merged_data["answers"].append({
            "text": g["texts"],
            "answer_start": g["starts"],
        })
        total_answers += len(g["texts"])

    logger.info(
        "Merged rows: %d -> %d unique (question, context) rows | %d total answers retained",
        len(ds),
        len(merged_data["id"]),
        total_answers,
    )
    return Dataset.from_dict(merged_data)


def split_by_contract(
    ds: Dataset,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Split dataset at the contract level (by unique context) into train and validation sets.

    Ensures that all question-answer pairs for any given contract reside exclusively
    in either train or validation, preventing context leakage.
    Selection uses integer indices over sorted unique contexts.
    """
    contexts: List[str] = list(ds["context"])
    unique_contexts = sorted(set(contexts))

    rng = np.random.default_rng(seed)
    n_val = max(1, round(len(unique_contexts) * val_fraction))
    picked_indices = set(rng.choice(len(unique_contexts), size=n_val, replace=False))
    val_contexts_set = {unique_contexts[i] for i in picked_indices}

    train_indices: List[int] = []
    val_indices: List[int] = []

    for idx, ctx in enumerate(contexts):
        if ctx in val_contexts_set:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    train_split = ds.select(train_indices)
    val_split = ds.select(val_indices)

    logger.info(
        "Contract split (fraction=%.2f, seed=%d): Train=%d rows, Val=%d rows (from %d total contracts)",
        val_fraction,
        seed,
        len(train_split),
        len(val_split),
        len(unique_contexts),
    )
    return train_split, val_split


def load_cuad(cfg: AppConfig) -> DatasetDict:
    """Load the CUAD QA dataset and clean unwanted columns."""
    logger.info("Loading CUAD dataset from '%s'...", cfg.data.dataset_id)
    raw_ds = load_dataset(cfg.data.dataset_id)

    # Clean unused metadata columns if present
    for split_name in list(raw_ds.keys()):
        cols_to_drop = [c for c in cfg.data.columns_to_drop if c in raw_ds[split_name].column_names]
        if cols_to_drop:
            raw_ds[split_name] = raw_ds[split_name].remove_columns(cols_to_drop)

    return raw_ds


def load_local_json(path: Path | str, val_fraction: float = 0.10, seed: int = 42) -> DatasetDict:
    """Load local contract QA JSON and apply row merging and contract splitting."""
    file_path = Path(path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    ds = Dataset.from_list(data)
    merged = merge_rows(ds)
    train_split, val_split = split_by_contract(merged, val_fraction=val_fraction, seed=seed)
    return DatasetDict({"train": train_split, "validation": val_split})


def load_all(cfg: AppConfig) -> DatasetDict:
    """Load CUAD, merge training rows, create contract-level validation split, and verify test."""
    raw_dict = load_cuad(cfg)

    train_raw = raw_dict["train"]
    test_raw = raw_dict["test"]

    # Verify expected raw counts
    if len(train_raw) != 11178:
        logger.warning("Expected 11178 train rows in raw CUAD, got %d", len(train_raw))
    if len(test_raw) != 1244:
        logger.warning("Expected 1244 test rows in raw CUAD, got %d", len(test_raw))

    # Merge train rows
    train_merged = merge_rows(train_raw)
    if len(train_merged) != 5449:
        logger.warning("Expected 5449 merged train rows, got %d", len(train_merged))

    # Split train into train and validation by contract
    train_split, val_split = split_by_contract(
        train_merged,
        val_fraction=cfg.data.val_fraction,
        seed=cfg.data.split_seed,
    )

    if len(val_split) != 527:
        logger.warning("Expected 527 validation rows with default seed, got %d", len(val_split))
    if len(train_split) != 4922:
        logger.warning("Expected 4922 train split rows, got %d", len(train_split))

    # Assert no context overlap
    assert_no_overlap(train_split, val_split, test_raw)

    return DatasetDict({
        "train": train_split,
        "validation": val_split,
        "test": test_raw,
    })

