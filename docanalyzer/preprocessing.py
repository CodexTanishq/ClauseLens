"""Tokenization, sliding window chunking, label alignment, and negative chunk filtering."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset, DatasetDict

from docanalyzer.config import AppConfig

logger = logging.getLogger(__name__)


def preprocess_train(
    examples: Dict[str, List[Any]],
    tokenizer: Any,
    max_length: int = 512,
    stride: int = 128,
) -> Dict[str, Any]:
    """Tokenize question-context pairs with sliding window chunking and align answer token spans.

    Iterates through all gold answers for each example and assigns the first answer
    that fits completely within the context of the current chunk. Chunks without a fitting
    answer are assigned the CLS token index.
    """
    questions = [str(q).strip() for q in examples["question"]]
    contexts = [str(c) for c in examples["context"]]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions: List[int] = []
    end_positions: List[int] = []

    cls_index = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        sequence_ids = tokenized.sequence_ids(i)

        # Identify context token bounds (sequence_id == 1)
        context_start = 0
        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1

        context_end = len(sequence_ids) - 1
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1

        # Chunk contains no context tokens
        if context_start > context_end or context_start >= len(sequence_ids):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        chunk_start_char = offsets[context_start][0]
        chunk_end_char = offsets[context_end][1]

        answers = examples["answers"][sample_idx]
        gold_texts = answers.get("text", [])
        gold_starts = answers.get("answer_start", [])

        found_fit = False

        # Search for first gold answer that fits entirely inside the chunk
        for text, start_char in zip(gold_texts, gold_starts):
            end_char = start_char + len(text)
            if start_char >= chunk_start_char and end_char <= chunk_end_char:
                # Find start token: first context token whose end offset > start_char
                token_start = context_start
                while token_start <= context_end and offsets[token_start][1] <= start_char:
                    token_start += 1

                # Find end token: last context token whose start offset < end_char
                token_end = context_end
                while token_end >= context_start and offsets[token_end][0] >= end_char:
                    token_end -= 1

                if token_start <= token_end:
                    start_positions.append(token_start)
                    end_positions.append(token_end)
                    found_fit = True
                    break

        if not found_fit:
            start_positions.append(cls_index)
            end_positions.append(cls_index)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def preprocess_eval(
    examples: Dict[str, List[Any]],
    tokenizer: Any,
    max_length: int = 512,
    stride: int = 128,
) -> Dict[str, Any]:
    """Tokenize validation/test examples, keeping example_id and setting non-context offsets to None."""
    questions = [str(q).strip() for q in examples["question"]]
    contexts = [str(c) for c in examples["context"]]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    raw_offsets = tokenized["offset_mapping"]

    cleaned_offsets: List[List[Optional[Tuple[int, int]]]] = []
    example_ids: List[str] = []

    for i in range(len(tokenized["input_ids"])):
        sample_idx = sample_mapping[i]
        example_ids.append(str(examples["id"][sample_idx]))

        sequence_ids = tokenized.sequence_ids(i)
        chunk_offsets = raw_offsets[i]

        filtered_chunk: List[Optional[Tuple[int, int]]] = []
        for o_idx, (seq_id, offset) in enumerate(zip(sequence_ids, chunk_offsets)):
            if seq_id == 1:
                filtered_chunk.append((int(offset[0]), int(offset[1])))
            else:
                filtered_chunk.append(None)
        cleaned_offsets.append(filtered_chunk)

    tokenized["offset_mapping"] = cleaned_offsets
    tokenized["example_id"] = example_ids
    return tokenized


def filter_empty_chunks(dataset: Dataset, keep_prob: float = 0.15, seed: int = 42) -> Dataset:
    """Subsample unanswerable (CLS-labeled) training chunks using a seeded RNG."""
    cls_positions = dataset["start_positions"]
    rng = np.random.default_rng(seed)

    selected_indices: List[int] = []
    for idx, pos in enumerate(cls_positions):
        if pos != 0:
            # Positive chunk: always keep
            selected_indices.append(idx)
        else:
            # Negative chunk: sample with keep_prob
            if rng.random() < keep_prob:
                selected_indices.append(idx)

    logger.info(
        "Seeded negative chunk filter (keep_prob=%.2f, seed=%d): %d -> %d chunks (%.1f%% kept)",
        keep_prob,
        seed,
        len(dataset),
        len(selected_indices),
        (len(selected_indices) / len(dataset) * 100) if len(dataset) > 0 else 0,
    )
    return dataset.select(selected_indices)


def build_features(
    split_ds: DatasetDict,
    tokenizer: Any,
    cfg: AppConfig,
    is_training: bool = True,
) -> DatasetDict:
    """Build processed feature datasets for training and evaluation splits."""
    proc = cfg.train.num_proc

    processed = {}
    if "train" in split_ds:
        logger.info("Processing train features...")
        cols_to_remove = split_ds["train"].column_names
        tr_features = split_ds["train"].map(
            lambda ex: preprocess_train(ex, tokenizer, cfg.tokens.max_length, cfg.tokens.stride),
            batched=True,
            num_proc=proc,
            remove_columns=cols_to_remove,
            desc="Tokenizing train split",
        )
        if is_training and cfg.data.empty_chunk_keep_prob < 1.0:
            tr_features = filter_empty_chunks(
                tr_features,
                keep_prob=cfg.data.empty_chunk_keep_prob,
                seed=cfg.data.filter_seed,
            )
        processed["train"] = tr_features

    for eval_split in ("validation", "test"):
        if eval_split in split_ds:
            logger.info("Processing %s features...", eval_split)
            cols_to_remove = [c for c in split_ds[eval_split].column_names if c != "id"]
            eval_features = split_ds[eval_split].map(
                lambda ex: preprocess_eval(ex, tokenizer, cfg.tokens.max_length, cfg.tokens.stride),
                batched=True,
                num_proc=proc,
                remove_columns=cols_to_remove,
                desc=f"Tokenizing {eval_split} split",
            )
            processed[eval_split] = eval_features

    return DatasetDict(processed)

