"""Post-processing raw start/end logits into character-level answer text predictions."""

from __future__ import annotations

import collections
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def postprocess_qa_predictions(
    examples: Any,
    features: Any,
    raw_predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int,
    max_answer_length: int,
    return_details: bool = False,
) -> List[Dict[str, Any]]:
    """Convert raw start and end logits across sliding window chunks into character-level answer spans.

    Args:
        examples: Dataset or list-of-dicts containing the original document examples.
        features: Tokenized features dataset containing 'example_id' and 'offset_mapping'.
        raw_predictions: Tuple of (start_logits, end_logits).
        n_best_size: Number of candidate start/end logits to consider per chunk.
        max_answer_length: Maximum allowed answer length in tokens. (Explicitly required).
        return_details: If True, include start_char, end_char, and logit score in output.

    Returns:
        List of prediction dictionaries compatible with SQuAD evaluation.
    """
    if max_answer_length is None or max_answer_length <= 0:
        raise ValueError("max_answer_length must be explicitly specified and > 0. Never hardcode 30.")

    all_start_logits, all_end_logits = raw_predictions

    # Read feature columns ONCE into Python lists for fast indexing
    feature_example_ids: List[str] = list(features["example_id"])
    feature_offset_mappings: List[List[Optional[Tuple[int, int]]]] = list(features["offset_mapping"])

    # Map each example_id to the list of feature indices belonging to it
    example_to_features = collections.defaultdict(list)
    for idx, feature_ex_id in enumerate(feature_example_ids):
        example_to_features[feature_ex_id].append(idx)

    # Read original examples mapping
    examples_dict: Dict[str, str] = {}
    if hasattr(examples, "__getitem__") and "id" in examples and "context" in examples:
        # Hugging Face Dataset format
        ex_ids = list(examples["id"])
        ex_contexts = list(examples["context"])
        for i, c in zip(ex_ids, ex_contexts):
            examples_dict[str(i)] = str(c)
    else:
        # List of records or dict format
        for ex in examples:
            examples_dict[str(ex["id"])] = str(ex["context"])

    predictions: List[Dict[str, Any]] = []

    for ex_id, context in examples_dict.items():
        feature_indices = example_to_features.get(ex_id, [])

        best_score = -float("inf")
        best_span = ("", 0, 0)  # (text, start_char, end_char)

        for feature_idx in feature_indices:
            start_logits = all_start_logits[feature_idx]
            end_logits = all_end_logits[feature_idx]
            offset_mapping = feature_offset_mappings[feature_idx]

            # Find top n_best start and end token indices
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Validate boundary conditions
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    score = float(start_logits[start_index] + end_logits[end_index])
                    if score > best_score:
                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        text = context[start_char:end_char].strip()

                        best_score = score
                        best_span = (text, start_char, end_char)

        pred_record: Dict[str, Any] = {
            "id": ex_id,
            "prediction_text": best_span[0],
        }
        if return_details:
            pred_record["score"] = best_score if best_score != -float("inf") else 0.0
            pred_record["start_char"] = best_span[1]
            pred_record["end_char"] = best_span[2]

        predictions.append(pred_record)

    return predictions

