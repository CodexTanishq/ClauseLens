"""Tests for logit postprocessing, span selection, and answer length constraints."""

from __future__ import annotations

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from docanalyzer.postprocessing import postprocess_qa_predictions


def test_postprocess_picks_best_span_across_chunks():
    """Verify postprocessing selects the candidate with highest logit sum across all chunks."""
    context = "The agreement is governed by the laws of India and exclusive jurisdiction of Mumbai courts."
    examples = [{"id": "ex_1", "context": context}]

    # 2 chunks for the same example
    features = {
        "example_id": ["ex_1", "ex_1"],
        "offset_mapping": [
            [None, (0, 13), (14, 22), (23, 25), (26, 30), None],           # chunk 0
            [None, (33, 40), (41, 46), (47, 52), (53, 56), (57, 63), None], # chunk 1: "laws of India"
        ],
    }

    # Chunk 0 logits (weak score = 2.0 + 2.0 = 4.0)
    c0_start = np.array([0.0, 2.0, 1.0, 0.5, 0.2, 0.0])
    c0_end = np.array([0.0, 1.0, 2.0, 0.5, 0.2, 0.0])

    # Chunk 1 logits (strong score = 5.0 + 6.0 = 11.0 on token 1..3 -> chars 33..52: "laws of India")
    c1_start = np.array([0.0, 5.0, 0.2, 0.1, 0.0, 0.0, 0.0])
    c1_end = np.array([0.0, 0.2, 0.1, 6.0, 0.0, 0.0, 0.0])

    start_logits = np.array([c0_start, c1_start], dtype=object)
    end_logits = np.array([c0_end, c1_end], dtype=object)

    preds = postprocess_qa_predictions(
        examples=examples,
        features=features,
        raw_predictions=(start_logits, end_logits),
        n_best_size=5,
        max_answer_length=30,
        return_details=True,
    )

    assert len(preds) == 1
    p = preds[0]
    assert p["id"] == "ex_1"
    assert "laws of India" in p["prediction_text"]
    assert p["score"] >= 11.0


def test_postprocess_respects_max_answer_length():
    """Verify spans exceeding max_answer_length are discarded even if logits are high."""
    context = "A very long sentence that has many words representing a full multi-clause section."
    examples = [{"id": "ex_long", "context": context}]

    # Single chunk with 8 tokens
    offsets = [(i * 5, (i + 1) * 5) for i in range(8)]
    features = {
        "example_id": ["ex_long"],
        "offset_mapping": [offsets],
    }

    # Span from token 0 to token 7 has 8 tokens
    start_logits = np.array([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    end_logits = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]])

    # When max_answer_length=3, the 8-token span must be rejected
    preds = postprocess_qa_predictions(
        examples=examples,
        features=features,
        raw_predictions=(start_logits, end_logits),
        n_best_size=5,
        max_answer_length=3,
        return_details=True,
    )

    assert len(preds) == 1
    # Resulting prediction must have token span <= 3
    assert len(preds[0]["prediction_text"].split()) <= 4


def test_postprocess_rejects_missing_max_answer_length():
    """Verify ValueError is raised if max_answer_length is None or <= 0."""
    try:
        postprocess_qa_predictions(
            examples=[],
            features={"example_id": [], "offset_mapping": []},
            raw_predictions=(np.array([]), np.array([])),
            n_best_size=20,
            max_answer_length=None,  # Must fail
        )
        assert False, "Expected ValueError when max_answer_length is None"
    except ValueError:
        pass
