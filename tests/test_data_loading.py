"""Tests for row deduplication / merging, contract-level splitting, and context overlap assertions."""

from __future__ import annotations

from datasets import Dataset

from docanalyzer.data_loading import assert_no_overlap, merge_rows, split_by_contract


def test_merge_rows_combines_answers():
    """Verify repeated (question, context) pairs are merged and duplicates dropped."""
    raw_data = {
        "id": ["id_1", "id_2", "id_3"],
        "question": ["What is governing law?", "What is governing law?", "What is the term?"],
        "context": ["Contract context A", "Contract context A", "Contract context B"],
        "answers": [
            {"text": ["laws of India"], "answer_start": [10]},
            {"text": ["laws of India", "jurisdiction of Mumbai"], "answer_start": [10, 40]},
            {"text": ["11 months"], "answer_start": [5]},
        ],
    }
    ds = Dataset.from_dict(raw_data)
    merged = merge_rows(ds)

    assert len(merged) == 2
    # Verify answers for first question
    ans_0 = merged[0]["answers"]
    assert len(ans_0["text"]) == 2
    assert "laws of India" in ans_0["text"]
    assert "jurisdiction of Mumbai" in ans_0["text"]


def test_split_by_contract_zero_leakage():
    """Verify split_by_contract keeps contracts strictly disjoint between train and val."""
    raw_data = {
        "id": [f"id_{i}" for i in range(10)],
        "question": [f"Q_{i}" for i in range(10)],
        "context": [f"Contract context {i % 4}" for i in range(10)],
        "answers": [{"text": ["Sample"], "answer_start": [0]} for _ in range(10)],
    }
    ds = Dataset.from_dict(raw_data)
    train_split, val_split = split_by_contract(ds, val_fraction=0.25, seed=42)

    train_contexts = set(train_split["context"])
    val_contexts = set(val_split["context"])

    assert len(train_contexts.intersection(val_contexts)) == 0


def test_assert_no_overlap_raises_on_leak():
    """Verify assert_no_overlap catches any leaked context."""
    train_ds = Dataset.from_dict({"context": ["A", "B", "C"]})
    val_ds = Dataset.from_dict({"context": ["D", "E"]})
    test_ds = Dataset.from_dict({"context": ["B", "F"]})  # "B" leaks from train!

    try:
        assert_no_overlap(train_ds, val_ds, test_ds)
        assert False, "Should have raised AssertionError due to context leakage"
    except AssertionError:
        pass

