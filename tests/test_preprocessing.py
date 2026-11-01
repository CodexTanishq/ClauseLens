"""Tests for preprocessing, sliding window chunking, and label alignment."""

from __future__ import annotations

try:
    import pytest
except ImportError:
    pytest = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

from docanalyzer.preprocessing import filter_empty_chunks, preprocess_eval, preprocess_train


class DummyBatchEncoding(dict):
    def __init__(self, data, seq_ids):
        super().__init__(data)
        self._seq_ids = seq_ids

    def sequence_ids(self, idx):
        return self._seq_ids[idx]


class DummyTokenizer:
    """Lightweight dummy tokenizer for testing logic without model downloads."""

    def __init__(self):
        self.cls_token_id = 0
        self.sep_token_id = 2
        self.pad_token_id = 1

    def __call__(
        self,
        questions,
        contexts,
        truncation=None,
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    ):
        input_ids = [[0] * 50, [0] * 50]
        attention_mask = [[1] * 50, [1] * 50]
        sample_mapping = [0, 0]

        offsets_0 = [(0, 0)] + [(i * 4, (i + 1) * 4) for i in range(48)] + [(0, 0)]
        offsets_1 = [(0, 0)] + [(150 + i * 4, 150 + (i + 1) * 4) for i in range(48)] + [(0, 0)]

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "overflow_to_sample_mapping": sample_mapping,
            "offset_mapping": [offsets_0, offsets_1],
        }

        seq_ids = [
            [None] + [1] * 48 + [None],
            [None] + [1] * 48 + [None],
        ]

        return DummyBatchEncoding(data, seq_ids)


def test_chunk_b_only_positive():
    """Verify chunk A gets CLS (0) when answer is in chunk B only."""
    tokenizer = DummyTokenizer()
    # Answer at char 250..270 (only fits in chunk 1, which covers 150..350)
    examples = {
        "question": ["What is the governing law?"],
        "context": ["A" * 150 + "B" * 200],
        "answers": [{"text": ["B" * 20], "answer_start": [250]}],
    }

    features = preprocess_train(examples, tokenizer, max_length=512, stride=128)
    # Chunk 0 should have CLS label (0)
    assert features["start_positions"][0] == 0
    assert features["end_positions"][0] == 0

    # Chunk 1 should have positive span (> 0)
    assert features["start_positions"][1] > 0
    assert features["end_positions"][1] >= features["start_positions"][1]


def test_two_answers_different_chunks():
    """Verify both chunks become positive when each holds an answer."""
    tokenizer = DummyTokenizer()
    # Answer 1 at char 20..40 (fits chunk 0)
    # Answer 2 at char 250..270 (fits chunk 1)
    examples = {
        "question": ["Who signs?"],
        "context": ["A" * 150 + "B" * 200],
        "answers": [{
            "text": ["A" * 20, "B" * 20],
            "answer_start": [20, 250],
        }],
    }

    features = preprocess_train(examples, tokenizer, max_length=512, stride=128)
    # Both chunks should have positive labels (> 0)
    assert features["start_positions"][0] > 0
    assert features["start_positions"][1] > 0


def test_preprocess_eval_retains_metadata():
    """Verify eval preprocessing stores example_id and masks non-context offsets."""
    tokenizer = DummyTokenizer()
    examples = {
        "id": ["doc_123"],
        "question": ["What is the term?"],
        "context": ["Context string here..."],
    }
    features = preprocess_eval(examples, tokenizer, max_length=512, stride=128)
    assert "example_id" in features
    assert features["example_id"][0] == "doc_123"
    # Special tokens at beginning/end should be masked to None
    assert features["offset_mapping"][0][0] is None
