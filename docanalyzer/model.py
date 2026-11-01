"""Model and tokenizer loaders, training argument builder, and Hugging Face Trainer factory."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from docanalyzer.config import AppConfig

logger = logging.getLogger(__name__)


def load_tokenizer(path_or_id: str | Path, token: Optional[str] = None) -> AutoTokenizer:
    """Load Hugging Face tokenizer from local checkpoint path or remote repository."""
    hf_token = token or os.getenv("HF_TOKEN")
    logger.info("Loading tokenizer from '%s'...", str(path_or_id))
    return AutoTokenizer.from_pretrained(str(path_or_id), token=hf_token)


def load_model(
    path_or_id: str | Path,
    device: Optional[str] = None,
    token: Optional[str] = None,
) -> AutoModelForQuestionAnswering:
    """Load Question Answering model from local path or Hugging Face hub."""
    hf_token = token or os.getenv("HF_TOKEN")
    logger.info("Loading QA model from '%s'...", str(path_or_id))
    model = AutoModelForQuestionAnswering.from_pretrained(str(path_or_id), token=hf_token)

    if device:
        if device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = device
        model.to(target_device)

    return model


def build_training_args(cfg: AppConfig, output_dir: Path | str) -> TrainingArguments:
    """Construct Hugging Face TrainingArguments.

    Ensures all epoch checkpoints are retained (save_total_limit=None)
    and load_best_model_at_end=False (best epoch is selected later via validation EM/F1).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Use eval_strategy for modern transformers; fallback to evaluation_strategy if needed
    kwargs = {
        "output_dir": str(out_path),
        "learning_rate": cfg.train.learning_rate,
        "num_train_epochs": cfg.train.epochs,
        "per_device_train_batch_size": cfg.train.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.train.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
        "weight_decay": cfg.train.weight_decay,
        "gradient_checkpointing": cfg.train.gradient_checkpointing,
        "fp16": cfg.train.fp16,
        "logging_steps": cfg.train.logging_steps,
        "seed": cfg.train.seed,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": None,            # Keep ALL epoch checkpoints
        "load_best_model_at_end": False,     # Picked post-hoc with SQuAD EM/F1
        "report_to": "none",
        "save_only_model": cfg.train.save_only_model,
    }

    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        # Compatibility with older transformers naming
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return TrainingArguments(**kwargs)


def build_trainer(
    model: AutoModelForQuestionAnswering,
    tokenizer: AutoTokenizer,
    training_args: TrainingArguments,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
) -> Trainer:
    """Construct Trainer instance with dynamic token padding and no in-loop token accuracy."""
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,  # Token accuracy is misleading; SQuAD post-processing used instead
    )

