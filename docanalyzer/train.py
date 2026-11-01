"""Training CLI for Stage 1 (CUAD) and Stage 2 (Domain Adaptation)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from docanalyzer.config import AppConfig, PROJECT_ROOT, save_run_config
from docanalyzer.data_analysis import analyze_dataset_stats, compute_max_answer_length
from docanalyzer.data_loading import load_all, load_local_json
from docanalyzer.model import build_trainer, build_training_args, load_model, load_tokenizer
from docanalyzer.preprocessing import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_train(
    run_name: str,
    base_checkpoint: Optional[str] = None,
    data_source: str = "cuad",
    learning_rate: Optional[float] = None,
    epochs: Optional[int] = None,
    smoke: bool = False,
    max_rows: Optional[int] = None,
    resume: bool = False,
) -> Path:
    """Execute training pipeline and save checkpoints."""
    cfg = AppConfig()

    if base_checkpoint:
        cfg.train.base_checkpoint = base_checkpoint
    if learning_rate is not None:
        cfg.train.learning_rate = learning_rate
    if epochs is not None:
        cfg.train.epochs = epochs

    output_run_dir = PROJECT_ROOT / "outputs" / run_name
    output_run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    logger.info("Loading dataset (source: %s)...", data_source)
    if data_source.lower() == "cuad":
        ds_dict = load_all(cfg)
    else:
        ds_dict = load_local_json(data_source, val_fraction=cfg.data.val_fraction, seed=cfg.data.split_seed)

    # Apply row limits if smoke or max_rows requested
    if smoke:
        logger.info("SMOKE MODE: subsampling to 20 rows and 20 training steps.")
        cfg.train.epochs = 1
        cfg.train.logging_steps = 1
        for split in ds_dict.keys():
            ds_dict[split] = ds_dict[split].select(range(min(20, len(ds_dict[split]))))
    elif max_rows is not None:
        for split in ds_dict.keys():
            limit = min(max_rows, len(ds_dict[split]))
            ds_dict[split] = ds_dict[split].select(range(limit))

    # 2. Tokenizer and statistics
    tokenizer = load_tokenizer(cfg.train.base_checkpoint)
    max_len = compute_max_answer_length(ds_dict["train"], tokenizer)
    cfg.tokens.max_answer_length = max_len

    analyze_dataset_stats(
        train_ds=ds_dict["train"],
        val_ds=ds_dict.get("validation"),
        test_ds=ds_dict.get("test"),
        tokenizer=tokenizer,
        output_dir=output_run_dir,
    )

    # 3. Features & Negative Chunk Filtering
    features_dict = build_features(ds_dict, tokenizer, cfg, is_training=True)
    train_features = features_dict["train"]
    val_features = features_dict.get("validation")

    train_features.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "start_positions", "end_positions"],
    )
    if val_features is not None:
        val_features.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "start_positions", "end_positions"],
        )

    # 4. Model & Trainer
    training_args = build_training_args(cfg, output_run_dir)
    if smoke:
        training_args.max_steps = 20

    model = load_model(cfg.train.base_checkpoint)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_features,
        eval_dataset=val_features,
    )

    # 5. Train
    logger.info("Commencing training for run '%s'...", run_name)
    train_result = trainer.train(resume_from_checkpoint=resume)

    # Save final model state
    final_model_dir = output_run_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    # Save run config and log history
    save_run_config(
        cfg,
        output_run_dir,
        extra_meta={
            "run_name": run_name,
            "train_runtime_seconds": train_result.metrics.get("train_runtime", 0),
            "global_step": train_result.global_step,
            "train_loss": train_result.training_loss,
        },
    )

    with open(output_run_dir / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    logger.info("Training complete. Outputs saved in '%s'.", output_run_dir)
    print("\nNext step: evaluate validation checkpoints to pick the best epoch:")
    print(f"python -m docanalyzer.evaluate --run-dir outputs/{run_name} --all-checkpoints\n")

    return output_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="ClauseLens model training CLI.")
    parser.add_argument("--run-name", required=True, type=str, help="Identifier for this training run")
    parser.add_argument("--base-checkpoint", type=str, default=None, help="Base HF model or checkpoint path")
    parser.add_argument("--data", type=str, default="cuad", help="'cuad' or path to local json")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate override")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--smoke", action="store_true", help="Run 20-step smoke test on 20 samples")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows for debugging")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint in output_dir")

    args = parser.parse_args()

    run_train(
        run_name=args.run_name,
        base_checkpoint=args.base_checkpoint,
        data_source=args.data,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        smoke=args.smoke,
        max_rows=args.max_rows,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

