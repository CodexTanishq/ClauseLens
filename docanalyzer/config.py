"""Configuration dataclasses, model registry, and path resolution for ClauseLens."""

from __future__ import annotations

import datetime
import json
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

APP_NAME = "ClauseLens"
APP_VERSION = "0.1.0"

# Project root resolved dynamically via pathlib - no hardcoded absolute user paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent


@dataclass
class DataConfig:
    """Settings for dataset loading, row deduplication, and splits."""

    dataset_id: str = "chenghao/cuad_qa"
    columns_to_drop: List[str] = field(default_factory=lambda: ["source", "lan"])
    val_fraction: float = 0.10
    split_seed: int = 42
    empty_chunk_keep_prob: float = 0.15
    filter_seed: int = 42


@dataclass
class TokenConfig:
    """Settings for tokenization, chunking, and extractive span extraction."""

    max_length: int = 512
    stride: int = 128
    n_best_size: int = 20
    max_answer_length: Optional[int] = None


@dataclass
class TrainConfig:
    """Hyperparameters and runtime settings for model training."""

    base_checkpoint: str = "deepset/roberta-base-squad2"
    learning_rate: float = 2e-5
    epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True
    fp16: bool = field(default_factory=lambda: torch.cuda.is_available())
    optim: str = "adamw_torch" if not torch.cuda.is_available() else "adamw_torch_fused"
    logging_steps: int = 50
    seed: int = 42
    num_proc: int = 2
    save_only_model: bool = False


@dataclass
class ReliabilityThresholds:
    """Thresholds for per-category reliability classification."""

    high_min_em: float = 80.0
    medium_min_em: float = 60.0


@dataclass
class AppConfig:
    """Consolidated configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    tokens: TokenConfig = field(default_factory=TokenConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reliability: ReliabilityThresholds = field(default_factory=ReliabilityThresholds)


def get_default_cuad_model_path() -> Path:
    """Resolve default local CUAD checkpoint path or local model directory."""
    env_path = os.getenv("CUAD_MODEL_ID")
    if env_path:
        return Path(env_path)

    # Check relative paths in priority order
    candidates = [
        WORKSPACE_ROOT / "cuad_roberta_v2_results" / "checkpoint-7401",
        WORKSPACE_ROOT / "cuad_roberta_v2_final",
        PROJECT_ROOT / "models" / "cuad_us_only",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def get_model_sources() -> OrderedDict[str, Optional[str]]:
    """Return registry mapping display name to model path/repo id.

    Can be overridden via environment variables or Streamlit secrets.
    """
    default_cuad = str(get_default_cuad_model_path())
    cuad_model_id = os.getenv("CUAD_MODEL_ID", default_cuad)
    indian_model_id = os.getenv("INDIAN_MODEL_ID", None)

    sources: OrderedDict[str, Optional[str]] = OrderedDict()
    sources["CUAD (US contracts)"] = cuad_model_id
    sources["CUAD + Indian contracts"] = indian_model_id
    sources["RAG"] = None
    return sources


MODEL_SOURCES = get_model_sources()


def save_run_config(
    cfg: AppConfig,
    run_dir: Path,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save execution configuration and metadata to outputs/<run_name>/run_config.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    out_file = run_dir / "run_config.json"

    meta: Dict[str, Any] = {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if extra_meta:
        meta.update(extra_meta)

    payload = {
        "metadata": meta,
        "data": asdict(cfg.data),
        "tokens": asdict(cfg.tokens),
        "train": asdict(cfg.train),
        "reliability": asdict(cfg.reliability),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_file

