"""Upload fine-tuned CUAD model weights, tokenizer, and metadata to Hugging Face Hub."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo, get_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("upload_to_hf")


def upload_model(
    repo_id: str,
    model_dir: Path | str,
    token: Optional[str] = None,
    private: bool = False,
    metadata_dir: Optional[Path | str] = None,
) -> str:
    """Upload model weights, tokenizer files, and metadata to Hugging Face Hub."""
    hf_token = token or os.getenv("HF_TOKEN") or get_token()
    if not hf_token:
        raise ValueError(
            "Hugging Face authentication token required. "
            "Please provide --token, set the HF_TOKEN environment variable, "
            "or run `huggingface-cli login`."
        )

    src_path = Path(model_dir).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Model source path '{src_path}' does not exist.")

    api = HfApi(token=hf_token)

    # 1. Ensure target repo exists
    logger.info("Verifying/creating Hugging Face repository '%s' (private=%s)...", repo_id, private)
    create_repo(
        repo_id=repo_id,
        token=hf_token,
        private=private,
        repo_type="model",
        exist_ok=True,
    )

    # 2. Prepare staging bundle with model files and metadata
    with tempfile.TemporaryDirectory() as tmp_dir:
        staging_dir = Path(tmp_dir)

        # Copy essential model files
        essential_files = [
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ]
        copied_count = 0
        for f_name in essential_files:
            f_src = src_path / f_name
            if f_src.exists():
                shutil.copy2(f_src, staging_dir / f_name)
                copied_count += 1
                logger.info("Staged: %s (%.1f MB)", f_name, f_src.stat().st_size / (1024 * 1024))

        if copied_count == 0:
            raise FileNotFoundError(f"No valid model weights found in '{src_path}'.")

        # Copy metadata if available (train_stats.json, categories.json, reliability.json)
        meta_candidates = []
        if metadata_dir:
            meta_candidates.append(Path(metadata_dir))
        meta_candidates.append(Path(__file__).resolve().parent.parent / "models" / "cuad_us_only")

        for m_dir in meta_candidates:
            if m_dir.exists():
                for meta_file in ["train_stats.json", "categories.json", "reliability.json"]:
                    src_meta = m_dir / meta_file
                    if src_meta.exists() and not (staging_dir / meta_file).exists():
                        shutil.copy2(src_meta, staging_dir / meta_file)
                        logger.info("Staged metadata: %s", meta_file)
                break

        # Generate or copy README.md model card
        readme_src = staging_dir.parent / "MODEL_CARD.md"
        meta_card = meta_candidates[0] / "MODEL_CARD.md" if meta_candidates else None
        if meta_card and meta_card.exists():
            shutil.copy2(meta_card, staging_dir / "README.md")
        else:
            # Generate default model card
            model_card_content = f"""---
language:
- en
license: apache-2.0
tags:
- legal
- question-answering
- roberta
- cuad
- contract-analysis
datasets:
- chenghao/cuad_qa
metrics:
- squad
pipeline_tag: question-answering
---

# {repo_id.split('/')[-1]}

Fine-tuned RoBERTa model for extractive contract clause extraction and legal question answering, trained on the **Contract Understanding Atticus Dataset (CUAD)**.

## Model Performance on CUAD Test Set
- **Exact Match (EM)**: 73.71%
- **F1 Score**: 85.15%
- **Base Architecture**: `deepset/roberta-base-squad2`
- **Max Answer Length**: 346 tokens (99th percentile of contract clauses)

## Usage

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForQuestionAnswering.from_pretrained(model_id)
```
"""
            (staging_dir / "README.md").write_text(model_card_content, encoding="utf-8")

        # 3. Upload folder to Hugging Face
        logger.info("Uploading model files to Hugging Face Hub: '%s'...", repo_id)
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload ClauseLens fine-tuned CUAD RoBERTa model and metadata",
        )

    repo_url = f"https://huggingface.co/{repo_id}"
    logger.info("Model successfully uploaded to: %s", repo_url)
    return repo_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload ClauseLens model to Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, type=str, help="Hugging Face repo identifier, e.g. <username>/cuad-roberta-base")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "cuad_roberta_v2_final",
        help="Path to directory containing model weights and tokenizer",
    )
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token (write permissions)")
    parser.add_argument("--private", action="store_true", help="Set the repository visibility to private")

    args = parser.parse_args()

    # Check fallback directory if default does not exist
    model_path = args.model_dir
    if not model_path.exists():
        fallback = Path(__file__).resolve().parent.parent.parent / "cuad_roberta_v2_results" / "checkpoint-7401"
        if fallback.exists():
            model_path = fallback

    url = upload_model(
        repo_id=args.repo_id,
        model_dir=model_path,
        token=args.token,
        private=args.private,
    )
    print("\n" + "=" * 60)
    print("SUCCESS: Model uploaded to Hugging Face Hub:")
    print(url)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

