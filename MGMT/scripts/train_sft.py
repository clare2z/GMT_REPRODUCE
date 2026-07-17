"""
SFT training script for LA-Mo-GMT (supports all methods: GMT, Mo-GMT, LA-GMT, LA-Mo-GMT, RMT).

Usage:
    # LA-Mo-GMT (our full method)
    python scripts/train_sft.py --config configs/code_generation.yaml

    # GMT baseline
    python scripts/train_sft.py --config configs/code_generation.yaml \
        --override mask.method=gmt

    # Vanilla SFT baseline
    python scripts/train_sft.py --config configs/code_generation.yaml \
        --override mask.method=none

    # Mo-GMT only
    python scripts/train_sft.py --config configs/code_generation.yaml \
        --override mask.method=mo_gmt

    # LA-GMT only
    python scripts/train_sft.py --config configs/code_generation.yaml \
        --override mask.method=la_gmt
"""

from __future__ import annotations

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path

import torch
import numpy as np

# Fix for PyTorch 2.6+ weights_only=True default: force weights_only=False everywhere
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, 'weights_only': False})

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.la_mo_gmt.trainer import LAMoGMTTrainer
from src.la_mo_gmt.data_utils import load_sft_dataset, SFTDataCollator

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LA-Mo-GMT SFT Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--override", type=str, nargs="*", default=[],
                        help="Override config values, e.g. mask.method=gmt")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from a checkpoint path or True to auto-detect")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(config_path: str, overrides: list[str] = None) -> dict:
    """Load YAML config with optional key=value overrides."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if overrides:
        for ov in overrides:
            keys, value = ov.split("=", 1)
            keys = keys.split(".")
            cfg = config
            for k in keys[:-1]:
                cfg = cfg[k]
            # Try to convert value to appropriate type
            try:
                value = eval(value)
            except (NameError, SyntaxError):
                pass
            cfg[keys[-1]] = value

    return config


def main():
    args = parse_args()
    config = load_config(args.config, args.override)
    set_seed(args.seed)

    # ── Log config ──
    logger.info("=" * 60)
    logger.info(f"LA-Mo-GMT Training — Method: {config['mask']['method']}")
    logger.info(f"Model: {config['model_name_or_path']}")
    logger.info(f"Data: {config['data_path']}")
    logger.info(f"Output: {config['output_dir']}")
    logger.info("=" * 60)

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model with memory optimizations ──
    model_kwargs = {
        "torch_dtype": getattr(torch, config["model"].get("torch_dtype", "bfloat16")),
        "use_cache": False,  # Disable KV cache during training
    }

    if config["model"].get("load_in_8bit"):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif config["model"].get("load_in_4bit"):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, config["model"].get("torch_dtype", "bfloat16")),
        )

    if config["model"].get("use_flash_attention"):
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            logger.warning("flash-attn not installed, using SDPA (PyTorch native fast attention)")
            model_kwargs["attn_implementation"] = "sdpa"
    else:
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        **model_kwargs,
    )

    # Enable gradient checkpointing for memory
    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    model.config.use_cache = False

    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

    # ── Load dataset ──
    train_dataset = load_sft_dataset(
        dataset_path=config["data_path"],
        tokenizer=tokenizer,
        max_length=config["training"]["max_seq_length"],
        prompt_template=config["data"].get("prompt_template", "default"),
        split="train",
    )

    logger.info(f"Train dataset: {len(train_dataset)} examples")

    # ── Data collator ──
    data_collator = SFTDataCollator(tokenizer=tokenizer)

    # ── Training arguments ──
    tc = config["training"]
    logger.info(f"DEBUG: save_strategy={tc.get('save_strategy','steps')} max_steps={tc.get('max_steps',-1)} save_final_model={tc.get('save_final_model',True)}")
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=tc["num_epochs"],
        per_device_train_batch_size=tc["per_device_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler"],
        warmup_ratio=tc["warmup_ratio"],
        weight_decay=tc["weight_decay"],
        logging_steps=tc["logging_steps"],
        save_steps=tc["save_steps"],
        save_total_limit=tc.get("save_total_limit", 3),
        bf16=tc["bf16"],
        optim=tc.get("optim", "adamw_8bit"),
        dataloader_num_workers=tc.get("dataloader_num_workers", 4),
        save_strategy=tc.get("save_strategy", "steps"),
        max_steps=tc.get("max_steps", -1),
        ddp_find_unused_parameters=False,
        remove_unused_columns=True,
        report_to="wandb" if config.get("wandb") else "none",
        run_name=config.get("wandb", {}).get("run_name", None),
        seed=args.seed,
    )

    # ── Create trainer ──
    mc = config["mask"]
    dgmm_cfg = mc.get("dgmm", {})
    trainer = LAMoGMTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        mask_method=mc["method"],
        mask_global_ratio=mc.get("global_ratio", 0.0),
        mask_alpha=mc.get("alpha", 0.0),
        mask_beta1=mc.get("beta1", 0.9),
        mask_min_ratio=mc.get("min_ratio", 0.0),
        mask_max_ratio=mc.get("max_ratio", 0.0),
        mask_warmup_steps=mc.get("warmup_steps", 0),
        dgmm_config=dgmm_cfg,
    )

    # ── Save config to output dir ──
    os.makedirs(config["output_dir"], exist_ok=True)
    with open(os.path.join(config["output_dir"], "config.yaml"), "w") as f:
        yaml.dump(config, f)
    logger.info(f"Config saved to {config['output_dir']}/config.yaml")

    # ── Train (with optional resume) ──
    resume = args.resume_from_checkpoint
    if resume and resume.lower() == "true":
        resume = True
    trainer.train(resume_from_checkpoint=resume)

    # ── Save final model ──
    if tc.get("save_final_model", True):
        logger.info(f"Training complete! Model saved to {final_path}")
    else:
        logger.info("Training complete! Final model saving skipped.")
        final_path = os.path.join(config["output_dir"], "final")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)

    # ── Save training summary ──
    summary = trainer.get_summary()
    summary["model"] = config["model_name_or_path"]
    summary["dataset"] = config["data_path"]
    with open(os.path.join(config["output_dir"], "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary: {json.dumps(summary, indent=2)}")

    # ── GPU memory ──
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i} peak: {torch.cuda.max_memory_allocated(i)/1e9:.1f} GB")

    if tc.get("save_final_model", True):
        logger.info(f"Training complete! Model saved to {final_path}")
    else:
        logger.info("Training complete! Final model saving skipped.")


if __name__ == "__main__":
    main()
