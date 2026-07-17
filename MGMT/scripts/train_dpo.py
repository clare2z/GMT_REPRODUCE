"""
DPO training script for LA-Mo-GMT.

Memory note for 80GB:
  - DPO requires both policy model + reference model
  - Set load_in_8bit=true in config for 13B models
  - Reference model uses 8-bit quantization to save memory
  - With 7B model: ~50GB (policy) + ~7GB (ref in 8-bit) = ~57GB ✓

Usage:
    python scripts/train_dpo.py --config configs/general_dpo.yaml
"""

from __future__ import annotations

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.la_mo_gmt.trainer import LAMoGMTTrainer
from src.la_mo_gmt.optimizer import GradientMaskOptimizer
from src.la_mo_gmt.data_utils import load_dpo_dataset

logger = logging.getLogger(__name__)



class DPODataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id or 0
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        import torch
        batch = {}
        for key in ["chosen_input_ids", "chosen_attention_mask", "rejected_input_ids", "rejected_attention_mask"]:
            seqs = [f[key] for f in features]
            max_len = max(len(s) for s in seqs)
            m = self.pad_to_multiple_of
            if m and max_len % m != 0:
                max_len = max_len + m - max_len % m
            padded = [s + [self.pad_id] * (max_len - len(s)) for s in seqs]
            batch[key] = torch.tensor(padded)
        return batch
class DPOTrainer(LAMoGMTTrainer):
    """
    DPO Trainer with gradient masking support.

    Uses reference model for the DPO loss computation.
    Memory: reference model loaded in 8-bit to save memory.
    """

    def __init__(
        self,
        reference_model: AutoModelForCausalLM,
        dpo_beta: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reference_model = reference_model
        self.dpo_beta = dpo_beta

        # Freeze reference model
        for p in self.reference_model.parameters():
            p.requires_grad = False
        self.reference_model.eval()

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False, **kwargs,
    ):
        """
        Compute DPO loss with chosen/rejected pairs.

        inputs should contain:
          - chosen_input_ids, chosen_attention_mask
          - rejected_input_ids, rejected_attention_mask
        """
        device = next(model.parameters()).device

        chosen_ids = inputs["chosen_input_ids"].to(device)
        chosen_mask = inputs["chosen_attention_mask"].to(device)
        rejected_ids = inputs["rejected_input_ids"].to(device)
        rejected_mask = inputs["rejected_attention_mask"].to(device)

        # ── Forward for chosen ──
        chosen_logits = model(
            input_ids=chosen_ids,
            attention_mask=chosen_mask,
        ).logits

        # ── Forward for rejected ──
        rejected_logits = model(
            input_ids=rejected_ids,
            attention_mask=rejected_mask,
        ).logits

        # ── Reference model forward (no grad) ──
        with torch.no_grad():
            ref_chosen_logits = self.reference_model(
                input_ids=chosen_ids,
                attention_mask=chosen_mask,
            ).logits
            ref_rejected_logits = self.reference_model(
                input_ids=rejected_ids,
                attention_mask=rejected_mask,
            ).logits

        # ── Compute DPO loss ──
        loss = dpo_loss(
            policy_chosen_logits=chosen_logits,
            policy_rejected_logits=rejected_logits,
            ref_chosen_logits=ref_chosen_logits,
            ref_rejected_logits=ref_rejected_logits,
            chosen_ids=chosen_ids,
            rejected_ids=rejected_ids,
            beta=self.dpo_beta,
        )

        return loss


def dpo_loss(
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    ref_chosen_logits: torch.Tensor,
    ref_rejected_logits: torch.Tensor,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    beta: float = 0.1,
    label_pad_token_id: int = -100,
) -> torch.Tensor:
    """
    Standard DPO loss: -log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))
    """
    # Compute per-token log-probabilities
    chosen_logps = _get_logp(policy_chosen_logits, chosen_ids, label_pad_token_id)
    rejected_logps = _get_logp(policy_rejected_logits, rejected_ids, label_pad_token_id)

    with torch.no_grad():
        ref_chosen_logps = _get_logp(ref_chosen_logits, chosen_ids, label_pad_token_id)
        ref_rejected_logps = _get_logp(ref_rejected_logits, rejected_ids, label_pad_token_id)

    # DPO reward difference
    chosen_rewards = beta * (chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (rejected_logps - ref_rejected_logps)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss


def _get_logp(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int = -100,
) -> torch.Tensor:
    """Compute per-sequence average log-probability (shifted for next-token prediction)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask padding
    mask = (shift_labels != pad_token_id).float()
    per_token_logps = per_token_logps * mask

    # Average over non-padding tokens
    seq_logps = per_token_logps.sum(-1) / mask.sum(-1).clamp(min=1)
    return seq_logps


def parse_args():
    parser = argparse.ArgumentParser(description="LA-Mo-GMT DPO Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", type=str, nargs="*", default=[])
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(config_path: str, overrides: list[str] = None) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if overrides:
        for ov in overrides:
            keys, value = ov.split("=", 1)
            keys = keys.split(".")
            cfg = config
            for k in keys[:-1]:
                cfg = cfg[k]
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

    logger.info("=" * 60)
    logger.info(f"LA-Mo-GMT DPO Training — Method: {config['mask']['method']}")
    logger.info(f"Model: {config['model_name_or_path']}")
    logger.info(f"Data: {config['data_path']}")
    logger.info("=" * 60)

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load policy model ──
    mc = config["model"]
    model_kwargs = {
        "torch_dtype": getattr(torch, mc.get("torch_dtype", "bfloat16")),
        "use_cache": False,
    }
    if mc.get("use_flash_attention"):
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        **model_kwargs,
    )
    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    # ── Load reference model (8-bit to save memory) ──
    ref_path = config["dpo"].get("reference_model_path", config["model_name_or_path"])
    ref_kwargs = {
        "torch_dtype": getattr(torch, mc.get("torch_dtype", "bfloat16")),
        "use_cache": False,
        "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
    }
    if mc.get("use_flash_attention"):
        ref_kwargs["attn_implementation"] = "flash_attention_2"

    reference_model = AutoModelForCausalLM.from_pretrained(ref_path, **ref_kwargs)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    ref_params = sum(p.numel() for p in reference_model.parameters())
    logger.info(f"Reference model: {ref_params / 1e9:.2f}B params (8-bit, "
                f"~{ref_params / 1e9 * 1:.1f} GB)")

    # ── Load DPO dataset ──
    tc = config["training"]
    train_dataset = load_dpo_dataset(
        dataset_path=config["data_path"],
        tokenizer=tokenizer,
        max_length=tc["max_seq_length"],
        max_prompt_length=tc.get("max_prompt_length", 512),
        split="train",
    )
    logger.info(f"DPO dataset: {len(train_dataset)} examples")

    # ── Training args ──
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
        save_total_limit=3,
        bf16=tc["bf16"],
        optim=tc.get("optim", "adamw_8bit"),
        dataloader_num_workers=tc.get("dataloader_num_workers", 4),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="wandb" if config.get("wandb") else "none",
        seed=args.seed,
    )

    # ── Trainer ──
    mask_cfg = config["mask"]
    collator = DPODataCollator(tokenizer=tokenizer)
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        reference_model=reference_model,
        dpo_beta=config["dpo"].get("beta", 0.1),
        mask_method=mask_cfg["method"],
        mask_global_ratio=mask_cfg["global_ratio"],
        mask_alpha=mask_cfg["alpha"],
        mask_beta1=mask_cfg.get("beta1", 0.9),
        mask_warmup_steps=mask_cfg.get("warmup_steps", 0),
    )

    trainer.train()

    # ── Save ──
    final_path = os.path.join(config["output_dir"], "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    logger.info(f"DPO training complete! Model saved to {final_path}")


if __name__ == "__main__":
    main()
