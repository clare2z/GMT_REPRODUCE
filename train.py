"""
统一训练脚本 — 支持所有算法，训练完保存 checkpoint，评测用 eval.py

用法:
    python train.py --algorithm DGMM --model_name mistralai/Mistral-7B-v0.1 --epochs 3 --output_dir checkpoints/dgmm_v1
    python train.py --algorithm SFT  --model_name mistralai/Mistral-7B-v0.1 --epochs 3 --output_dir checkpoints/sft_v1
    python train.py --algorithm Drop --drop_rate 0.1 --epochs 3
    python train.py --algorithm HFT  --top_k 50 --epochs 3
    python train.py --algorithm RMT  --momentum 0.9 --epochs 3
    python train.py --algorithm GMT  --k_percent 50 --accumulation_steps 4 --epochs 3

    DGMM_DISABLED=1 python train.py --algorithm DGMM  # baseline 模式
"""

import os
import sys

if os.environ.get("HF_ENDPOINT") is None:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if os.environ.get("HF_HOME") is None:
    os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"

import torch
import logging
import csv
import time
import json
import math
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


LOCAL_PATHS = {
    "mistralai/Mistral-7B-v0.1": "/root/autodl-tmp/model/Mistral-7B-v0.1",
    "deepseek-ai/DeepSeek-Coder-Base-6.7B": "/root/autodl-tmp/model/deepseek-coder-6.7b-base",
    "dataset": "/root/autodl-tmp/dataset/Magicoder-Evol-Instruct-110K"
}


# ═══════════════════════════════════════════════════════════════
# 数据加载（所有算法共用）
# ═══════════════════════════════════════════════════════════════

def load_magicoder_dataset(subset=None, no_shuffle=False, seed=42):
    logger.info("Loading Magicoder-Evol-Instruct-110K dataset...")
    dataset_path = LOCAL_PATHS["dataset"]
    if os.path.exists(dataset_path):
        dataset = load_dataset(dataset_path, split="train")
    else:
        logger.info(f"Local path not found: {dataset_path}, downloading from Hugging Face...")
        dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
    if subset and subset < len(dataset):
        if no_shuffle:
            dataset = dataset.select(range(subset))
            logger.info(f"Dataset subset: {subset} samples (first-N, no shuffle)")
        else:
            dataset = dataset.shuffle(seed=seed).select(range(subset))
            logger.info(f"Dataset subset: {subset} samples (shuffled seed=42)")
    else:
        logger.info(f"Dataset loaded with {len(dataset)} samples")
    return dataset


def preprocess_dataset(dataset, tokenizer, max_length=256):
    """预处理数据集：只对 response 部分计算 loss，instruction 和 PAD 忽略"""
    def format_and_tokenize(examples):
        instructions = examples['instruction']
        responses = examples['response']

        # 完整文本（strip 去首尾空白，加 EOS 防模型学会输出换行）
        full_texts = [f"### Instruction:\n{inst.strip()}\n\n### Response:\n{resp.strip()}{tokenizer.eos_token}"
                      for inst, resp in zip(instructions, responses)]
        tokenized = tokenizer(full_texts, truncation=True, max_length=max_length, padding="max_length")

        # 构建 labels：只计算 response 部分的 loss
        labels = []
        for i, (inst, resp) in enumerate(zip(instructions, responses)):
            # 计算 instruction 前缀的 token 数
            prefix = f"### Instruction:\n{inst.strip()}\n\n### Response:\n"
            prefix_tokens = tokenizer(prefix, truncation=True, max_length=max_length)["input_ids"]
            # 去掉 BOS token（tokenizer 自动加的），用实际 token 数对齐
            prefix_len = len(prefix_tokens) - 1 if tokenizer.bos_token_id is not None else len(prefix_tokens)

            input_ids = tokenized["input_ids"][i]
            label = [-100] * len(input_ids)
            for j in range(prefix_len, len(input_ids)):
                if input_ids[j] != tokenizer.pad_token_id:
                    label[j] = input_ids[j]
            labels.append(label)

        tokenized["labels"] = labels
        return tokenized

    dataset = dataset.map(format_and_tokenize, batched=True, remove_columns=["instruction", "response"])
    return dataset


def create_dataloader(dataset, batch_size=4, seed=42):
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)


def load_model(model_name, device="cuda", use_quantization=False):
    logger.info(f"Loading model: {model_name}")
    local_path = LOCAL_PATHS.get(model_name)
    model_path = local_path if (local_path and os.path.exists(local_path)) else model_name
    logger.info(f"Model path: {model_path}")

    quantization_config = None
    if use_quantization and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=quantization_config,
        device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════
# Trainer 类（每个算法一个）
# ═══════════════════════════════════════════════════════════════

class SFTTrainer:
    """标准微调 — 支持 gradient accumulation + cosine scheduler + warmup"""
    def __init__(self, model, device="cuda", lr=2e-5, weight_decay=0.0,
                 grad_accum=1, warmup_ratio=0.03, total_steps=5000,
                 lr_scheduler_type="cosine"):
        self.model = model
        self.device = device
        self.grad_accum = grad_accum
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        num_updates = max(1, math.ceil(total_steps / grad_accum))
        num_warmup = max(1, int(num_updates * warmup_ratio))
        if warmup_ratio <= 0 or lr_scheduler_type == "constant":
            from torch.optim.lr_scheduler import LambdaLR
            self.scheduler = LambdaLR(self.optimizer, lambda step: 1.0)
            logger.info("  [SFT] constant lr (no scheduler)")
        else:
            from transformers import get_cosine_schedule_with_warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=num_warmup, num_training_steps=num_updates)
            logger.info(f"  [SFT] cosine scheduler: warmup={num_warmup}/{num_updates}")

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, count, update_count = 0.0, 0, 0
        for batch in dataloader:
            inputs = {"input_ids": batch["input_ids"].to(self.device), "attention_mask": batch["attention_mask"].to(self.device)}
            outputs = self.model(**inputs, labels=batch['labels'].to(self.device))
            loss = outputs.loss / self.grad_accum
            loss.backward()
            total_loss += outputs.loss.item()
            count += 1

            if count % self.grad_accum == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                update_count += 1
            if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                return float('nan')
        return total_loss / count if count > 0 else 0.0


class DropTrainer:
    """随机梯度丢弃"""
    def __init__(self, model, drop_rate=0.1, device="cuda", lr=2e-5,
                 weight_decay=0.0, grad_accum=1, warmup_ratio=0.03, total_steps=5000,
                 lr_scheduler_type="cosine"):
        self.model = model
        self.drop_rate = drop_rate
        self.device = device
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, count = 0.0, 0
        for batch in dataloader:
            inputs = {"input_ids": batch["input_ids"].to(self.device), "attention_mask": batch["attention_mask"].to(self.device)}
            outputs = self.model(**inputs, labels=batch['labels'].to(self.device))
            self.optimizer.zero_grad()
            outputs.loss.backward()
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = param.grad * (torch.rand_like(param.grad) > self.drop_rate)
            self.optimizer.step()
            total_loss += outputs.loss.item()
            count += 1
        return total_loss / count if count > 0 else 0.0


class HFTTrainer:
    """Half Fine-Tuning — 随机冻结一半可训练参数，只训练另一半"""
    def __init__(self, model, top_k=50, device="cuda", lr=2e-5,
                 weight_decay=0.0, grad_accum=1, warmup_ratio=0.03, total_steps=5000,
                 lr_scheduler_type="cosine"):
        self.model = model
        self.device = device
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        n = len(trainable_params)
        mask = torch.rand(n) < (top_k / 100.0)
        for i, p in enumerate(trainable_params):
            if not mask[i].item():
                p.requires_grad = False
        frozen = sum(1 for p in trainable_params if not p.requires_grad)
        logger.info(f"  [HFT] Frozen {frozen}/{n} trainable params, training {n-frozen}")
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, count = 0.0, 0
        for batch in dataloader:
            inputs = {"input_ids": batch["input_ids"].to(self.device), "attention_mask": batch["attention_mask"].to(self.device)}
            outputs = self.model(**inputs, labels=batch['labels'].to(self.device))
            self.optimizer.zero_grad()
            outputs.loss.backward()
            self.optimizer.step()
            total_loss += outputs.loss.item()
            count += 1
        return total_loss / count if count > 0 else 0.0


class RMTTrainer:
    """Random Mask Tuning — 随机保留 k% 梯度"""
    def __init__(self, model, momentum=0.9, device="cuda", lr=2e-5,
                 weight_decay=0.0, grad_accum=1, warmup_ratio=0.03, total_steps=5000,
                 lr_scheduler_type="cosine"):
        self.model = model
        self.keep = momentum
        self.device = device
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, count = 0.0, 0
        for batch in dataloader:
            inputs = {"input_ids": batch["input_ids"].to(self.device), "attention_mask": batch["attention_mask"].to(self.device)}
            outputs = self.model(**inputs, labels=batch['labels'].to(self.device))
            self.optimizer.zero_grad()
            outputs.loss.backward()

            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    mask = torch.rand_like(param.grad) < self.keep
                    param.grad = param.grad * mask.float()

            self.optimizer.step()
            total_loss += outputs.loss.item()
            count += 1
        return total_loss / count if count > 0 else 0.0


class GMTTrainer:
    """梯度掩码训练 — 每步全局 top-k 幅度阈值"""
    def __init__(self, model, k_percent=80, accumulation_steps=1, device="cuda", lr=2e-5,
                 weight_decay=0.0, grad_accum=1, warmup_ratio=0.03, total_steps=5000,
                 lr_scheduler_type="cosine"):
        self.model = model
        self.k_percent = k_percent
        self.device = device
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.keep = k_percent / 100.0

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, count = 0.0, 0

        for batch in dataloader:
            inputs = {"input_ids": batch["input_ids"].to(self.device), "attention_mask": batch["attention_mask"].to(self.device)}
            outputs = self.model(**inputs, labels=batch['labels'].to(self.device))
            self.optimizer.zero_grad()
            outputs.loss.backward()

            # GMT mask: keep top keep% trainable gradients by magnitude
            if self.keep < 1.0:
                trainable_grads = [p.grad.detach().abs().flatten() for p in self.model.parameters()
                                   if p.requires_grad and p.grad is not None]
                all_flat = torch.cat(trainable_grads)
                num_keep = max(1, int(all_flat.numel() * self.keep))
                thr = float(torch.topk(all_flat, num_keep, largest=True).values.min().item())

                kept_elems, total_elems, n_masked = 0, 0, 0
                for param in self.model.parameters():
                    if param.requires_grad and param.grad is not None:
                        mask = param.grad.abs() >= thr
                        param.grad = param.grad * mask.float()
                        kept_elems += mask.sum().item()
                        total_elems += mask.numel()
                        n_masked += 1

            self.optimizer.step()
            total_loss += outputs.loss.item()
            count += 1

            if self.keep < 1.0 and count == 1:
                logger.info(f"  [GMT] masking {n_masked} trainable params | actual_keep_global={kept_elems/max(total_elems,1):.4f} target={self.keep:.4f}")
            if count % 50 == 0 and self.keep < 1.0:
                actual_keep_global = kept_elems / max(total_elems, 1)
                logger.info(f"  [GMT] step {count} | actual_keep_global={actual_keep_global:.4f} (target={self.keep:.4f})")

        return total_loss / count if count > 0 else 0.0


class DGMMTrainer:
    """DGMM — 动态梯度流形掩码（唯一需要 DGMM.py 的算法）"""
    def __init__(self, model, device="cuda", lr=2e-5, dgmm_config=None,
                 weight_decay=0.0, grad_accum=1, warmup_ratio=0.03, total_steps=5000,
                 lr_scheduler_type="cosine"):
        self.model = model
        self.device = device
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from DGMM import DGMMFramework
        self.dgmm = DGMMFramework(device=device, **(dgmm_config or {}))
        self.best_loss = float('inf')

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, count = 0.0, 0
        total_batches = len(dataloader)
        t_start = time.time()

        for batch in dataloader:
            inputs = {"input_ids": batch["input_ids"].to(self.device), "attention_mask": batch["attention_mask"].to(self.device)}
            labels = batch['labels'].to(self.device)
            outputs = self.model(**inputs, labels=labels)

            if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                logger.error(f"NaN/Inf loss at step {count}")
                return float('nan')

            self.optimizer.zero_grad()
            outputs.loss.backward()

            accumulated_grads = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    accumulated_grads[name] = param.grad.clone().detach()

            dgmm_info = None
            if accumulated_grads:
                masked_grads, dgmm_info = self.dgmm.apply_mask(accumulated_grads)
                for name, param in self.model.named_parameters():
                    if name in masked_grads:
                        param.grad = masked_grads[name]
                del accumulated_grads, masked_grads

            self.optimizer.step()
            torch.cuda.empty_cache()
            total_loss += outputs.loss.item()
            count += 1

            if count % 10 == 0 or count == 1:
                elapsed = time.time() - t_start
                eta = (elapsed / count) * (total_batches - count)
                imp_str = ""
                if dgmm_info:
                    imp_str = (f" imp={dgmm_info.get('avg_importance', 0):.3f}"
                               f" mk={dgmm_info.get('mask_keep_mean', 0):.3f}"
                               f" tgt=[{dgmm_info.get('target_keep_min', 0):.2f},{dgmm_info.get('target_keep_max', 0):.2f}]"
                               f" lo={dgmm_info.get('lowest_layers', '?')}"
                               f" hi={dgmm_info.get('highest_layers', '?')}")
                self.best_loss = min(self.best_loss, outputs.loss.item())
                status = "✅" if outputs.loss.item() <= self.best_loss else "  "
                logger.info(f"  {status} {count}/{total_batches} | loss={outputs.loss.item():.4f} | eta={eta:.0f}s{imp_str}")

        return total_loss / count if count > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# Trainer 工厂
# ═══════════════════════════════════════════════════════════════

TRAINER_MAP = {
    "SFT": SFTTrainer,
    "Drop": DropTrainer,
    "HFT": HFTTrainer,
    "RMT": RMTTrainer,
    "GMT": GMTTrainer,
    "DGMM": DGMMTrainer,
}

ALGORITHM_PARAMS = {
    "SFT": [],
    "Drop": ["drop_rate"],
    "HFT": ["top_k"],
    "RMT": ["momentum"],
    "GMT": ["k_percent", "accumulation_steps"],
    "DGMM": [],
}


def create_trainer(algorithm, model, args, device, total_steps=5000):
    """工厂方法：根据算法名创建对应的 Trainer"""

    # GMT k=100 → 直接用 SFT, 保证完全等价
    if algorithm == "GMT" and args.k_percent >= 100:
        logger.info("GMT k=100 → using SFTTrainer (guaranteed equivalence)")
        return SFTTrainer(model, device=device, lr=args.lr, weight_decay=args.weight_decay,
                          grad_accum=args.gradient_accumulation_steps,
                          warmup_ratio=args.warmup_ratio, total_steps=total_steps,
                          lr_scheduler_type=args.lr_scheduler_type)

    if algorithm not in TRAINER_MAP:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(TRAINER_MAP.keys())}")

    trainer_cls = TRAINER_MAP[algorithm]
    kwargs = {"model": model, "device": device, "lr": args.lr}

    # Paper-style params for all trainers
    kwargs["weight_decay"] = args.weight_decay
    kwargs["grad_accum"] = args.gradient_accumulation_steps
    kwargs["warmup_ratio"] = args.warmup_ratio
    kwargs["total_steps"] = total_steps
    kwargs["lr_scheduler_type"] = args.lr_scheduler_type

    for param in ALGORITHM_PARAMS.get(algorithm, []):
        kwargs[param] = getattr(args, param)

    if algorithm == "DGMM":
        kwargs["dgmm_config"] = {
            "warmup_steps": args.dgmm_warmup,
            "mask_floor": args.dgmm_mask_floor,
            "encoder_hidden_dim": args.dgmm_encoder_dim,
            "ablate": args.dgmm_ablate,
            "soft_alpha": args.dgmm_soft_alpha,
            "late_start": args.dgmm_late_start,
        }

    return trainer_cls(**kwargs)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument("--algorithm", type=str, required=True,
                        choices=list(TRAINER_MAP.keys()), help="Training algorithm")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Checkpoint output dir (default: checkpoints/{model}_{algorithm})")
    parser.add_argument("--quantize", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    # 算法特定参数
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="RMT: 随机保留比例 (0-1)")
    parser.add_argument("--k_percent", type=int, default=80,
                        help="GMT: 保留 top-k% 梯度 (等价 keep_pct, 92→keep 92%)")
    parser.add_argument("--dgmm_warmup", type=int, default=500,
                        help="DGMM: warmup 步数 (默认 500)")
    parser.add_argument("--dgmm_mask_floor", type=float, default=0.2,
                        help="DGMM: 梯度掩码最低比例 (默认 0.2)")
    parser.add_argument("--dgmm_soft_alpha", type=float, default=0.0,
                        help="DGMM: soft scaling alpha (0=hard zero, 0.5=缩到50%)")
    parser.add_argument("--dgmm_late_start", type=int, default=0,
                        help="DGMM: 延迟到 N 步后启动 mask (0=立即)")
    parser.add_argument("--dgmm_ablate", type=str, default="",
                        help="消融: direction,volatility,synergy (逗号分隔)")
    parser.add_argument("--dgmm_encoder_dim", type=int, default=256,
                        help="DGMM: 梯度编码器隐藏维度 (默认 256)")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--lora", action="store_true", default=False,
                        help="使用 LoRA 全精度微调（梯度信号更干净）")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (默认 16)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (默认 32)")
    parser.add_argument("--subset", type=int, default=None,
                        help="只用前 N 条数据（快速验证用，默认全量 110K）")
    parser.add_argument("--seed", type=int, default=42,
                        help="全局随机种子 (默认 42)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数 (默认 1)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="warmup 比例 (默认 0.03)")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="lr scheduler 类型 (默认 cosine)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="weight decay (论文 0)")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="使用 bf16 训练")
    parser.add_argument("--no_shuffle", action="store_true", default=False,
                        help="不 shuffle，取前 N 条（复现旧结果用）")
    args = parser.parse_args()
    set_seed(args.seed)

    if args.output_dir is None:
        safe_name = args.model_name.replace("/", "_")
        args.output_dir = f"checkpoints/{safe_name}_{args.algorithm}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"===== Training: {args.algorithm} on {args.model_name} =====")
    logger.info(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    logger.info(f"Output: {args.output_dir} | DGMM_DISABLED: {os.environ.get('DGMM_DISABLED', '0')}")

    # 1. 加载模型
    logger.info(">>> [1/4] Loading model...")
    # LoRA 下不用 4-bit，需要全精度梯度
    use_quant = args.quantize and not args.lora
    model, tokenizer = load_model(args.model_name, device, use_quantization=use_quant)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {all_params:,} ({100*trainable/all_params:.1f}%)")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.lora:
        logger.info(f">>> LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 2. 加载数据
    logger.info(">>> [2/4] Loading dataset...")
    dataset = load_magicoder_dataset(subset=args.subset, no_shuffle=args.no_shuffle, seed=args.seed)
    preprocessed = preprocess_dataset(dataset, tokenizer, max_length=args.max_length)
    dataloader = create_dataloader(preprocessed, batch_size=args.batch_size, seed=args.seed)
    logger.info(f">>> [2/4] Dataloader: {len(dataloader)} batches")

    # 3. 训练
    trainer = create_trainer(args.algorithm, model, args, device, total_steps=len(dataloader) * args.epochs)
    os.makedirs(args.output_dir, exist_ok=True)
    log_rows = []

    for epoch in range(args.epochs):
        t_start = time.time()
        logger.info(f">>> [3/4] Epoch {epoch+1}/{args.epochs} starting...")
        loss = trainer.train_epoch(dataloader)
        elapsed = time.time() - t_start
        logger.info(f">>> [3/4] Epoch {epoch+1}/{args.epochs} done | loss={loss:.4f} | time={elapsed:.0f}s")
        log_rows.append({"epoch": epoch + 1, "loss": loss, "time_s": elapsed})
        if loss != loss:  # NaN check
            logger.error("Training diverged! Aborting.")
            break

    # 4. 保存 checkpoint
    logger.info(f">>> [4/4] Saving checkpoint to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = {
        "algorithm": args.algorithm, "model_name": args.model_name,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "max_length": args.max_length,
        "seed": args.seed, "subset": args.subset, "no_shuffle": args.no_shuffle,
        "lora": args.lora, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "quantize": args.quantize, "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio, "lr_scheduler_type": args.lr_scheduler_type,
        "weight_decay": args.weight_decay, "bf16": args.bf16,
        "prompt_format": "### Instruction:\n{instruction}\n\n### Response:\n{response}",
        "dgmm_disabled": os.environ.get("DGMM_DISABLED", "0"),
        "timestamp": datetime.now().isoformat(),
    }
    for param in ALGORITHM_PARAMS.get(args.algorithm, []):
        config[param] = getattr(args, param)
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    log_path = os.path.join(args.output_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "time_s"])
        writer.writeheader()
        writer.writerows(log_rows)

    logger.info(f"===== Done! Checkpoint: {args.output_dir} =====")
    del model; del tokenizer; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
