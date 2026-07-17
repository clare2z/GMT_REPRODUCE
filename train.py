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
import shutil
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


def _save_checkpoint_safe(model, tokenizer, output_dir, global_step, args, loss, epoch):
    """在 optimizer.step() 后立即保存中间 checkpoint"""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    state = {
        "algorithm": args.algorithm, "model_name": args.model_name,
        "global_step": global_step, "epoch": epoch, "current_loss": loss,
        "learning_rate": args.lr, "save_steps": args.save_steps,
        "lora": args.lora, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "ddgmm_config": getattr(args, 'dgmm_ablate', ''),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(ckpt_dir, "train_state.json"), "w") as f:
        json.dump(state, f, indent=2)
    logger.info(f">>> Saved intermediate checkpoint: {ckpt_dir}")
    # 删除旧 checkpoint
    if args.save_total_limit > 0:
        all_dirs = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint-")])
        while len(all_dirs) > args.save_total_limit:
            old = os.path.join(output_dir, all_dirs.pop(0))
            shutil.rmtree(old, ignore_errors=True)
            logger.info(f">>> Removed old checkpoint: {old}")


LOCAL_PATHS = {
    "mistralai/Mistral-7B-v0.1": "/root/autodl-tmp/model/Mistral-7B-v0.1",
    "deepseek-ai/DeepSeek-Coder-Base-6.7B": "/root/autodl-tmp/model/deepseek-coder-6.7b-base",
    "LLM-Research/llama-2-7b": "/root/autodl-tmp/model/Llama-2-7b",
    "dataset": "/root/autodl-tmp/dataset/Magicoder-Evol-Instruct-110K",
    "tulu": "/root/autodl-tmp/dataset/tulu-v2-sft-mixture"
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


def load_tulu_dataset():
    logger.info("Loading Tulu V2 SFT Mixture dataset...")
    tulu_path = LOCAL_PATHS.get("tulu", "")
    if os.path.exists(tulu_path):
        from datasets import load_from_disk
        dataset = load_from_disk(tulu_path)
        if isinstance(dataset, dict):
            dataset = dataset.get("train", list(dataset.values())[0])
    else:
        logger.info(f"Local path not found: {tulu_path}, fallback HF...")
        dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    logger.info(f"Tulu dataset loaded with {len(dataset)} samples")
    return dataset


def preprocess_tulu_dataset(dataset, tokenizer, max_length=2048):
    """预处理 Tulu: 逐消息 tokenize 后拼接 input_ids/labels/attention_mask"""
    def process_one(messages):
        input_ids, labels = [], []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if role == "user":
                prefix = f"### Instruction:\n{content}\n\n### Response:\n"
                p_ids = tokenizer(prefix, add_special_tokens=(len(input_ids) == 0))["input_ids"]
                input_ids.extend(p_ids)
                labels.extend([-100] * len(p_ids))
            elif role == "assistant":
                resp_text = f"{content}\n"
                r_ids = tokenizer(resp_text, add_special_tokens=False)["input_ids"]
                input_ids.extend(r_ids)
                labels.extend(r_ids)

        # 加 EOS
        eos_id = tokenizer.eos_token_id
        input_ids.append(eos_id)
        labels.append(eos_id)

        # 截断或填充到 max_length
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        attn_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(tokenizer.pad_token_id or 0)
            labels.append(-100)
            attn_mask.append(0)

        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

    def mapper(examples):
        results = {"input_ids": [], "attention_mask": [], "labels": []}
        for messages in examples["messages"]:
            row = process_one(messages)
            for k in results:
                results[k].append(row[k])
        return results

    dataset = dataset.map(mapper, batched=True, remove_columns=dataset.column_names)
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

    def train_epoch(self, dataloader, save_fn=None):
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
                if save_fn: save_fn(outputs.loss.item())
            if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                return (float('nan'), count)
        opt = math.ceil(count / self.grad_accum)
        return (total_loss / count, opt) if count > 0 else (0.0, 0)


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

    def train_epoch(self, dataloader, save_fn=None):
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
            if save_fn: save_fn()
            total_loss += outputs.loss.item()
            count += 1
        return (total_loss / count, count) if count > 0 else (0.0, 0)


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

    def train_epoch(self, dataloader, save_fn=None):
        self.model.train()
        total_loss, count = 0.0, 0
        for batch in dataloader:
            inputs = {"input_ids": batch["input_ids"].to(self.device), "attention_mask": batch["attention_mask"].to(self.device)}
            outputs = self.model(**inputs, labels=batch['labels'].to(self.device))
            self.optimizer.zero_grad()
            outputs.loss.backward()
            self.optimizer.step()
            if save_fn: save_fn()
            total_loss += outputs.loss.item()
            count += 1
        return (total_loss / count, count) if count > 0 else (0.0, 0)


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

    def train_epoch(self, dataloader, save_fn=None):
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
            if save_fn: save_fn()
            total_loss += outputs.loss.item()
            count += 1
        return (total_loss / count, count) if count > 0 else (0.0, 0)


def _estimate_global_threshold(named_params, keep_ratio, max_samples=5_000_000):
    """随机采样估计全局梯度top-k阈值，避免OOM"""
    active = [(name, p) for name, p in named_params if p.requires_grad and p.grad is not None]
    total_elements = sum(p.grad.numel() for _, p in active)
    if total_elements == 0:
        return None
    sample_ratio = min(1.0, max_samples / total_elements)
    samples = []
    for _, p in active:
        grad_abs = p.grad.detach().abs().flatten()
        if sample_ratio < 1.0:
            n_sample = max(100, int(grad_abs.numel() * sample_ratio))
            n_sample = min(n_sample, grad_abs.numel())
            idx = torch.randint(0, grad_abs.numel(), (n_sample,), device=grad_abs.device)
            sampled = grad_abs[idx]
        else:
            sampled = grad_abs
        samples.append(sampled.float())
    sampled_values = torch.cat(samples)
    mask_ratio = 1.0 - keep_ratio
    kth = max(1, int(sampled_values.numel() * mask_ratio))
    kth = min(kth, sampled_values.numel())
    threshold = torch.kthvalue(sampled_values, kth).values.item()
    del sampled_values, samples
    return threshold


class GMTTrainer:
    """梯度掩码训练 — 每步全局 top-k 幅度阈值（采样估计避免OOM）"""
    def __init__(self, model, k_percent=80, accumulation_steps=1, device="cuda", lr=2e-5,
                 weight_decay=0.0, grad_accum=1, warmup_ratio=0.03, total_steps=5000,
                 lr_scheduler_type="cosine"):
        self.model = model
        self.k_percent = k_percent
        self.device = device
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.keep = k_percent / 100.0

    def train_epoch(self, dataloader, save_fn=None):
        self.model.train()
        total_loss, count = 0.0, 0

        for batch in dataloader:
            inputs = {"input_ids": batch["input_ids"].to(self.device), "attention_mask": batch["attention_mask"].to(self.device)}
            outputs = self.model(**inputs, labels=batch['labels'].to(self.device))
            self.optimizer.zero_grad()
            outputs.loss.backward()

            if self.keep < 1.0:
                named_params = list(self.model.named_parameters())
                thr = _estimate_global_threshold(named_params, keep_ratio=self.keep, max_samples=5_000_000)
                if thr is not None:
                    for _, param in named_params:
                        if param.grad is not None:
                            mask = param.grad.detach().abs() >= thr
                            param.grad.mul_(mask.to(dtype=param.grad.dtype, device=param.grad.device))

            self.optimizer.step()
            total_loss += outputs.loss.item()
            count += 1

            if count % 500 == 0 and self.keep < 1.0:
                ke = float(sum((p.grad != 0).float().mean().item() for _, p in self.model.named_parameters() if p.requires_grad and p.grad is not None) / max(1, sum(1 for _, p in self.model.named_parameters() if p.requires_grad and p.grad is not None)))
                logger.info(f"  [GMT] step {count} | actual_keep~{ke:.3f} target={self.keep:.3f}")

        return (total_loss / count, count) if count > 0 else (0.0, 0)


class DGMMTrainer:
    """DGMM — 动态梯度流形掩码（唯一需要 DGMM.py 的算法）"""
    def __init__(self, model, device="cuda", lr=2e-5, dgmm_config=None,
                 weight_decay=0.0, grad_accum=1, warmup_ratio=0.03, total_steps=5000,
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
        else:
            from transformers import get_cosine_schedule_with_warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=num_warmup, num_training_steps=num_updates)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from DGMM import DGMMFramework
        self.dgmm = DGMMFramework(device=device, **(dgmm_config or {}))
        self.best_loss = float('inf')

    def train_epoch(self, dataloader, save_fn=None):
        self.model.train()
        total_loss, micro_step, update_step = 0.0, 0, 0
        total_batches = len(dataloader)
        t_start = time.time()

        self.optimizer.zero_grad()
        for batch in dataloader:
            inputs = {"input_ids": batch["input_ids"].to(self.device), "attention_mask": batch["attention_mask"].to(self.device)}
            labels = batch['labels'].to(self.device)
            outputs = self.model(**inputs, labels=labels)

            if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                logger.error(f"NaN/Inf loss at micro_step {micro_step}")
                return float('nan')

            loss = outputs.loss / self.grad_accum
            loss.backward()
            total_loss += outputs.loss.item()
            micro_step += 1

            should_update = (micro_step % self.grad_accum == 0) or (micro_step == total_batches)

            if should_update:
                accumulated_grads = {
                    name: param.grad.detach().clone()
                    for name, param in self.model.named_parameters()
                    if param.grad is not None
                }

                dgmm_info = None
                if accumulated_grads:
                    masked_grads, dgmm_info = self.dgmm.apply_mask(accumulated_grads)
                    for name, param in self.model.named_parameters():
                        if name in masked_grads:
                            param.grad.copy_(masked_grads[name].to(dtype=param.grad.dtype, device=param.grad.device))
                    del accumulated_grads, masked_grads

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                if save_fn: save_fn(outputs.loss.item())
                update_step += 1

                if update_step % 10 == 0 or update_step == 1:
                    elapsed = time.time() - t_start
                    eta = (elapsed / micro_step) * (total_batches - micro_step)
                    imp_str = ""
                    if dgmm_info:
                        imp_str = (f" imp={dgmm_info.get('avg_importance', 0):.3f}"
                                   f" mk={dgmm_info.get('mask_keep_mean', 0):.3f}"
                                   f" tgt=[{dgmm_info.get('target_keep_min', 0):.2f},{dgmm_info.get('target_keep_max', 0):.2f}]")
                    logger.info(f"  {update_step}/{total_batches//self.grad_accum} | loss={outputs.loss.item():.4f} "
                                f"| eta={eta:.0f}s | micro={micro_step}{imp_str}")

        return (total_loss / micro_step, update_step) if micro_step > 0 else (0.0, 0)


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
            "keep_update_interval": args.dgmm_keep_interval,
        }

    return trainer_cls(**kwargs)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument("--algorithm", type=str, required=True,
                        choices=list(TRAINER_MAP.keys()), help="Training algorithm")
    parser.add_argument("--dataset", type=str, default="magicoder",
                        choices=["magicoder", "tulu"], help="训练数据集")
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
    parser.add_argument("--dgmm_keep_interval", type=int, default=20,
                        help="DGMM: 每 N 步更新 layer statistics (默认 20)")
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
    parser.add_argument("--skip_save", action="store_true", default=False,
                        help="跳过保存checkpoint（sanity用）")
    parser.add_argument("--save_steps", type=int, default=0,
                        help="中间checkpoint保存间隔步数（0=不保存）")
    parser.add_argument("--save_total_limit", type=int, default=0,
                        help="最多保留的中间checkpoint数量（0=不删除旧checkpoint）")
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
    logger.info(f">>> [2/4] Loading dataset: {args.dataset}")
    if args.dataset == "tulu":
        dataset = load_tulu_dataset()
        preprocessed = preprocess_tulu_dataset(dataset, tokenizer, max_length=args.max_length)
    else:
        dataset = load_magicoder_dataset(subset=args.subset, no_shuffle=args.no_shuffle, seed=args.seed)
        preprocessed = preprocess_dataset(dataset, tokenizer, max_length=args.max_length)
    dataloader = create_dataloader(preprocessed, batch_size=args.batch_size, seed=args.seed)
    logger.info(f">>> [2/4] Dataloader: {len(dataloader)} batches")

    # 3. 训练
    trainer = create_trainer(args.algorithm, model, args, device, total_steps=len(dataloader) * args.epochs)
    os.makedirs(args.output_dir, exist_ok=True)
    log_rows = []
    global_step = 0

    for epoch in range(args.epochs):
        t_start = time.time()
        logger.info(f">>> [3/4] Epoch {epoch+1}/{args.epochs} starting...")
        # 闭包：每次 optimizer.step() 后立即保存
        _save_step = [0]  # mutable counter
        def _save_fn(loss_val=0.0):
            _save_step[0] += 1
            s = _save_step[0]
            if s % args.save_steps == 0:
                _save_checkpoint_safe(model, tokenizer, args.output_dir, s, args, loss_val, epoch + 1)
        result = trainer.train_epoch(dataloader, save_fn=_save_fn if args.save_steps > 0 else None)
        if isinstance(result, tuple):
            loss, opt_steps = result
        else:
            loss, opt_steps = result, len(dataloader)
        elapsed = time.time() - t_start
        global_step += opt_steps
        logger.info(f">>> [3/4] Epoch {epoch+1}/{args.epochs} done | loss={loss:.4f} | time={elapsed:.0f}s | opt_steps={opt_steps} global={global_step}")
        log_rows.append({"epoch": epoch + 1, "loss": loss, "time_s": elapsed})

        if loss != loss:
            logger.error("Training diverged! Aborting.")
            break

    # 4. 保存 checkpoint
    if args.skip_save:
        logger.info(">>> [4/4] Skipped saving (--skip_save)")
    else:
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
