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
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

LOCAL_PATHS = {
    "mistralai/Mistral-7B-v0.1": "/root/autodl-tmp/model/Mistral-7B-v0.1",
    "deepseek-ai/DeepSeek-Coder-Base-6.7B": "/root/autodl-tmp/model/deepseek-coder-6.7b-base",
    "dataset": "/root/autodl-tmp/dataset/Magicoder-Evol-Instruct-110K"
}


# ═══════════════════════════════════════════════════════════════
# 数据加载（所有算法共用）
# ═══════════════════════════════════════════════════════════════

def load_magicoder_dataset(subset=None):
    logger.info("Loading Magicoder-Evol-Instruct-110K dataset...")
    dataset_path = LOCAL_PATHS["dataset"]
    if os.path.exists(dataset_path):
        dataset = load_dataset(dataset_path, split="train")
    else:
        logger.info(f"Local path not found: {dataset_path}, downloading from Hugging Face...")
        dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
    if subset and subset < len(dataset):
        dataset = dataset.select(range(subset))
        logger.info(f"Dataset subset: {subset} samples")
    else:
        logger.info(f"Dataset loaded with {len(dataset)} samples")
    return dataset


def preprocess_dataset(dataset, tokenizer, max_length=256):
    """预处理数据集：只对 response 部分计算 loss，instruction 和 PAD 忽略"""
    def format_and_tokenize(examples):
        instructions = examples['instruction']
        responses = examples['response']

        # 完整文本
        full_texts = [f"### Instruction:\n{inst}\n\n### Response:\n{resp}"
                      for inst, resp in zip(instructions, responses)]
        tokenized = tokenizer(full_texts, truncation=True, max_length=max_length, padding="max_length")

        # 构建 labels：只计算 response 部分的 loss
        labels = []
        for i, (inst, resp) in enumerate(zip(instructions, responses)):
            # 计算 instruction 前缀的 token 数
            prefix = f"### Instruction:\n{inst}\n\n### Response:\n"
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


def create_dataloader(dataset, batch_size=4):
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


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
    """标准微调 — 无梯度干预"""
    def __init__(self, model, device="cuda", lr=2e-5):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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
            if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                return float('nan')
        return total_loss / count if count > 0 else 0.0


class DropTrainer:
    """随机梯度丢弃"""
    def __init__(self, model, drop_rate=0.1, device="cuda", lr=2e-5):
        self.model = model
        self.drop_rate = drop_rate
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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
    """硬阈值梯度过滤 — 只保留 top_k% 大梯度"""
    def __init__(self, model, top_k=50, device="cuda", lr=2e-5):
        self.model = model
        self.top_k = top_k
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, count = 0.0, 0
        for batch in dataloader:
            inputs = {"input_ids": batch["input_ids"].to(self.device), "attention_mask": batch["attention_mask"].to(self.device)}
            outputs = self.model(**inputs, labels=batch['labels'].to(self.device))
            self.optimizer.zero_grad()
            outputs.loss.backward()

            all_grads, grad_params = [], []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads.append(param.grad.abs().flatten())
                    grad_params.append(param)

            if all_grads:
                all_flat = torch.cat(all_grads)
                k_idx = len(all_flat) - int(len(all_flat) * self.top_k / 100) + 1
                threshold = torch.kthvalue(all_flat, max(1, min(k_idx, len(all_flat)))).values

                # 简化：对每层独立做 threshold
                for param in self.model.parameters():
                    if param.grad is not None:
                        mask = param.grad.abs() >= threshold
                        param.grad = param.grad * mask.float()

            self.optimizer.step()
            total_loss += outputs.loss.item()
            count += 1
        return total_loss / count if count > 0 else 0.0


class RMTTrainer:
    """递归动量训练 — 用 EMA 平滑梯度"""
    def __init__(self, model, momentum=0.9, device="cuda", lr=2e-5):
        self.model = model
        self.momentum = momentum
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.grad_momentum = {}

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, count = 0.0, 0
        for batch in dataloader:
            inputs = {"input_ids": batch["input_ids"].to(self.device), "attention_mask": batch["attention_mask"].to(self.device)}
            outputs = self.model(**inputs, labels=batch['labels'].to(self.device))
            self.optimizer.zero_grad()
            outputs.loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name not in self.grad_momentum:
                        self.grad_momentum[name] = torch.zeros_like(param.grad)
                    self.grad_momentum[name] = self.momentum * self.grad_momentum[name] + (1 - self.momentum) * param.grad
                    param.grad = self.grad_momentum[name]

            self.optimizer.step()
            total_loss += outputs.loss.item()
            count += 1
        return total_loss / count if count > 0 else 0.0


class GMTTrainer:
    """梯度掩码训练 — 累积梯度后按幅度阈值掩码"""
    def __init__(self, model, k_percent=50, accumulation_steps=4, device="cuda", lr=2e-5):
        self.model = model
        self.k_percent = k_percent
        self.accumulation_steps = accumulation_steps
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, count = 0.0, 0
        accumulated_grads, step_count = {}, 0

        for batch in dataloader:
            inputs = {"input_ids": batch["input_ids"].to(self.device), "attention_mask": batch["attention_mask"].to(self.device)}
            outputs = self.model(**inputs, labels=batch['labels'].to(self.device))
            loss = outputs.loss / self.accumulation_steps
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name not in accumulated_grads:
                        accumulated_grads[name] = torch.zeros_like(param.grad)
                    accumulated_grads[name] += param.grad
                    param.grad = None

            total_loss += loss.item() * self.accumulation_steps
            count += 1
            step_count += 1

            if step_count % self.accumulation_steps == 0:
                all_grad_values = []
                for grad in accumulated_grads.values():
                    all_grad_values.append(grad.abs().flatten())

                if all_grad_values:
                    all_flat = torch.cat(all_grad_values)
                    k_idx = len(all_flat) - int(len(all_flat) * self.k_percent / 100) + 1
                    threshold = torch.kthvalue(all_flat, max(1, min(k_idx, len(all_flat)))).values

                    for name, param in self.model.named_parameters():
                        if name in accumulated_grads:
                            param.grad = accumulated_grads[name] / self.accumulation_steps
                            param.grad = param.grad * (param.grad.abs() >= threshold)

                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulated_grads = {}

        return total_loss / count if count > 0 else 0.0


class DGMMTrainer:
    """DGMM — 动态梯度流形掩码（唯一需要 DGMM.py 的算法）"""
    def __init__(self, model, device="cuda", lr=2e-5, dgmm_config=None):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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
                imp_str = f" imp={dgmm_info['avg_importance']:.3f}" if dgmm_info else ""
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


def create_trainer(algorithm, model, args, device):
    """工厂方法：根据算法名创建对应的 Trainer"""
    if algorithm not in TRAINER_MAP:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(TRAINER_MAP.keys())}")

    trainer_cls = TRAINER_MAP[algorithm]
    kwargs = {"model": model, "device": device, "lr": args.lr}

    for param in ALGORITHM_PARAMS.get(algorithm, []):
        kwargs[param] = getattr(args, param)

    if algorithm == "DGMM":
        kwargs["dgmm_config"] = {
            "warmup_steps": args.dgmm_warmup,
            "mask_floor": args.dgmm_mask_floor,
            "encoder_hidden_dim": args.dgmm_encoder_dim,
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
    parser.add_argument("--quantize", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    # 算法特定参数
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--k_percent", type=int, default=80,
                        help="GMT: 保留 top-k% 梯度 (论文默认 50，小数据建议 80)")
    parser.add_argument("--dgmm_warmup", type=int, default=500,
                        help="DGMM: warmup 步数 (默认 500)")
    parser.add_argument("--dgmm_mask_floor", type=float, default=0.2,
                        help="DGMM: 梯度掩码最低比例 (默认 0.2)")
    parser.add_argument("--dgmm_encoder_dim", type=int, default=256,
                        help="DGMM: 梯度编码器隐藏维度 (默认 256)")
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--subset", type=int, default=None,
                        help="只用前 N 条数据（快速验证用，默认全量 110K）")
    args = parser.parse_args()

    if args.output_dir is None:
        safe_name = args.model_name.replace("/", "_")
        args.output_dir = f"checkpoints/{safe_name}_{args.algorithm}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"===== Training: {args.algorithm} on {args.model_name} =====")
    logger.info(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    logger.info(f"Output: {args.output_dir} | DGMM_DISABLED: {os.environ.get('DGMM_DISABLED', '0')}")

    # 1. 加载模型
    logger.info(">>> [1/4] Loading model...")
    model, tokenizer = load_model(args.model_name, device, use_quantization=args.quantize)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 2. 加载数据
    logger.info(">>> [2/4] Loading dataset...")
    dataset = load_magicoder_dataset(subset=args.subset)
    preprocessed = preprocess_dataset(dataset, tokenizer, max_length=args.max_length)
    dataloader = create_dataloader(preprocessed, batch_size=args.batch_size)
    logger.info(f">>> [2/4] Dataloader: {len(dataloader)} batches")

    # 3. 训练
    trainer = create_trainer(args.algorithm, model, args, device)
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
