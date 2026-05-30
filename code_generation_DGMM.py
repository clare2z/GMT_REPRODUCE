import os

# ⚠️ 必须在 import transformers / datasets 之前设置，否则 huggingface_hub 已用默认地址初始化
if os.environ.get("HF_ENDPOINT") is None:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import csv
from datetime import datetime
from typing import Dict, Tuple
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 本地路径配置
LOCAL_PATHS = {
    "mistralai/Mistral-7B-v0.1": "/root/autodl-tmp/model/Mistral-7B-v0___1",
    "deepseek-ai/DeepSeek-Coder-Base-6.7B": "/root/autodl-tmp/model/deepseek-coder-6.7b-base",
    "dataset": "/root/autodl-tmp/dataset/Magicoder-Evol-Instruct-110K"
}


class GradientEncoder(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.norm(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.normalize(x, dim=-1)


class ContrastiveLearner(nn.Module):
    def __init__(self, encoder_dim: int = 64, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim)
        )

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        anchors = self.projection_head(anchors)
        positives = self.projection_head(positives)
        negatives = self.projection_head(negatives)

        pos_sim = torch.sum(anchors * positives, dim=-1) / self.temperature
        neg_sim = torch.mm(anchors, negatives.t()) / self.temperature

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(anchors.size(0), dtype=torch.long, device=anchors.device)
        return F.cross_entropy(logits, labels)


class LayerAttentionFusion(nn.Module):
    def __init__(self, feature_dim: int = 64, num_layers: int = 12):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, layer_features: torch.Tensor) -> torch.Tensor:
        layer_features = layer_features.unsqueeze(0)
        queries = self.query_proj(layer_features)
        keys = self.key_proj(layer_features)
        values = self.value_proj(layer_features)

        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(layer_features.size(-1))
        attn_weights = F.softmax(attn_scores, dim=-1)

        fused = torch.bmm(attn_weights, values)
        fused = self.output_proj(fused)

        return fused.squeeze(0)


class DGMMFramework:
    def __init__(
        self,
        encoder_hidden_dim: int = 128,
        encoder_output_dim: int = 64,
        contrastive_temperature: float = 0.5,
        contrastive_weight: float = 0.1,
        consistency_weight: float = 0.2,
        ema_alpha: float = 0.9,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        grad_history_window: int = 5
    ):
        self.device = device
        self.dtype = dtype
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight
        self.ema_alpha = ema_alpha
        self.grad_history_window = grad_history_window

        self.gradient_encoder = GradientEncoder(
            input_dim=encoder_hidden_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim
        ).to(device).to(dtype)

        self.contrastive_learner = ContrastiveLearner(
            encoder_dim=encoder_output_dim,
            temperature=contrastive_temperature
        ).to(device).to(dtype)

        self.layer_attention = LayerAttentionFusion(
            feature_dim=encoder_output_dim,
            num_layers=12
        ).to(device).to(dtype)

        self.layer_importance: Dict[str, float] = {}
        self.prev_layer_importance: Dict[str, float] = {}
        self.global_importance_threshold = 0.5

        self.grad_history: Dict[str, list] = {}

        # 将梯度编码特征(64维) + 统计特征(6维: 正/负/零比例 + 标准差/波动/动量) 融合
        self.feature_fusion = nn.Linear(encoder_output_dim + 6, encoder_output_dim).to(device).to(dtype)

        self.meta_optimizer = torch.optim.AdamW(
            list(self.gradient_encoder.parameters()) +
            list(self.contrastive_learner.parameters()) +
            list(self.layer_attention.parameters()) +
            list(self.feature_fusion.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )

    def _compute_layer_gradients(self, accumulated_grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        layer_grads = {}
        for name, grad in accumulated_grads.items():
            layer_name = name.split('.')[0]
            if layer_name not in layer_grads:
                layer_grads[layer_name] = []
            layer_grads[layer_name].append(grad.to(self.device).flatten())

        for layer_name in layer_grads:
            layer_grads[layer_name] = torch.cat(layer_grads[layer_name], dim=0)

        return layer_grads

    def _analyze_gradient_direction(self, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positive_ratio = (grad > 0).float().mean()
        negative_ratio = (grad < 0).float().mean()
        zero_ratio = (grad == 0).float().mean()
        return positive_ratio, negative_ratio, zero_ratio

    def _analyze_gradient_stability(self, layer_name: str, current_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_name not in self.grad_history:
            self.grad_history[layer_name] = []
        
        grad_std = 0.0
        grad_diff = 0.0
        momentum = 0.0
        
        current_grad_norm = float(current_grad.norm().item())
        
        if len(self.grad_history[layer_name]) > 0:
            grad_std = float(np.std(self.grad_history[layer_name]))
            
            if len(self.grad_history[layer_name]) >= 2:
                recent_grad_norm = self.grad_history[layer_name][-1]
                prev_grad_norm = self.grad_history[layer_name][-2]
                grad_diff = float(np.abs(recent_grad_norm - prev_grad_norm))
                
                if len(self.grad_history[layer_name]) >= 3:
                    prev_prev_grad_norm = self.grad_history[layer_name][-3]
                    prev_diff = np.abs(prev_grad_norm - prev_prev_grad_norm)
                    momentum = float(grad_diff / (prev_diff + 1e-8))
        
        self.grad_history[layer_name].append(current_grad_norm)
        if len(self.grad_history[layer_name]) > self.grad_history_window:
            self.grad_history[layer_name].pop(0)
        
        return (
            torch.tensor(grad_std, device=self.device),
            torch.tensor(grad_diff, device=self.device),
            torch.tensor(momentum, device=self.device)
        )

    def _analyze_layer_correlation(self, layer_features: torch.Tensor) -> torch.Tensor:
        num_layers = layer_features.size(0)
        if num_layers < 2:
            return torch.tensor(0.0, device=self.device)

        # 皮尔逊相关系数：先对每层(每行)做 z-score 归一化，再计算协方差矩阵
        normalized_features = (layer_features - layer_features.mean(dim=1, keepdim=True)) / (layer_features.std(dim=1, keepdim=True) + 1e-8)
        correlation_matrix = torch.mm(normalized_features, normalized_features.t()) / self.encoder_output_dim
        avg_correlation = correlation_matrix.mean()

        return avg_correlation

    def _extract_layer_features(self, layer_grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        layer_features = []

        for layer_name in sorted(layer_grads.keys()):
            grad = layer_grads[layer_name]

            pos_ratio, neg_ratio, zero_ratio = self._analyze_gradient_direction(grad)
            stability, grad_diff, momentum = self._analyze_gradient_stability(layer_name, grad)

            # 方向特征(3) + 稳定性特征(3) → 6维统计向量
            stats = torch.stack([
                pos_ratio.detach(), neg_ratio.detach(), zero_ratio.detach(),
                stability.detach(), grad_diff.detach(), momentum.detach()
            ])

            if grad.size(0) < self.encoder_hidden_dim:
                grad = F.pad(grad, (0, self.encoder_hidden_dim - grad.size(0)))
            elif grad.size(0) > self.encoder_hidden_dim:
                grad = grad[:self.encoder_hidden_dim]

            base_features = self.gradient_encoder(grad.unsqueeze(0).to(self.dtype))
            fused = self.feature_fusion(torch.cat([base_features.squeeze(0), stats.to(self.dtype)], dim=0))

            layer_features.append(fused.unsqueeze(0))

        layer_features = torch.cat(layer_features, dim=0)

        return layer_features

    def _build_contrastive_samples(self, layer_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors = layer_features
        positives = layer_features.roll(1, dims=0)
        negatives = layer_features[torch.randperm(layer_features.size(0))]
        return anchors, positives, negatives

    def apply_mask(self, accumulated_grads: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict]:
        layer_grads = self._compute_layer_gradients(accumulated_grads)
        layer_features = self._extract_layer_features(layer_grads)

        layer_correlation = self._analyze_layer_correlation(layer_features)

        anchors, positives, negatives = self._build_contrastive_samples(layer_features)
        contrastive_loss = self.contrastive_learner(anchors, positives, negatives)

        fused_features = self.layer_attention(layer_features)
        importance_scores = torch.sigmoid(torch.mean(fused_features, dim=-1))

        avg_importance = importance_scores.mean().item()

        consistency_loss = 0.0
        for i, layer_name in enumerate(sorted(layer_grads.keys())):
            importance = importance_scores[i].item()
            if layer_name in self.prev_layer_importance:
                consistency_loss += (importance - self.prev_layer_importance[layer_name]) ** 2

            if layer_name in self.layer_importance:
                self.layer_importance[layer_name] = self.ema_alpha * self.layer_importance[layer_name] + (1 - self.ema_alpha) * importance
            else:
                self.layer_importance[layer_name] = importance

            self.prev_layer_importance[layer_name] = importance

        total_meta_loss = self.contrastive_weight * contrastive_loss + self.consistency_weight * consistency_loss - 0.05 * layer_correlation

        self.meta_optimizer.zero_grad()
        total_meta_loss.backward()
        self.meta_optimizer.step()

        masked_grads = {}
        for name, grad in accumulated_grads.items():
            layer_name = name.split('.')[0]
            importance = self.layer_importance.get(layer_name, self.global_importance_threshold)

            # 重要性加权：重要性高 → 梯度保留多；重要性低 → 梯度衰减多
            # 夹到 [0.1, 1.0] 防止完全归零
            weight = max(0.1, min(1.0, importance))
            masked_grads[name] = grad * weight

        info = {
            'avg_importance': avg_importance,
            'layer_corr': layer_correlation.item(),
            'contrastive_loss': contrastive_loss.item(),
            'consistency_loss': consistency_loss,
        }
        return masked_grads, info


class DGMMTrainer:
    def __init__(self, model, tokenizer, dgmm_config=None, device="cuda", lr=2e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        if dgmm_config is None:
            dgmm_config = {}
        self.dgmm = DGMMFramework(device=device, **dgmm_config)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        count = 0
        total_batches = len(dataloader)
        t_start = time.time()
        best_loss = float('inf')

        print(f"  [锚点] 训练开始 | 总步数={total_batches} | 时间={datetime.now().strftime('%H:%M:%S')}")
        print(f"  [锚点] 显存状态: {torch.cuda.memory_summary().split(chr(10))[0] if torch.cuda.is_available() else 'N/A'}")

        for batch in dataloader:
            inputs = {
                "input_ids": batch["labels"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device)
            }
            labels = batch['labels'].to(self.device)

            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss

            # NaN 检测
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ❌ [中止] Step {count}: loss={loss.item()} — 立即停止!")
                return float('nan')

            self.optimizer.zero_grad()
            loss.backward()

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

            total_loss += loss.item()
            count += 1

            if count % 10 == 0 or count == 1:
                elapsed = time.time() - t_start
                avg_time = elapsed / count
                eta = avg_time * (total_batches - count)
                mask_pct = ""
                if dgmm_info:
                    imp = dgmm_info.get('avg_importance', 0)
                    mask_pct = f" | imp={imp:.3f} corr={dgmm_info.get('layer_corr', 0):.3f}"
                status = "✅" if loss.item() < best_loss else "  "
                best_loss = min(best_loss, loss.item())
                logger.info(f"  {status} Step {count}/{total_batches} | loss={loss.item():.4f} | "
                            f"elapsed={elapsed:.0f}s | eta={eta:.0f}s{mask_pct}")
                print(f"  {status} Step {count}/{total_batches} | loss={loss.item():.4f} | "
                      f"elapsed={elapsed:.0f}s | eta={eta:.0f}s{mask_pct}")

        print(f"  [锚点] 训练完成 | avg_loss={total_loss/count:.4f} | 耗时={time.time()-t_start:.0f}s")
        return total_loss / count if count > 0 else 0.0


def load_magicoder_dataset():
    logger.info("Loading Magicoder-Evol-Instruct-110K dataset from local path...")
    dataset_path = LOCAL_PATHS["dataset"]
    if os.path.exists(dataset_path):
        dataset = load_dataset(dataset_path, split="train")
    else:
        logger.info(f"Local path not found: {dataset_path}, downloading from Hugging Face...")
        dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    return dataset


def preprocess_dataset(dataset, tokenizer, max_length=256):
    def format_instruction(example):
        instruction = example.get('instruction', '')
        response = example.get('response', '')
        return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

    dataset = dataset.map(lambda x: {"text": format_instruction(x)})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["instruction", "response", "text"])
    tokenized_dataset = tokenized_dataset.rename_column("input_ids", "labels")

    return tokenized_dataset


def create_dataloader(dataset, batch_size=4):
    dataset.set_format(type="torch", columns=["labels", "attention_mask"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def load_model(model_name, device="cuda", use_quantization=False):
    logger.info(f"Loading model: {model_name}")

    # 检查本地路径
    local_path = LOCAL_PATHS.get(model_name)
    if local_path and os.path.exists(local_path):
        model_path = local_path
        logger.info(f"Using local model path: {model_path}")
    else:
        model_path = model_name
        logger.info(f"Local path not found, using Hugging Face: {model_path}")

    quantization_config = None
    if use_quantization and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    logger.info(f"Model loaded successfully. Parameters: {model.num_parameters():,}")
    return model, tokenizer


def evaluate_on_benchmark(model, tokenizer, benchmark_name, device="cuda"):
    logger.info(f"Evaluating on {benchmark_name}...")

    try:
        if benchmark_name == "humaneval":
            dataset = load_dataset("openai_humaneval", split="test")
        elif benchmark_name == "mbpp":
            dataset = load_dataset("mbpp", split="test")
        elif benchmark_name == "humaneval_plus":
            from evalplus.data import get_human_eval_plus
            problems = get_human_eval_plus()
            dataset = [{"prompt": v["prompt"], "test": v["test"], "entry_point": v.get("entry_point", "")}
                       for v in problems.values()]
        elif benchmark_name == "mbpp_plus":
            from evalplus.data import get_mbpp_plus
            problems = get_mbpp_plus()
            dataset = [{"prompt": v["prompt"], "test": v["test"], "entry_point": v.get("entry_point", "")}
                       for v in problems.values()]
        else:
            logger.warning(f"Unknown benchmark: {benchmark_name}")
            return 0.0

        correct = 0
        total = min(len(dataset), 100)
        t_start = time.time()
        total_generated = 0  # 统计非空生成数

        print(f"  [锚点] {benchmark_name}: 开始评测 {total} 题 | {datetime.now().strftime('%H:%M:%S')}")

        model.eval()
        for i in range(total):
            example = dataset[i]
            prompt = example.get('prompt', '')
            test = example.get('test', '')
            entry_point = example.get('entry_point', '')

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs.input_ids.shape[1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                top_k=1,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_ids = outputs[0][input_len:]
            generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True)

            if not generated_code.strip():
                continue

            total_generated += 1

            # 前3个样本打印出来确认模型在正常生成
            if i < 3:
                print(f"  [样本 #{i+1}] {benchmark_name}")
                print(f"    prompt[:80]: {prompt[:80]}...")
                print(f"    generated[:120]: {generated_code[:120]}...")
                print(f"    ----")

            try:
                exec_globals = {}
                # HumanEval: prompt 是函数签名 (合法 Python), 拼接后 exec
                exec(prompt + generated_code + "\n" + test, exec_globals)
                correct += 1
            except Exception:
                try:
                    # MBPP: prompt 是自然语言描述, 只用生成部分 exec
                    exec(generated_code + "\n" + test, exec_globals)
                    correct += 1
                except Exception:
                    pass

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                avg_time = elapsed / (i + 1)
                eta = avg_time * (total - i - 1)
                logger.info(f"  [{benchmark_name}] {i+1}/{total} | correct={correct} | "
                            f"elapsed={elapsed:.0f}s | eta={eta:.0f}s")

        pass_rate = correct / total if total > 0 else 0.0
        logger.info(f"{benchmark_name} pass@1: {pass_rate:.4f}")
        print(f"  [锚点] {benchmark_name} 完成 | 有效生成: {total_generated}/{total} | "
              f"正确: {correct}/{total} | pass@1={pass_rate:.4f} | 耗时={time.time()-t_start:.0f}s")
        return pass_rate

    except Exception as e:
        logger.error(f"Error evaluating on {benchmark_name}: {e}")
        return 0.0


def run_experiment(model_name, dataset, algorithm_name="DGMM", num_epochs=3, batch_size=1, lr=2e-5):
    logger.info(f"\n===== Running {algorithm_name} with {model_name} =====")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(">>> [CHECKPOINT 1/6] Loading model...")
    model, tokenizer = load_model(model_name, device=device, use_quantization=True)
    model.gradient_checkpointing_enable()
    logger.info(f">>> [CHECKPOINT 2/6] Model loaded successfully on {device}")

    logger.info(">>> [CHECKPOINT 3/6] Creating dataloader...")
    dataloader = create_dataloader(dataset, batch_size=batch_size)
    logger.info(">>> [CHECKPOINT 3/6] Dataloader created")

    trainer = DGMMTrainer(model, tokenizer, device=device, lr=lr)

    # 模型保存路径（训练完保存，下次可用 skip_training=True 跳过训练）
    safe_name = model_name.replace("/", "_")
    ckpt_dir = f"checkpoints/{safe_name}_{algorithm_name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        logger.info(f">>> [CHECKPOINT 4.{epoch+1}/{num_epochs}] Training epoch {epoch+1} started...")
        loss = trainer.train_epoch(dataloader)
        t_epoch = time.time() - t_epoch_start
        logger.info(f">>> [CHECKPOINT 4.{epoch+1}/{num_epochs}] Training epoch {epoch+1} completed, "
                    f"loss: {loss:.4f}, time: {t_epoch:.0f}s")

    # 保存模型，避免断连后白训
    logger.info(f">>> [CHECKPOINT 4.5] Saving model to {ckpt_dir}...")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    logger.info(f">>> [CHECKPOINT 4.5] Model saved ✓")

    benchmarks = ["humaneval", "mbpp", "humaneval_plus", "mbpp_plus"]
    results = {}
    for i, benchmark in enumerate(benchmarks):
        logger.info(f">>> [CHECKPOINT 5.{i+1}/{len(benchmarks)}] Evaluating on {benchmark}...")
        score = evaluate_on_benchmark(model, tokenizer, benchmark, device=device)
        results[benchmark] = score
        logger.info(f">>> [CHECKPOINT 5.{i+1}/{len(benchmarks)}] {benchmark} completed, score: {score:.4f}")

    avg_score = sum(results.values()) / len(results) if results else 0.0
    results['average'] = avg_score
    logger.info(f">>> [CHECKPOINT 6/6] All evaluations completed, average score: {avg_score:.4f}")

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return results


def main():
    algorithm_name = "DGMM"
    models = [
        "mistralai/Mistral-7B-v0.1",
        # "deepseek-ai/DeepSeek-Coder-Base-6.7B",  # 先注释掉，跑完一个再看
    ]
    benchmarks = ["HumanEval", "MBPP", "HumanEval+", "MBPP+", "Average"]

    logger.info(f"===== Starting Code Generation Experiment - {algorithm_name} =====")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Models: {models}")
    logger.info(f"Benchmarks: {benchmarks}")

    all_results = []

    for model_name in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = load_magicoder_dataset()
        preprocessed_dataset = preprocess_dataset(dataset, tokenizer)

        results = run_experiment(model_name, preprocessed_dataset, algorithm_name)
        if results:
            all_results.append({
                "Model": model_name,
                "Algorithm": algorithm_name,
                "HumanEval": results.get('humaneval', 0.0),
                "MBPP": results.get('mbpp', 0.0),
                "HumanEval+": results.get('humaneval_plus', 0.0),
                "MBPP+": results.get('mbpp_plus', 0.0),
                "Average": results.get('average', 0.0)
            })

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"results/code_generation_{algorithm_name.lower()}_results_{timestamp}.csv"

    with open(csv_filename, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Algorithm", "HumanEval", "MBPP", "HumanEval+", "MBPP+", "Average"])

        for result in all_results:
            writer.writerow([
                result["Model"],
                result["Algorithm"],
                f"{result['HumanEval']:.4f}",
                f"{result['MBPP']:.4f}",
                f"{result['HumanEval+']:.4f}",
                f"{result['MBPP+']:.4f}",
                f"{result['Average']:.4f}"
            ])

    logger.info(f"\n===== Results saved to {csv_filename} =====")

    print("\n" + "="*60)
    print(f"Final Results Summary - {algorithm_name}")
    print("="*60)
    for result in all_results:
        print(f"\n{result['Model']} - {result['Algorithm']}:")
        print(f"  HumanEval:  {result['HumanEval']:.4f}")
        print(f"  MBPP:       {result['MBPP']:.4f}")
        print(f"  HumanEval+: {result['HumanEval+']:.4f}")
        print(f"  MBPP+:      {result['MBPP+']:.4f}")
        print(f"  Average:    {result['Average']:.4f}")


if __name__ == "__main__":
    main()