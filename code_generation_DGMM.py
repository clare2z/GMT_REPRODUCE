import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import csv
import os
from datetime import datetime
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
        dtype: torch.dtype = torch.bfloat16
    ):
        self.device = device
        self.dtype = dtype
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight
        self.ema_alpha = ema_alpha

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

        self.meta_optimizer = torch.optim.AdamW(
            list(self.gradient_encoder.parameters()) +
            list(self.contrastive_learner.parameters()) +
            list(self.layer_attention.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )

        self.layer_importance: Dict[str, float] = {}
        self.prev_layer_importance: Dict[str, float] = {}
        self.global_importance_threshold = 0.5

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

    def _extract_layer_features(self, layer_grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        layer_features = []
        for layer_name in sorted(layer_grads.keys()):
            grad = layer_grads[layer_name]
            if grad.size(0) < self.encoder_hidden_dim:
                grad = F.pad(grad, (0, self.encoder_hidden_dim - grad.size(0)))
            elif grad.size(0) > self.encoder_hidden_dim:
                grad = grad[:self.encoder_hidden_dim]

            features = self.gradient_encoder(grad.unsqueeze(0).to(self.dtype))
            layer_features.append(features)

        return torch.cat(layer_features, dim=0)

    def _build_contrastive_samples(self, layer_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors = layer_features
        positives = layer_features.roll(1, dims=0)
        negatives = layer_features[torch.randperm(layer_features.size(0))]
        return anchors, positives, negatives

    def apply_mask(self, accumulated_grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        layer_grads = self._compute_layer_gradients(accumulated_grads)
        layer_features = self._extract_layer_features(layer_grads)

        anchors, positives, negatives = self._build_contrastive_samples(layer_features)
        contrastive_loss = self.contrastive_learner(anchors, positives, negatives)

        fused_features = self.layer_attention(layer_features)
        importance_scores = torch.sigmoid(torch.mean(fused_features, dim=-1))

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

        total_meta_loss = self.contrastive_weight * contrastive_loss + self.consistency_weight * consistency_loss

        self.meta_optimizer.zero_grad()
        total_meta_loss.backward()
        self.meta_optimizer.step()

        masked_grads = {}
        for name, grad in accumulated_grads.items():
            layer_name = name.split('.')[0]
            importance = self.layer_importance.get(layer_name, self.global_importance_threshold)

            mask = torch.rand(grad.size(), device=self.device) < importance
            masked_grads[name] = grad * mask.to(self.dtype)

        return masked_grads


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

        for batch in dataloader:
            inputs = {
                "input_ids": batch["labels"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device)
            }
            labels = batch['labels'].to(self.device)

            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss

            self.optimizer.zero_grad()
            loss.backward()

            accumulated_grads = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    accumulated_grads[name] = param.grad.clone().detach()

            if accumulated_grads:
                masked_grads = self.dgmm.apply_mask(accumulated_grads)
                for name, param in self.model.named_parameters():
                    if name in masked_grads:
                        param.grad = masked_grads[name]

            self.optimizer.step()

            total_loss += loss.item()
            count += 1

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


def preprocess_dataset(dataset, tokenizer, max_length=512):
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


def load_model(model_name, device="cuda", use_quantization=True):
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
            dataset = load_dataset("evalplus/humaneval_plus", split="test")
        elif benchmark_name == "mbpp_plus":
            dataset = load_dataset("evalplus/mbpp_plus", split="test")
        else:
            logger.warning(f"Unknown benchmark: {benchmark_name}")
            return 0.0

        correct = 0
        total = min(len(dataset), 100)

        model.eval()
        for i, example in enumerate(dataset.select(range(total))):
            prompt = example.get('prompt', '')
            test = example.get('test', '')

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                top_k=1,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

            try:
                exec_globals = {}
                exec(generated_code + "\n" + test, exec_globals)
                correct += 1
            except Exception:
                pass

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{total}, Correct: {correct}")

        pass_rate = correct / total if total > 0 else 0.0
        logger.info(f"{benchmark_name} pass@1: {pass_rate:.4f}")
        return pass_rate

    except Exception as e:
        logger.error(f"Error evaluating on {benchmark_name}: {e}")
        return 0.0


def run_experiment(model_name, dataset, algorithm_name="DGMM", num_epochs=3, batch_size=4, lr=2e-5):
    logger.info(f"\n===== Running {algorithm_name} with {model_name} =====")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(model_name, device=device)
    dataloader = create_dataloader(dataset, batch_size=batch_size)

    trainer = DGMMTrainer(model, tokenizer, device=device, lr=lr)

    for epoch in range(num_epochs):
        loss = trainer.train_epoch(dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    benchmarks = ["humaneval", "mbpp", "humaneval_plus", "mbpp_plus"]
    results = {}
    for benchmark in benchmarks:
        score = evaluate_on_benchmark(model, tokenizer, benchmark, device=device)
        results[benchmark] = score

    avg_score = sum(results.values()) / len(results) if results else 0.0
    results['average'] = avg_score

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return results


def main():
    algorithm_name = "DGMM"
    models = ["mistralai/Mistral-7B-v0.1", "deepseek-ai/DeepSeek-Coder-Base-6.7B"]
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