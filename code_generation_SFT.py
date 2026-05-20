import torch
import torch.nn as nn
import torch.nn.functional as F
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


class SFTTrainer:
    def __init__(self, model, tokenizer, device="cuda", lr=2e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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


def run_experiment(model_name, dataset, algorithm_name="SFT", num_epochs=3, batch_size=4, lr=2e-5):
    logger.info(f"\n===== Running {algorithm_name} with {model_name} =====")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(model_name, device=device)
    dataloader = create_dataloader(dataset, batch_size=batch_size)

    trainer = SFTTrainer(model, tokenizer, device=device, lr=lr)

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
    algorithm_name = "SFT"
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