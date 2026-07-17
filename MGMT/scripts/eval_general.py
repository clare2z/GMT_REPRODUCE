"""
General domain evaluation: MMLU, GSM8k, BBH, TyDiQA, TruthfulQA, HumanEval.

Matches GMT paper Table 2 evaluation protocol.
Uses lm-evaluation-harness under the hood for most benchmarks.

Usage:
    python scripts/eval_general.py \
        --model_path ./outputs/general_llama2_7b_la_mo_gmt/final \
        --tasks mmlu gsm8k bbh tydiqa truthfulqa humaneval
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Evaluation config matching GMT paper
TASK_CONFIG = {
    "mmlu": {"num_fewshot": 0, "metric": "acc"},
    "gsm8k": {"num_fewshot": 8, "metric": "exact_match"},
    "bbh": {"num_fewshot": 3, "metric": "exact_match"},
    "tydiqa": {"num_fewshot": 1, "metric": "f1"},
    "truthfulqa_mc2": {"num_fewshot": 0, "metric": "acc"},
    "humaneval": {"num_fewshot": 0, "metric": "pass@10"},
}


def evaluate_with_harness(model, tokenizer, tasks, batch_size=1):
    """Run lm-evaluation-harness."""
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.error("lm-eval not installed. Run: pip install lm_eval")
        sys.exit(1)

    results = {}
    for task in tasks:
        cfg = TASK_CONFIG.get(task, {})
        logger.info(f"Evaluating {task} ({cfg.get('num_fewshot', 0)}-shot)...")

        try:
            lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

            result = simple_evaluate(
                model=lm,
                tasks=[task],
                num_fewshot=cfg.get("num_fewshot", 0),
                batch_size=batch_size,
                log_samples=False,
            )

            metric_key = cfg.get("metric", "acc")
            task_result = result["results"].get(task, {})
            score = task_result.get(metric_key)
            if score is None:
                # lm-eval may use suffixes like "exact_match,flexible-extract"
                for k, v in task_result.items():
                    if k.startswith(metric_key) and "_stderr" not in k and v is not None:
                        score = v
                        metric_key = k
                        break

            if score is not None:
                results[task] = {
                    "score": score * 100,
                    "num_fewshot": cfg.get("num_fewshot", 0),
                    "metric": metric_key,
                }
                logger.info(f"  {task}: {score * 100:.1f}")
            else:
                logger.warning(f"  {task}: no result for metric={metric_key}")
                logger.info(f"  Available metrics: {list(task_result.keys())}")

        except Exception as e:
            logger.error(f"  {task}: FAILED - {e}")

    return results


def _evaluate_gsm8k_with_shots(model, tokenizer, shot_examples, max_samples=None):
    """Evaluate GSM8k with few-shot chain-of-thought examples."""
    import re
    from datasets import load_dataset
    from tqdm import tqdm

    dataset = load_dataset("gsm8k", "main", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0
    for example in tqdm(dataset, desc="GSM8k-8shot"):
        prompt = shot_examples + "<|user|>" + chr(10) + "Question: " + example["question"] + chr(10) + "Let's solve this step by step." + chr(10) + "<|assistant|>" + chr(10)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, temperature=0.0,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Extract final answer after ####
        match = re.search(r"####\s*(\S+)", response)
        pred = match.group(1).strip() if match else ""
        match = re.search(r"####\s*(\S+)", example["answer"])
        true_ans = match.group(1).strip() if match else ""

        # Normalize
        pred = re.sub(r"[^\d.,\-]", "", pred).replace(",", "")
        true_ans = re.sub(r"[^\d.,\-]", "", true_ans).replace(",", "")
        try:
            if abs(float(pred) - float(true_ans)) < 1e-6:
                correct += 1
        except ValueError:
            pass
        total += 1

    acc = 100.0 * correct / total if total > 0 else 0
    logger.info(f"  GSM8k: {acc:.1f}% ({correct}/{total})")
    return {"gsm8k": acc, "gsm8k_correct": correct, "gsm8k_total": total}


def _squad_f1(prediction, references):
    """Compute SQuAD-style F1 score over multiple references."""
    import re
    from collections import Counter

    def _tokenize(text):
        return re.sub(r"[^\w\s]", "", text.lower()).split()

    pred_tokens = _tokenize(prediction)
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for ref in references:
        ref_tokens = _tokenize(ref)
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def _evaluate_tydiqa(model, tokenizer, max_samples=None, batch_size=1):
    """Evaluate TyDiQA with 1-shot using datasets directly."""
    from datasets import load_dataset
    from tqdm import tqdm

    train_ds = load_dataset("google-research-datasets/tydiqa", "secondary_task", split="train")
    val_ds = load_dataset("google-research-datasets/tydiqa", "secondary_task", split="validation")

    shot = train_ds[0]
    shot_prompt = "Passage: " + shot["context"] + chr(10)
    shot_prompt += "Question: " + shot["question"] + chr(10)
    shot_prompt += "Answer: " + shot["answers"]["text"][0] + chr(10) + chr(10)

    if max_samples:
        val_ds = val_ds.select(range(min(max_samples, len(val_ds))))

    total_f1 = 0.0
    total = 0
    for ex in tqdm(val_ds, desc="TyDiQA"):
        prompt = shot_prompt + "Passage: " + ex["context"] + chr(10)
        prompt += "Question: " + ex["question"] + chr(10)
        prompt += "Answer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        pred = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip().split(chr(10))[0].strip()

        refs = ex["answers"]["text"]
        total_f1 += _squad_f1(pred, refs)
        total += 1

    avg_f1 = 100.0 * total_f1 / total if total > 0 else 0
    logger.info(f"  TyDiQA: F1={avg_f1:.1f} ({total} samples)")
    return avg_f1


def evaluate_manual(model, tokenizer, tasks, max_samples=None, batch_size=1):
    """
    Fallback evaluation using datasets directly.
    Handles tasks that lm-eval-harness might struggle with.
    """
    results = {}
    from datasets import load_dataset
    from tqdm import tqdm

    if "mmlu" in tasks:
        logger.info("Evaluating MMLU (0-shot)...")
        import re
        subjects = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology", "high_school_statistics",
            "high_school_us_history", "high_school_world_history", "human_aging",
            "human_sexuality", "international_law", "jurisprudence", "logical_fallacies",
            "machine_learning", "management", "marketing", "medical_genetics",
            "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
            "philosophy", "prehistory", "professional_accounting", "professional_law",
            "professional_medicine", "professional_psychology", "public_relations",
            "security_studies", "sociology", "us_foreign_policy",
            "virology", "world_religions",
        ]

        total_correct = 0
        total_samples = 0
        for subject in tqdm(subjects):
            try:
                ds = load_dataset("cais/mmlu", subject, split="test")
                for ex in ds:
                    choices = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(ex["choices"])])
                    prompt = f"Question: {ex['question']}\n{choices}\nAnswer:"

                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        logits = model(**inputs).logits[0, -1, :]
                        # Only consider A, B, C, D tokens
                        option_ids = [tokenizer.encode(chr(65+i))[-1] for i in range(len(ex["choices"]))]
                        option_logits = logits[option_ids]
                        pred = option_logits.argmax().item()

                    if pred == ex["answer"]:
                        total_correct += 1
                    total_samples += 1
                    if max_samples and total_samples >= max_samples:
                        break
                if max_samples and total_samples >= max_samples:
                    break
            except Exception:
                continue

        score = 100 * total_correct / total_samples if total_samples > 0 else 0
        results["mmlu"] = {"score": score}
        logger.info(f"  MMLU: {score:.1f} ({total_correct}/{total_samples})")

    if "gsm8k" in tasks:
        logger.info("Evaluating GSM8k (8-shot)...")

        # 8-shot chain-of-thought examples (standard GSM8k prompt)
        shot_examples = "<|user|>\nQuestion: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n<|assistant|>\nLet's solve this step by step.\nThere are 15 trees originally. Then there were 21 trees after some more were planted. So 21 - 15 = 6 trees were planted.\n#### 6\n\n<|user|>\nQuestion: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n<|assistant|>\nLet's solve this step by step.\nThere are originally 3 cars. 2 more cars arrive. 3 + 2 = 5 cars.\n#### 5\n\n<|user|>\nQuestion: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n<|assistant|>\nLet's solve this step by step.\nOriginally, Leah had 32 chocolates. Her sister had 42. So total chocolates = 32 + 42 = 74. After eating 35, remaining = 74 - 35 = 39.\n#### 39\n\n<|user|>\nQuestion: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\n<|assistant|>\nLet's solve this step by step.\nJason started with 20 lollipops. Then he had 12 after giving some to Denny. So 20 - 12 = 8 lollipops were given to Denny.\n#### 8\n\n<|user|>\nQuestion: Shawn has five toys. For Christmas, he got two toys from his mom and two from his dad. How many toys does he have now?\n<|assistant|>\nLet's solve this step by step.\nShawn started with 5 toys. He got 2 from mom, so 5 + 2 = 7. He got 2 from dad, so 7 + 2 = 9 toys.\n#### 9\n\n<|user|>\nQuestion: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\n<|assistant|>\nLet's solve this step by step.\nThere were originally 9 computers. For each day from monday to thursday, 5 computers were added. That's 4 days x 5 = 20 computers. So 9 + 20 = 29 computers.\n#### 29\n\n<|user|>\nQuestion: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n<|assistant|>\nLet's solve this step by step.\nMichael started with 58 golf balls. After losing 23 on tuesday, 58 - 23 = 35. After losing 2 on wednesday, 35 - 2 = 33 golf balls.\n#### 33\n\n<|user|>\nQuestion: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\n<|assistant|>\nLet's solve this step by step.\nOlivia had 23 dollars. 5 bagels for 3 dollars each = 5 x 3 = 15 dollars. Money left = 23 - 15 = 8 dollars.\n#### 8"
        gsm8k_result = _evaluate_gsm8k_with_shots(model, tokenizer, shot_examples, max_samples)
        results["gsm8k"] = {"score": gsm8k_result["gsm8k"]}

    if "tydiqa" in tasks:
        logger.info("Evaluating TyDiQA (1-shot)...")
        f1_score = _evaluate_tydiqa(model, tokenizer, max_samples, batch_size)
        results["tydiqa"] = {
            "score": f1_score,
            "num_fewshot": 1,
            "metric": "f1",
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["mmlu", "gsm8k", "bbh", "tydiqa", "truthfulqa", "humaneval"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_harness", action="store_true",
                        help="Use lm-evaluation-harness (recommended)")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    args = parser.parse_args()

    os.makedirs(args.model_path, exist_ok=True)

    load_kwargs = {"torch_dtype": torch.bfloat16}
    if args.use_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    if not args.use_8bit:
        model = model.cuda()
    model.eval()

    # Load existing results for resume
    existing_results = {}
    result_file = os.path.join(args.model_path, "eval_results.json")
    if os.path.exists(result_file):
        with open(result_file) as f:
            existing = json.load(f)
            existing_results = existing.get("results", {})
    
    # Filter out already-completed tasks
    remaining_tasks = [t for t in args.tasks if t not in existing_results]
    skipped_tasks = [t for t in args.tasks if t in existing_results]
    if skipped_tasks:
        logger.info(f"Skipping completed tasks: {skipped_tasks}")
    if not remaining_tasks:
        logger.info("All tasks already completed!")
        results = existing_results
    else:
        # Run evaluation
        if args.use_harness:
            harness_tasks = [t for t in remaining_tasks if t not in ("tydiqa", "gsm8k")]
            manual_tasks = [t for t in remaining_tasks if t in ("tydiqa", "gsm8k")]
            results = {}
            if harness_tasks:
                results.update(evaluate_with_harness(model, tokenizer, harness_tasks, batch_size=args.batch_size))
            if manual_tasks:
                results.update(evaluate_manual(model, tokenizer, manual_tasks, args.max_samples, args.batch_size))
        else:
            results = evaluate_manual(model, tokenizer, remaining_tasks, args.max_samples, args.batch_size)
        # Merge with existing
        results.update(existing_results)

    # ── Summary ──
    scores = {}
    for task, info in results.items():
        scores[task] = info["score"]

    vals = [v for v in scores.values() if v is not None]
    avg = sum(vals) / len(vals) if vals else 0

    summary = {"results": results, "average": avg}
    result_file = os.path.join(args.model_path, "eval_results.json")
    with open(result_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=" * 50)
    logger.info("General Domain Evaluation Results")
    logger.info("=" * 50)
    for task, score in scores.items():
        logger.info(f"  {task:20s}: {score:.1f}" if score else f"  {task:20s}: N/A")
    logger.info(f"  {'Average':20s}: {avg:.1f}")
    logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
