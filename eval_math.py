"""
Math reasoning evaluation: GSM8k, MATH.

Usage:
    python scripts/eval_math.py \
        --model_path ./outputs/math_mistral_la_mo_gmt/final \
        --tasks gsm8k math \
        --num_samples 10
"""

from __future__ import annotations

import os
import sys
import json
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

MATH_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

GSM8K_PROMPT = """Question: {question}

Let's solve this step by step.
"""


def extract_gsm8k_answer(text: str) -> str:
    """Extract the final numeric answer from GSM8k output."""
    # Look for patterns like "#### 42" or "the answer is 42"
    match = re.search(r"####\s*(\S+)", text)
    if match:
        return match.group(1).strip()

    # Try "answer is X" pattern
    match = re.search(r"(?:answer|final answer|result)\s*(?:is|:|=)\s*(\S+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip(".")

    # Last number in the text
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return numbers[-1] if numbers else ""


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content inside \\boxed{...} with proper nested-brace handling."""
    # Find all occurrences of \boxed{
    idx = 0
    candidates = []
    while True:
        pos = text.find("\\boxed{", idx)
        if pos == -1:
            break
        # Brace-count from the opening brace
        start = pos + len("\\boxed{")
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            candidates.append(text[start:i - 1])
        idx = pos + 1

    if not candidates:
        return None
    # Last \boxed{} is usually the final answer
    return candidates[-1]


def _remove_latex_commands(s: str) -> str:
    """Remove LaTeX formatting commands like \\text{...}, \\mathrm{...} etc.
    Handles nested braces and keeps the inner content."""
    cmds = ["text", "mathrm", "mathbf", "mathit", "mathsf", "mathtt",
            "bm", "boldsymbol", "emph", "textrm", "textsf", "texttt"]

    result = []
    i = 0
    while i < len(s):
        if s[i] == "\\":
            matched = False
            for cmd in cmds:
                prefix = "\\" + cmd + "{"
                if s[i:i + len(prefix)] == prefix:
                    start = i + len(prefix)
                    depth = 1
                    j = start
                    while j < len(s) and depth > 0:
                        if s[j] == "{":
                            depth += 1
                        elif s[j] == "}":
                            depth -= 1
                        j += 1
                    if depth == 0:
                        inner = s[start:j - 1]
                        result.append(inner)
                        i = j
                        matched = True
                        break
            if not matched:
                result.append(s[i])
                i += 1
        else:
            result.append(s[i])
            i += 1
    return "".join(result)


def extract_math_answer(text: str) -> str:
    """Extract answer from MATH dataset output with nested-brace-aware \\boxed{} parsing."""
    ans = _extract_boxed(text)
    if ans:
        ans = _remove_latex_commands(ans)
        ans = ans.replace("\\%", "%")
        return ans.strip()

    # Fallback: last line
    lines = text.strip().split("\n")
    return lines[-1].strip()


def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison, preserving LaTeX mathematical structure."""
    ans = ans.strip()

    # Remove LaTeX formatting commands but keep their inner content
    ans = _remove_latex_commands(ans)

    # Normalize common LaTeX aliases
    ans = ans.replace("\\%", "%")
    ans = ans.replace("\\left", "")
    ans = ans.replace("\\right", "")
    ans = ans.replace("\\,", " ")
    ans = ans.replace("\\;", " ")
    ans = ans.replace("\\!", "")
    ans = ans.replace("\\ ", " ")

    # Remove backslash from simple commands that should be bare symbols
    for sym in ["leq", "geq", "neq", "approx", "equiv", "pm", "times", "div",
                "cdot", "ast", "star", "circ", "bullet", "sum", "prod", "int",
                "infty", "alpha", "beta", "gamma", "delta", "epsilon", "pi"]:
        ans = ans.replace("\\" + sym + " ", sym + " ")
        if ans.endswith("\\" + sym):
            ans = ans[:-len("\\" + sym)] + sym

    # Normalize whitespace
    ans = re.sub(r"\s+", " ", ans).strip()

    # Lowercase for case-insensitive comparison
    ans = ans.lower()

    return ans


def evaluate_gsm8k(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_samples: int = None,
) -> Dict:
    """Evaluate on GSM8k."""
    dataset = load_dataset("gsm8k", "main", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0

    for example in tqdm(dataset, desc="GSM8k"):
        question = example["question"]
        prompt = GSM8K_PROMPT.format(question=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        pred = extract_gsm8k_answer(response)
        true_ans = extract_gsm8k_answer(example["answer"])

        if normalize_answer(pred) == normalize_answer(true_ans):
            correct += 1
        total += 1

    acc = 100.0 * correct / total if total > 0 else 0
    return {"gsm8k": acc, "gsm8k_correct": correct, "gsm8k_total": total}


def evaluate_math(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_samples: int = None,
) -> Dict:
    """Evaluate on MATH dataset."""
    dataset = load_dataset("/root/autodl-tmp/dataset/competition_math", split="test", trust_remote_code=True)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0

    for example in tqdm(dataset, desc="MATH"):
        prompt = MATH_PROMPT.format(
            instruction=f"Solve the following math problem step by step:\n\n{example['problem']}"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        pred = extract_math_answer(response)
        true_ans = _extract_boxed(example["solution"]) if "\\boxed{" in example["solution"] else ""

        if normalize_answer(pred) == normalize_answer(true_ans):
            correct += 1
        total += 1

    acc = 100.0 * correct / total if total > 0 else 0
    return {"math": acc, "math_correct": correct, "math_total": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+", default=["gsm8k", "math"])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples for quick test")
    parser.add_argument("--use_8bit", action="store_true")
    args = parser.parse_args()

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

    all_metrics = {}

    if "gsm8k" in args.tasks:
        m = evaluate_gsm8k(model, tokenizer, args.max_samples)
        all_metrics.update(m)
        logger.info(f"GSM8k: {m['gsm8k']:.1f}% ({m['gsm8k_correct']}/{m['gsm8k_total']})")

    if "math" in args.tasks:
        m = evaluate_math(model, tokenizer, args.max_samples)
        all_metrics.update(m)
        logger.info(f"MATH: {m['math']:.1f}% ({m['math_correct']}/{m['math_total']})")

    valid_scores = [v for k, v in all_metrics.items()
                    if k in ("gsm8k", "math") and isinstance(v, (int, float))]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    all_metrics["average"] = avg

    logger.info(f"Math Results: {json.dumps({k: v for k, v in all_metrics.items() if isinstance(v, (int, float))}, indent=2)}")
    logger.info(f"Average: {avg:.2f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

