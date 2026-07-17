"""
Data loading utilities for SFT and DPO training.
Handles three data formats auto-detected from column names:
  1. instruction + output   → Magicoder
  2. query + response       → MetaMathQA
  3. messages [{role, content}] → Tulu V2 / chat format
"""

from __future__ import annotations

import glob as glob_mod
import os
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Tuple
import torch


# ── Prompt templates ──

# DeepSeek-Coder (code generation)
CODE_PROMPT_DEEPSEEK = (
    "<｜begin▁of▁sentence｜>You are an AI programming assistant. "
    "You only answer questions related to computer science.\n"
    "### Instruction:\n{instruction}\n### Response:\n"
)

# Mistral (code / math)
CODE_PROMPT_MISTRAL = "[INST] {instruction} [/INST]"

# Math reasoning (Mistral / Llama3)
MATH_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

# Tulu V2 / Llama2 chat format
TULU_PROMPT_TEMPLATE = "<|user|>\n{instruction}\n<|assistant|>\n"


# ── Main SFT loading ──

def load_sft_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    prompt_template: str = "default",
    split: str = "train",
) -> Dataset:
    """
    Load and tokenize an SFT dataset. Auto-detects column format.

    Supported column formats:
      - 'instruction' + 'output'   (Magicoder)
      - 'query' + 'response'       (MetaMathQA)
      - 'messages' [{'role':'user','content':...}, ...]  (Tulu V2)
      - 'prompt' + 'completion'    (generic)
      - 'question' + 'answer'      (generic)
    """
    dataset = _load_dataset(dataset_path, split)
    columns = dataset.column_names

    # Detect format
    fmt = _detect_format(columns)
    # Get prompt function
    pfunc = _get_prompt_func(prompt_template)

    def format_and_tokenize(examples):
        instructions, responses = _extract_instruction_response(examples, fmt)

        texts = []
        for inst, resp in zip(instructions, responses):
            prompt = pfunc(inst)
            text = prompt + resp + tokenizer.eos_token
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Remove original columns
    cols_remove = [c for c in columns if c not in ("input_ids", "labels", "attention_mask")]
    dataset = dataset.map(
        format_and_tokenize,
        batched=True,
        remove_columns=cols_remove,
        desc="Tokenizing",
    )
    return dataset


# ── DPO loading ──


def _to_text(msg):
    """Convert messages to text."""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, list):
        return chr(10).join([m.get("content","") if isinstance(m,dict) else str(m) for m in msg])
    return str(msg)

def load_dpo_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    max_prompt_length: int = 512,
    split: str = "train",
) -> Dataset:
    """Load DPO dataset. Expected columns: prompt, chosen, rejected."""
    dataset = _load_dataset(dataset_path, split)

    def tokenize_fn(examples):
        out = {
            "chosen_input_ids": [], "chosen_attention_mask": [],
            "rejected_input_ids": [], "rejected_attention_mask": [],
        }
        for prompt, chosen, rejected in zip(
            examples["prompt"], examples["chosen"], examples["rejected"]
        ):
            c_tok = tokenizer(
                prompt + _to_text(chosen) + tokenizer.eos_token,
                truncation=True, max_length=max_length, padding=False,
            )
            r_tok = tokenizer(
                prompt + _to_text(rejected) + tokenizer.eos_token,
                truncation=True, max_length=max_length, padding=False,
            )
            out["chosen_input_ids"].append(c_tok["input_ids"])
            out["chosen_attention_mask"].append(c_tok["attention_mask"])
            out["rejected_input_ids"].append(r_tok["input_ids"])
            out["rejected_attention_mask"].append(r_tok["attention_mask"])
        return out

    keep = ["chosen_input_ids", "chosen_attention_mask", "rejected_input_ids", "rejected_attention_mask"]
    cols_remove = [c for c in dataset.column_names if c not in keep]
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=cols_remove)
    return dataset


# ── Format detection & extraction ──

def _detect_format(columns: List[str]) -> str:
    """Detect the dataset format from column names."""
    cols = set(columns)
    if "messages" in cols:
        return "messages"
    if "instruction" in cols and "output" in cols:
        return "instruction_output"
    if "instruction" in cols and "response" in cols:
        return "instruction_response"
    if "query" in cols and "response" in cols:
        return "query_response"
    if "prompt" in cols and "completion" in cols:
        return "prompt_completion"
    if "question" in cols and "answer" in cols:
        return "question_answer"
    if "chosen" in cols and "rejected" in cols:
        return "dpo"
    # Fallback: try common single-text patterns
    raise ValueError(
        f"Cannot detect dataset format. Columns: {columns}. "
        "Expected one of: [instruction+output, query+response, messages, prompt+completion]"
    )


def _extract_instruction_response(examples: dict, fmt: str) -> Tuple[List[str], List[str]]:
    """Extract (instruction, response) pairs from a batch of examples."""
    instructions = []
    responses = []

    if fmt == "messages":
        # Tulu V2 / chat format: messages = [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
        for msgs in examples["messages"]:
            user_parts = []
            assistant_parts = []
            for m in msgs:
                role = m.get("role", m.get("from", ""))
                content = m.get("content", m.get("value", ""))
                if role in ("user", "human"):
                    user_parts.append(content)
                elif role in ("assistant", "gpt", "bot"):
                    assistant_parts.append(content)
            inst = "\n".join(user_parts) if user_parts else ""
            resp = "\n".join(assistant_parts) if assistant_parts else ""
            instructions.append(inst)
            responses.append(resp)

    elif fmt == "instruction_output":
        instructions = list(examples["instruction"])
        responses = list(examples["output"])

    elif fmt == "instruction_response":
        instructions = list(examples["instruction"])
        responses = list(examples["response"])

    elif fmt == "query_response":
        instructions = list(examples["query"])
        responses = list(examples["response"])

    elif fmt == "prompt_completion":
        instructions = list(examples["prompt"])
        responses = list(examples["completion"])

    elif fmt == "question_answer":
        instructions = list(examples["question"])
        responses = list(examples["answer"])

    else:
        raise ValueError(f"Unknown format: {fmt}")

    return instructions, responses


def _get_prompt_func(template_name: str):
    """Return a function that formats instruction into a prompt."""
    if template_name == "tulu":
        return lambda inst: TULU_PROMPT_TEMPLATE.format(instruction=inst)
    elif template_name == "code":
        return lambda inst: CODE_PROMPT_DEEPSEEK.format(instruction=inst)
    elif template_name == "mistral":
        return lambda inst: CODE_PROMPT_MISTRAL.format(instruction=inst)
    elif template_name == "math":
        return lambda inst: MATH_PROMPT.format(instruction=inst)
    else:
        return lambda inst: inst  # raw instruction


# ── Dataset file loading ──

def _load_dataset(path: str, split: str) -> Dataset:
    """Load dataset from local path or HuggingFace hub. Supports glob patterns."""
    expanded = path
    if "*" in path or "?" in path:
        expanded = sorted(glob_mod.glob(path))
        if not expanded:
            raise FileNotFoundError(f"No files match glob: {path}")

    if isinstance(expanded, list):
        # Load multiple files
        if all(f.endswith(".jsonl") or f.endswith(".json") for f in expanded):
            return load_dataset("json", data_files=expanded, split="train")
        elif all(f.endswith(".parquet") for f in expanded):
            return load_dataset("parquet", data_files=expanded, split="train")
        else:
            return load_dataset("json", data_files=expanded, split="train")
    else:
        if expanded.endswith((".jsonl", ".json")):
            return load_dataset("json", data_files=expanded, split="train")
        elif expanded.endswith(".parquet"):
            return load_dataset("parquet", data_files=expanded, split="train")
        else:
            return load_dataset(expanded, split=split)


# ── Data collator ──

class SFTDataCollator:
    """Pads SFT sequences to the longest in the batch."""

    def __init__(self, tokenizer: PreTrainedTokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100,
        )
        attention_mask = (input_ids != pad_id).long()

        # Pad to multiple_of for tensor core efficiency
        m = self.pad_to_multiple_of
        if m and input_ids.size(1) % m != 0:
            pad = m - input_ids.size(1) % m
            input_ids = torch.nn.functional.pad(input_ids, (0, pad), value=pad_id)
            labels = torch.nn.functional.pad(labels, (0, pad), value=-100)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad), value=0)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

