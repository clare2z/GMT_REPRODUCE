"""
通用评测脚本 — 加载任意 checkpoint 即可跑分，不依赖训练代码

用法:
    # 评测训练后的 checkpoint（任意算法训出来的都行）
    python eval.py --checkpoint checkpoints/dgmm_v1 --benchmarks humaneval,mbpp

    # 评测 baseline 模型
    python eval.py --model_name mistralai/Mistral-7B-v0.1 --benchmarks humaneval --max_samples 10

    # 全量评测
    python eval.py --checkpoint checkpoints/dgmm_v1 --max_samples 100
"""

import os

# ⚠️ 必须在 import transformers / datasets 之前设置
if os.environ.get("HF_ENDPOINT") is None:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if os.environ.get("HF_HOME") is None:
    os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"

import torch
import re
import logging
import csv
import time
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 代码清洗
# ═══════════════════════════════════════════════════════════════

def _clean_generated_code(text: str) -> str:
    """清洗生成代码：去除 markdown 包装、解释性文字、不完整 docstring 等"""
    code = text.strip()

    # 1. 去除首尾的 markdown 代码块标记
    code = re.sub(r'^```(?:python|python3)?\s*\n?', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n?```\s*$', '', code)

    # 2. 提取完整 markdown 代码块
    md_match = re.search(r'```(?:python)?\s*\n(.*?)\n```', code, re.DOTALL)
    if md_match:
        code = md_match.group(1).strip()

    # 3. 去除开头的解释性文字
    match = re.search(r'\n(def |class |import |from |\nif |\nfor |\nwhile )', code)
    if match and match.start() > 80:
        code = code[match.start() + 1:].strip()

    # 4. 定位第一个 def/class/import
    if not re.match(r'(def |class |import |from |[ \t]+)', code):
        first_def = re.search(r'\n(def |class )', code)
        if first_def and first_def.start() > 50:
            code = code[first_def.start() + 1:].strip()

    # 5. 修复未闭合的三引号
    triple_count = code.count('"""')
    if triple_count % 2 != 0:
        last_triple = code.rfind('"""')
        code = code[:last_triple].rstrip()

    return code


# ═══════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════

def load_model_from_checkpoint(checkpoint_path, device="cuda"):
    """从 checkpoint 目录加载模型和 tokenizer"""
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")
    return model, tokenizer


def load_model_from_hub(model_name, device="cuda"):
    """从 HuggingFace 加载原始模型（baseline 评测）"""
    logger.info(f"Loading baseline model: {model_name}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    logger.info(f"Baseline model loaded. Parameters: {model.num_parameters():,}")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════
# 评测核心（修复了 exec bug）
# ═══════════════════════════════════════════════════════════════

def evaluate_on_benchmark(model, tokenizer, benchmark_name, device="cuda", max_samples=100):
    logger.info(f"Evaluating on {benchmark_name} (max {max_samples} samples)...")

    # ── 加载数据集 ──────────────────────────────────────────
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
    except Exception as e:
        logger.error(f"Failed to load benchmark {benchmark_name}: {e}")
        return 0.0

    total = min(len(dataset), max_samples)
    correct = 0
    t_start = time.time()
    total_generated = 0

    # 检测数据集格式（不同数据集字段名不同）
    first_ex = dataset[0] if isinstance(dataset, list) else dataset[0]
    _keys = set(first_ex.keys()) if isinstance(first_ex, dict) else set(dir(first_ex))

    logger.info(f"  [{benchmark_name}] Starting {total} problems | {datetime.now().strftime('%H:%M:%S')}")

    for i in range(total):
        example = dataset[i]

        # ── 统一字段提取 ──────────────────────────────────────
        # prompt: humanEval=prompt, MBPP=text（自然语言）/ code（函数签名）
        prompt = example.get('prompt', '') or example.get('text', '')
        entry_point = example.get('entry_point', '')

        # MBPP 特殊处理：从 code 提取函数签名当 prompt，保证函数名与 test 匹配
        if benchmark_name in ("mbpp", "mbpp_plus") and not entry_point:
            code = example.get('code', '')
            # 提取 def 开头的第一行作为函数签名
            sig_match = re.match(r'^def\s+\w+\s*\([^)]*\)\s*:', code)
            if sig_match:
                prompt = sig_match.group(0)  # "def func_name(args):"
                entry_point = prompt[4:prompt.index('(')].strip()  # "func_name"

        # test 字段统一提取（不同数据集字段名不同）
        # HumanEval → test(string), MBPP → test_list(list), MBPP+ → assertion(string)
        test = example.get('test', '') or example.get('assertion', '')
        if not test:
            test_list = example.get('test_list', [])
            test_imports = example.get('test_imports', '')
            if test_list:
                test = '\n'.join(test_list)
                if test_imports:
                    test = test_imports + '\n' + test

        # 训练格式一致的 prompt 包装
        full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
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

        generated_code = _clean_generated_code(generated_code)
        if not generated_code.strip():
            continue

        total_generated += 1

        # 前 3 个样本打印（含 test 预览用于调试）
        if i < 3:
            print(f"  [样本 #{i+1}] {benchmark_name}")
            print(f"    prompt[:80]: {prompt[:80]}...")
            print(f"    generated[:120]: {generated_code[:120]}...")
            print(f"    test[:100]: {test[:100] if test else '(EMPTY)'}...")
            print(f"    ----")

        # ── 修复后的评测逻辑 ──────────────────────────────────
        # HumanEval: prompt=函数签名, test=def check(candidate):...
        # MBPP:      prompt=自然语言, test=assert 语句列表
        # HumanEval+: prompt=函数签名, test=测试代码(可能含check调用)
        # MBPP+:     prompt=函数签名, test=测试代码(含check调用)

        try:
            # 路径 1: prompt 是合法 Python（HumanEval 系），拼接后 exec
            exec_globals = {}
            full_code = prompt + "\n" + generated_code + "\n" + test
            if entry_point and 'check' in test and f"check({entry_point})" not in test:
                full_code += f"\ncheck({entry_point})"
            exec(full_code, exec_globals)
            correct += 1
        except Exception:
            try:
                # 路径 2: prompt 是自然语言（MBPP），只用 generated_code + test
                exec(generated_code + "\n" + test, {})
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
    logger.info(f"  [{benchmark_name}] Done | generated: {total_generated}/{total} | "
                f"correct: {correct}/{total} | pass@1={pass_rate:.4f} | "
                f"elapsed={time.time()-t_start:.0f}s")
    return pass_rate


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DGMM Evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Path to trained checkpoint")
    group.add_argument("--model_name", type=str, help="Base model name (baseline evaluation)")
    parser.add_argument("--benchmarks", type=str, default="humaneval,mbpp,humaneval_plus,mbpp_plus",
                        help="Comma-separated benchmark names")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Max samples per benchmark")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results CSV")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    # 确定评测名称
    if args.checkpoint:
        eval_name = os.path.basename(args.checkpoint.rstrip("/"))
    else:
        eval_name = args.model_name.replace("/", "_") + "_baseline"

    logger.info(f"===== DGMM Evaluation: {eval_name} =====")
    logger.info(f"Source: {args.checkpoint or args.model_name}")
    logger.info(f"Benchmarks: {benchmarks} | Max samples: {args.max_samples}")

    # 加载模型
    if args.checkpoint:
        model, tokenizer = load_model_from_checkpoint(args.checkpoint, device)
    else:
        model, tokenizer = load_model_from_hub(args.model_name, device)

    # 跑评测
    results = {}
    for i, benchmark in enumerate(benchmarks):
        logger.info(f">>> [{i+1}/{len(benchmarks)}] Evaluating {benchmark}...")
        score = evaluate_on_benchmark(model, tokenizer, benchmark, device, max_samples=args.max_samples)
        results[benchmark] = score
        logger.info(f">>> [{i+1}/{len(benchmarks)}] {benchmark}: {score:.4f}")

    avg_score = sum(results.values()) / len(results) if results else 0.0
    results['average'] = avg_score

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"eval_{eval_name}_{timestamp}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Benchmark", "Pass@1"])
        for k, v in results.items():
            writer.writerow([k, f"{v:.4f}"])

    # 打印结果
    print("\n" + "=" * 50)
    print(f"Evaluation Results: {eval_name}")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"\nResults saved to: {csv_path}")

    del model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
