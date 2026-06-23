"""
Code generation evaluation: HumanEval(+) and MBPP(+).
Uses EvalPlus official evaluate + training-consistent prompt.
"""

from __future__ import annotations
import os, sys, json, re, argparse, io, logging
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

OUR_PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n"


def generate_code(model, tokenizer, prompt, max_tokens=512, temperature=0.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        if temperature > 0:
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature,
                do_sample=True, top_p=0.95, pad_token_id=tokenizer.eos_token_id,
            )
        else:
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=0.0,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def extract_code(text: str) -> str:
    if '```' in text:
        blocks = re.findall(r'```(?:python)?\s*(.*?)```', text, re.DOTALL)
        return blocks[0].strip() if blocks else text.strip()
    return text.strip()


def generate_and_save(model, tokenizer, problems, prompt_template, jsonl_path,
                      num_samples=1, temperature=0.0, desc="Generate"):
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    with open(jsonl_path, "w") as f:
        for task_id, p in tqdm(list(problems.items()), desc=desc):
            prompt = prompt_template.format(instruction=p["prompt"])
            for _ in range(num_samples):
                raw = generate_code(model, tokenizer, prompt, temperature=temperature)
                code = extract_code(raw)
                f.write(json.dumps({"task_id": task_id, "completion": code}) + "\n")
    return jsonl_path


def run_evalplus(jsonl_path: str, dataset: str) -> dict:
    from evalplus import evaluate as ep_evaluate
    buf = io.StringIO()
    try:
        with __import__('contextlib').redirect_stdout(buf):
            # 兼容不同 EvalPlus 版本
            import inspect
            sig = inspect.signature(ep_evaluate.evaluate)
            params = list(sig.parameters.keys())
            if 'samples' in params:
                ep_evaluate.evaluate(samples=jsonl_path, dataset=dataset, parallel=1)
            else:
                ep_evaluate.evaluate(jsonl_path, dataset=dataset)
    except Exception as e:
        print(f"[EvalPlus ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {}
    text = buf.getvalue()
    print(text)

    results = {}
    prev_line = ''
    for line in text.split('\n'):
        line = line.strip()
        m = re.search(r'pass@1:\s*([\d.]+)', line)
        if m:
            score = float(m.group(1)) * 100
            pl = prev_line.lower()
            if 'humaneval+' in pl:
                results['HumanEval+'] = round(score, 1)
            elif 'humaneval' in pl:
                results['HumanEval'] = round(score, 1)
            elif 'mbpp+' in pl:
                results['MBPP+'] = round(score, 1)
            elif 'mbpp' in pl:
                results['MBPP'] = round(score, 1)
        prev_line = line
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+", default=["humaneval"])
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--eval_only", type=str, default="",
                        help="跳过生成，直接评已有 jsonl 文件")
    args = parser.parse_args()

    if args.eval_only:
        print(f"Eval-only mode: evaluating {args.eval_only}")
        from types import SimpleNamespace
        from evalplus.evaluate import evaluate as ep_evaluate
        flags = SimpleNamespace(
            dataset="humaneval",
            samples=args.eval_only,
            base_only=False,
            parallel=1,
            i_just_wanna_run=False,
            test_details=False,
            min_time_limit=0.2,
            gt_time_limit_factor=4.0,
            mini=False,
            noextreme=False,
        )
        ep_evaluate(flags)
        return

    load_kwargs = {"torch_dtype": torch.bfloat16}
    if args.use_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    if not args.use_8bit:
        model = model.cuda()
    model.eval()

    template = OUR_PROMPT
    all_results = {}
    tmp = Path(args.model_path) / "evalplus_temp"
    tmp.mkdir(parents=True, exist_ok=True)

    if "humaneval" in args.tasks:
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
        path = str(tmp / "humaneval_samples.jsonl")
        generate_and_save(model, tokenizer, problems, template, path,
                          num_samples=args.num_samples, temperature=args.temperature,
                          desc="HumanEval")
        res = run_evalplus(path, "humaneval")
        all_results.update(res)

    if "mbpp" in args.tasks:
        from evalplus.data import get_mbpp_plus
        problems = get_mbpp_plus()
        path = str(tmp / "mbpp_samples.jsonl")
        generate_and_save(model, tokenizer, problems, template, path,
                          num_samples=args.num_samples, temperature=args.temperature,
                          desc="MBPP")
        res = run_evalplus(path, "mbpp")
        all_results.update(res)

    valid = [v for v in all_results.values() if v is not None]
    avg = sum(valid) / len(valid) if valid else 0

    print()
    print("=" * 50)
    print("  Code Generation Results (EvalPlus)")
    print("=" * 50)
    for k, v in all_results.items():
        print(f"  {k:20s}: {v:.1f}")
    print(f"  {'Average':20s}: {avg:.1f}")
    print("=" * 50)

    result_file = os.path.join(args.model_path, "eval_results.json")
    all_results["Average"] = avg
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved to {result_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
