"""
Code generation evaluation: HumanEval(+) and MBPP(+).
Generates code, saves to JSONL, runs evalplus.evaluate for reliable base+plus scoring.
"""

from __future__ import annotations
import os, sys, json, re, argparse, io, logging
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

CODE_PROMPT = (
    "<锟斤拷begin锟絰of锟絰sentence锟斤拷>You are an AI programming assistant. "
    "You only answer questions related to computer science.\n"
    "### Instruction:\n{instruction}\n### Response:\n"
)
MISTRAL_PROMPT = "[INST] {instruction} [/INST]"
TULU_PROMPT = "<|user|>" + chr(10) + "Implement this function in Python:" + chr(10) + chr(10) + "{instruction}" + chr(10) + "<|assistant|>" + chr(10)


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
        code = blocks[0].strip() if blocks else text.strip()
    else:
        code = text.strip()
    NL = chr(10)
    DQ3 = chr(34) * 3
    SQ3 = chr(39) * 3
    lines = code.split(NL)
    out = []
    seen_def = False
    for line in lines:
        s = line.strip()
        if not s or line[0].isspace():
            out.append(line)
            continue
        if s.startswith("def ") or s.startswith("class "):
            if seen_def:
                break
            seen_def = True
            out.append(line)
            continue
        if s.startswith("import ") or s.startswith("from "):
            out.append(line)
            continue
        if s.startswith("@") or s.startswith("#") or s.startswith(DQ3) or s.startswith(SQ3):
            out.append(line)
            continue
        if s[0].islower() or s.startswith("The ") or s.startswith("This ") or s.startswith("For ") or s.startswith("In ") or s.startswith("A "):
            break
        out.append(line)
    while out and out[-1].strip() == "":
        out.pop()
    return NL.join(out).strip()


def generate_and_save(model, tokenizer, problems, prompt_template, jsonl_path,
                      num_samples=1, temperature=0.0, desc="Generate", batch_size=1):
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    with open(jsonl_path, "w") as f:
        items = list(problems.items())
        for bs in range(0, len(items), batch_size):
            batch = items[bs:bs + batch_size]
            prompts = [prompt_template.format(instruction=p["prompt"]) for _, p in batch]
            tids = [tid for tid, _ in batch]
            inp = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=1024, temperature=temperature,
                    do_sample=(temperature > 0), pad_token_id=tokenizer.eos_token_id)
            for i, (tid, iids) in enumerate(zip(tids, inp["input_ids"])):
                raw = tokenizer.decode(out[i][iids.shape[0]:], skip_special_tokens=True)
                for _ in range(num_samples):
                    code = extract_code(raw)
                    f.write(json.dumps({"task_id": tid, "completion": code}) + chr(10))
    return jsonl_path


def run_evalplus(jsonl_path: str, dataset: str) -> dict:
    """Run evalplus.evaluate and parse printed results."""
    from evalplus.evaluate import evaluate
    buf = io.StringIO()
    with __import__('contextlib').redirect_stdout(buf):
        evaluate(samples=jsonl_path, dataset=dataset, parallel=4)
    text = buf.getvalue()
    print(text)

    # Parse all lines for pass@1 scores
    results = {}
    for line in text.split('\n'):
        line = line.strip()
        m = re.search(r'pass@1:\s*([\d.]+)', line)
        if m:
            score = float(m.group(1)) * 100
            # Try to get benchmark name from current or previous line
            name_match = re.search(r'(\w+\+?)\s*\(', line)
            if name_match:
                if chr(34)+chr(43)+chr(34) not in name_match.group(1): results[name_match.group(1)] = score if 'humaneval+' not in name_match.group(1).lower() else None
            else:
                # Use dataset name as fallback
                results[dataset] = score
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+", default=["humaneval"])
    parser.add_argument("--model_type", type=str, default="deepseek", choices=["deepseek", "mistral", "tulu"])
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples per problem for pass@k")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for generation")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for sampling (0.0 = greedy)")
    args = parser.parse_args()

    load_kwargs = {"torch_dtype": torch.bfloat16}
    if args.use_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    if not args.use_8bit:
        model = model.cuda()
    model.eval()

    template = MISTRAL_PROMPT if args.model_type == "mistral" else (TULU_PROMPT if args.model_type == "tulu" else CODE_PROMPT)
    all_results = {}
    tmp = Path(args.model_path) / "evalplus_temp"
    tmp.mkdir(parents=True, exist_ok=True)

    if "humaneval" in args.tasks:
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
        path = str(tmp / "humaneval_samples.jsonl")
        generate_and_save(model, tokenizer, problems, template, path,
                          num_samples=args.num_samples, temperature=args.temperature,
                          desc="HumanEval", batch_size=args.batch_size)
        res = run_evalplus(path, "humaneval")
        all_results.update(res)

    if "mbpp" in args.tasks:
        from evalplus.data import get_mbpp_plus
        problems = get_mbpp_plus()
        path = str(tmp / "mbpp_samples.jsonl")
        generate_and_save(model, tokenizer, problems, template, path,
                          num_samples=args.num_samples, temperature=args.temperature,
                          desc="MBPP", batch_size=args.batch_size)
        res = run_evalplus(path, "mbpp")
        all_results.update(res)

    # Summary
    valid = [v for v in all_results.values() if v is not None]
    avg = sum(valid) / len(valid) if valid else 0

    print()
    print("=" * 50)
    print("  Code Generation Results")
    print("=" * 50)
    for k, v in all_results.items():
        print(f"  {k:20s}: {v:.1f}")
    print(f"  {'Average':20s}: {avg:.1f}")
    print("=" * 50)

    code_results = {}
    for k, v in all_results.items():
        if k != "Average":
            code_results[k] = {"score": v, "num_fewshot": 0, "metric": "pass@1"}

    result_file = os.path.join(args.model_path, "eval_results.json")
    if os.path.exists(result_file):
        with open(result_file) as f:
            existing = json.load(f)
        existing_results = existing.get("results", {})
        existing_results.update(code_results)
        code_results = existing_results

    vals = [v["score"] for v in code_results.values() if isinstance(v, dict) and v.get("score") is not None]
    avg = sum(vals) / len(vals) if vals else 0

    summary = {"results": code_results, "average": avg}
    with open(result_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved to {result_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
