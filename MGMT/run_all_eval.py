"""Run all evaluations for 3 DPO models. Errors are logged, not fatal."""
import subprocess, os, sys

os.chdir("/root/autodl-tmp/MGMT")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

models = [
    ("DPO MAGM", "./outputs/dpo_llama2_7b_tulu_magm_mask30/final"),
    ("DPO GMT",  "./outputs/dpo_llama2_7b_tulu_gmt_mask30/final"),
    ("DPO nomask", "./outputs/dpo_llama2_7b_tulu_nomask/final"),
]

commands = []
for name, path in models:
    commands += [
        (f"{name}: MMLU+GSM8k+BBH+TruthfulQA",
         f"python scripts/eval_general.py --model_path {path} --tasks mmlu gsm8k bbh truthfulqa_mc2 --use_harness --batch_size 32"),
        (f"{name}: TyDiQA",
         f"python scripts/eval_general.py --model_path {path} --tasks tydiqa --batch_size 8"),
        (f"{name}: HumanEval",
         f"python scripts/eval_code.py --model_path {path} --tasks humaneval --model_type tulu --batch_size 8"),
    ]

failed = []
for desc, cmd in commands:
    print(f"\n{'='*60}")
    print(f"RUNNING: {desc}")
    print(f"CMD: {cmd}")
    print(f"{'='*60}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"FAILED (code {r.returncode}): {desc}")
        failed.append(desc)
    else:
        print(f"OK: {desc}")

print(f"\n{'='*60}")
print(f"DONE. {len(commands)-len(failed)}/{len(commands)} passed")
if failed:
    print("FAILED:")
    for f in failed:
        print(f"  - {f}")
