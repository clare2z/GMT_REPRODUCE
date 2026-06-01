"""对比验证：DGMM训练后模型 vs 原版Mistral-7B baseline"""
import os, re
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

def clean_code(text):
    code = text.strip()
    code = re.sub(r'^```(?:python|python3)?\s*\n?', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n?```\s*$', '', code)
    md = re.search(r'```(?:python)?\s*\n(.*?)\n```', code, re.DOTALL)
    if md: code = md.group(1).strip()
    first_def = re.search(r'\n(def )', code)
    if first_def and first_def.start() > 50:
        code = code[first_def.start()+1:].strip()
    triple_count = code.count('"""')
    if triple_count % 2 != 0:
        last_triple = code.rfind('"""')
        code = code[:last_triple].rstrip()
    return code

def test_model(model, tokenizer, label, total=10):
    dataset = load_dataset("openai_humaneval", split="test")
    correct = 0
    for i in range(total):
        ex = dataset[i]
        prompt = ex["prompt"]
        test = ex["test"]
        full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0, top_k=1,
                                     pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        gen = clean_code(gen)
        try:
            exec(prompt + gen + "\n" + test, {})
            correct += 1
            print(f"[{label}] ✅ #{i+1}")
        except Exception as e:
            print(f"[{label}] ❌ #{i+1} | gen({len(gen)}ch): {repr(gen[:120])} | {str(e)[:50]}")
    print(f"[{label}] HumanEval: {correct}/{total} = {correct/total*100:.0f}%\n")
    return correct

# --- 测 baseline（原版）---
print("=" * 50)
print("1. BASELINE: 原版 Mistral-7B (未经DGMM训练)")
print("=" * 50)
base_path = "/root/autodl-tmp/model/Mistral-7B-v0___1"
if not os.path.exists(base_path):
    base_path = "mistralai/Mistral-7B-v0.1"
btok = AutoTokenizer.from_pretrained(base_path)
if btok.pad_token is None: btok.pad_token = btok.eos_token
bmod = AutoModelForCausalLM.from_pretrained(base_path, load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, device_map="auto")
bmod.eval()
base_score = test_model(bmod, btok, "BASELINE")
del bmod; torch.cuda.empty_cache()

# --- 测 DGMM 训练后 ---
print("=" * 50)
print("2. DGMM: 训练后模型")
print("=" * 50)
ckpt = "checkpoints/mistralai_Mistral-7B-v0.1_DGMM"
if not os.path.exists(ckpt):
    print("⚠ 未找到训练模型，跳过")
    dgmm_score = -1
else:
    dtok = AutoTokenizer.from_pretrained(ckpt)
    if dtok.pad_token is None: dtok.pad_token = dtok.eos_token
    dmod = AutoModelForCausalLM.from_pretrained(ckpt, load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, device_map="auto")
    dmod.eval()
    dgmm_score = test_model(dmod, dtok, "DGMM")
    del dmod; torch.cuda.empty_cache()

print("=" * 50)
print(f"BASELINE: {base_score}/10  |  DGMM: {dgmm_score}/10")
print("=" * 50)
