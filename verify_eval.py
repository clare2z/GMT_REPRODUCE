"""快速验证：加载已训练模型，用正确格式评测 10 题 HumanEval"""
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
    lines = code.split('\n')
    if lines and '"""' in lines[-1] and not lines[-1].strip().endswith('"""'):
        code = '\n'.join(lines[:-1])
    return code

model_path = "checkpoints/mistralai_Mistral-7B-v0.1_DGMM"
if not os.path.exists(model_path):
    print("❌ 没找到已训练模型，先测 baseline（原版 Mistral-7B）")
    model_path = "/root/autodl-tmp/model/Mistral-7B-v0___1"
    if not os.path.exists(model_path):
        model_path = "mistralai/Mistral-7B-v0.1"

print(f"加载模型: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path, load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, device_map="auto")
model.eval()

dataset = load_dataset("openai_humaneval", split="test")
correct = 0
total = 10

for i in range(total):
    ex = dataset[i]
    prompt = ex["prompt"]
    test = ex["test"]

    # 训练格式包裹
    full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.0, top_k=1,
                                 pad_token_id=tokenizer.eos_token_id)

    gen = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    gen = clean_code(gen)  # 清洗 markdown 和解释文字
    # exec
    try:
        exec(prompt + gen + "\n" + test, {})
        correct += 1
        print(f"✅ #{i+1}")
    except Exception as e:
        print(f"❌ #{i+1} | 生成: {gen[:80]}... | 错误: {str(e)[:60]}")

print(f"\nHumanEval: {correct}/{total} = {correct/total*100:.0f}%")

del model; torch.cuda.empty_cache()
