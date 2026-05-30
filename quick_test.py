"""快速验证 HF 镜像 + 模型推理是否正常"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
import torch

print("=" * 50)
print("1. 测试 HF 镜像连通性...")
print("=" * 50)

tests = {
    "openai_humaneval": "openai_humaneval",
    "mbpp": "mbpp",
    "evalplus/humaneval_plus": "evalplus/humaneval_plus",
    "evalplus/mbpp_plus": "evalplus/mbpp_plus",
}

for name, path in tests.items():
    try:
        ds = load_dataset(path, split="test")
        print(f"  ✅ {name}: {len(ds)} 条数据")
    except Exception as e:
        print(f"  ❌ {name}: {str(e)[:100]}")

print()
print("=" * 50)
print("2. 测试模型推理（检查是否生成正常代码）...")
print("=" * 50)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 用最小的推理测试
model_name = "/root/autodl-tmp/model/Mistral-7B-v0___1"
if not os.path.exists(model_name):
    model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

prompt = "def add(a, b):\n    "
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.0, top_k=1, pad_token_id=tokenizer.eos_token_id)

input_len = inputs.input_ids.shape[1]
generated_ids = outputs[0][input_len:]
generated = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"  Prompt:  {prompt}")
print(f"  Generated: {generated}")

if "return" in generated.lower() and len(generated.strip()) > 5:
    print("  ✅ 模型生成正常")
else:
    print("  ❌ 模型生成异常！")

del model
torch.cuda.empty_cache()
print()
print("全部检测完成！")
