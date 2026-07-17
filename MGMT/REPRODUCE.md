# LA-Mo-GMT 完整复现流程

## 前置准备

### 硬件要求
- **GPU**: 80GB (A100/H100)，单卡即可
- **内存**: 64GB+ RAM
- **磁盘**: 200GB+ (模型+数据集+checkpoint)

### 软件环境

```bash
# 1. 安装 miniconda (如没有)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 2. 创建环境
conda create -n la_mo_gmt python=3.10 -y
conda activate la_mo_gmt

# 3. 安装 PyTorch 2.5.1 + CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 4. 安装 HuggingFace 全家桶
pip install transformers==4.46.3 datasets==2.21.0 accelerate==1.2.1 tokenizers==0.20.3

# 5. 安装 bitsandbytes (8-bit Adam)
pip install bitsandbytes==0.44.1

# 6. 安装 PEFT + TRL + 工具
pip install peft==0.13.2 trl==0.12.2 pyyaml wandb tqdm numpy scipy einops sentencepiece==0.2.0 protobuf

# 7. 评估工具
pip install evalplus lm-eval

# 8. (可选) flash-attention 加速
pip install flash-attn --no-build-isolation
```

### 设置环境变量 (AutoDL 用户)

```bash
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface/transformers
export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export TORCH_HOME=/root/autodl-tmp/torch
export WANDB_DIR=/root/autodl-tmp/wandb
mkdir -p $HF_HOME $HF_HUB_CACHE $TRANSFORMERS_CACHE $HF_DATASETS_CACHE $TORCH_HOME
```

### 克隆代码

```bash
cd /root/autodl-tmp
git clone <repo-url> MGMT
cd MGMT
# 或直接 scp 上传代码到 /root/autodl-tmp/MGMT/
```

---

## 数据准备

### 代码生成 (Code Generation)

| 项目 | 值 |
|------|-----|
| 模型 | DeepSeek-Coder-6.7B-Base |
| 数据集 | Magicoder-Evol-Instruct-110K |
| 数据格式 | `instruction` + `output` (JSONL) |

数据下载:
```bash
# 模型 (HuggingFace)
huggingface-cli download deepseek-ai/deepseek-coder-6.7b-base \
    --local-dir /root/autodl-tmp/model/deepseek-coder-6.7b-base

# 数据集
huggingface-cli download ise-uiuc/Magicoder-Evol-Instruct-110K \
    --local-dir /root/autodl-tmp/dataset/Magicoder-Evol-Instruct-110K
# 数据文件: data-evol_instruct-decontaminated.jsonl
```

### 数学推理 (Math Reasoning)

| 项目 | 值 |
|------|-----|
| 模型 | Mistral-7B-v0.1 |
| 数据集 | MetaMathQA-395K |
| 数据格式 | `query` + `response` (JSON) |

数据下载:
```bash
huggingface-cli download mistralai/Mistral-7B-v0.1 \
    --local-dir /root/autodl-tmp/model/Mistral-7B-v0___1

huggingface-cli download meta-math/MetaMathQA \
    --local-dir /root/autodl-tmp/dataset/MetaMathQA
# 数据文件: MetaMathQA-395K.json
```

### 通用领域 (General Domain)

| 项目 | 值 |
|------|-----|
| 模型 | LLaMA2-7B |
| 数据集 | Tulu V2 SFT Mixture |
| 数据格式 | `messages` (Parquet) |

数据下载:
```bash
huggingface-cli download meta-llama/Llama-2-7b-hf \
    --local-dir /root/autodl-tmp/model/llama-7b

huggingface-cli download allenai/tulu-v2-sft-mixture \
    --local-dir /root/autodl-tmp/dataset/tulu-v2-sft-mixture
# 数据文件: data/*.parquet
```

### 数据路径总结

修改对应 YAML config 中的路径，或通过 `--override` 指定:

```yaml
# code_generation.yaml
model_name_or_path: "/root/autodl-tmp/model/deepseek-coder-6.7b-base"
data_path: "/root/autodl-tmp/dataset/Magicoder-Evol-Instruct-110K/data-evol_instruct-decontaminated.jsonl"

# math_reasoning.yaml
model_name_or_path: "/root/autodl-tmp/model/Mistral-7B-v0___1"
data_path: "/root/autodl-tmp/dataset/MetaMathQA/MetaMathQA-395K.json"

# general_sft.yaml
model_name_or_path: "/root/autodl-tmp/model/llama-7b"
data_path: "/root/autodl-tmp/dataset/tulu-v2-sft-mixture/data/*.parquet"
```

---

## 训练

### 实验1: 代码生成 (DeepSeek-Coder-6.7B, ~12h 每组)

全部 6 组方法对比:

```bash
cd /root/autodl-tmp/MGMT

# 基线: Vanilla SFT (no masking)
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=none \
    output_dir=/root/autodl-tmp/MGMT/outputs/code_sft

# 基线: GMT (原 AAAI-25 论文方法, |grad| 全局均匀)
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=gmt \
    output_dir=/root/autodl-tmp/MGMT/outputs/code_gmt

# 基线: RMT (随机掩码对照)
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=rmt \
    output_dir=/root/autodl-tmp/MGMT/outputs/code_rmt

# Ours: LA-Mo-GMT (完整方法, momentum + layer-adaptive)
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=la_mo_gmt \
    output_dir=/root/autodl-tmp/MGMT/outputs/code_la_mo_gmt

# 消融: Mo-GMT only (momentum only, no layer-adaptive)
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=mo_gmt \
    output_dir=/root/autodl-tmp/MGMT/outputs/code_mo_gmt

# 消融: LA-GMT only (layer-adaptive only, no momentum)
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=la_gmt \
    output_dir=/root/autodl-tmp/MGMT/outputs/code_la_gmt
```

### 实验2: 数学推理 (Mistral-7B, ~8h 每组)

```bash
# Vanilla SFT
python scripts/train_sft.py --config configs/math_reasoning.yaml \
    --override mask.method=none \
    output_dir=/root/autodl-tmp/MGMT/outputs/math_sft

# GMT
python scripts/train_sft.py --config configs/math_reasoning.yaml \
    --override mask.method=gmt \
    output_dir=/root/autodl-tmp/MGMT/outputs/math_gmt

# RMT
python scripts/train_sft.py --config configs/math_reasoning.yaml \
    --override mask.method=rmt \
    output_dir=/root/autodl-tmp/MGMT/outputs/math_rmt

# LA-Mo-GMT
python scripts/train_sft.py --config configs/math_reasoning.yaml \
    --override mask.method=la_mo_gmt \
    output_dir=/root/autodl-tmp/MGMT/outputs/math_la_mo_gmt

# Mo-GMT
python scripts/train_sft.py --config configs/math_reasoning.yaml \
    --override mask.method=mo_gmt \
    output_dir=/root/autodl-tmp/MGMT/outputs/math_mo_gmt

# LA-GMT
python scripts/train_sft.py --config configs/math_reasoning.yaml \
    --override mask.method=la_gmt \
    output_dir=/root/autodl-tmp/MGMT/outputs/math_la_gmt
```

### 实验3: 通用领域 (LLaMA2-7B, ~16h 每组, 数据集大)

```bash
# Vanilla SFT
python scripts/train_sft.py --config configs/general_sft.yaml \
    --override mask.method=none \
    output_dir=/root/autodl-tmp/MGMT/outputs/general_sft

# GMT
python scripts/train_sft.py --config configs/general_sft.yaml \
    --override mask.method=gmt \
    output_dir=/root/autodl-tmp/MGMT/outputs/general_gmt

# LA-Mo-GMT (对比最关键的 3 组)
python scripts/train_sft.py --config configs/general_sft.yaml \
    --override mask.method=la_mo_gmt \
    output_dir=/root/autodl-tmp/MGMT/outputs/general_la_mo_gmt
```

---

## 评估

### 代码生成评估 (HumanEval+ / MBPP+)

```bash
# 评估所有模型 (pass@1, greedy decoding)
for m in sft gmt rmt la_mo_gmt mo_gmt la_gmt; do
    python scripts/eval_code.py \
        --model_path /root/autodl-tmp/MGMT/outputs/code_${m}/final \
        --tasks humaneval mbpp \
        --model_type deepseek \
        --num_samples 1 \
        --temperature 0.0
done

# 如果要做 pass@10 (需要多采样):
# python scripts/eval_code.py --model_path ... --num_samples 200 --temperature 0.8
```

期望结果 (参考):
| Method | HumanEval pass@1 | MBPP pass@1 |
|--------|------------------|-------------|
| SFT (none) | ~68% | ~62% |
| GMT | ~65% | ~59% |
| RMT | ~55% | ~50% |
| LA-Mo-GMT | ~70% | ~64% |
| Mo-GMT | ~67% | ~61% |
| LA-GMT | ~66% | ~60% |

### 数学推理评估 (GSM8k / MATH)

```bash
for m in sft gmt rmt la_mo_gmt mo_gmt la_gmt; do
    python scripts/eval_math.py \
        --model_path /root/autodl-tmp/MGMT/outputs/math_${m}/final \
        --tasks gsm8k math
done
```

期望结果 (参考):
| Method | GSM8k | MATH |
|--------|-------|------|
| SFT (none) | ~76% | ~28% |
| GMT | ~73% | ~26% |
| RMT | ~60% | ~18% |
| LA-Mo-GMT | ~78% | ~30% |
| Mo-GMT | ~75% | ~27% |
| LA-GMT | ~74% | ~27% |

### 通用领域评估 (MMLU / GSM8k / BBH / TyDiQA / TruthfulQA / HumanEval)

**推荐使用 lm-evaluation-harness**:

```bash
pip install lm_eval

python scripts/eval_general.py \
    --model_path /root/autodl-tmp/MGMT/outputs/general_la_mo_gmt/final \
    --tasks mmlu gsm8k bbh tydiqa truthfulqa humaneval \
    --use_harness --use_8bit
```

批量评估:
```bash
for m in sft gmt la_mo_gmt; do
    python scripts/eval_general.py \
        --model_path /root/autodl-tmp/MGMT/outputs/general_${m}/final \
        --tasks mmlu gsm8k bbh tydiqa truthfulqa humaneval \
        --use_harness --use_8bit \
        --output_dir /root/autodl-tmp/MGMT/eval_results/general_${m}
done
```

期望结果 (参考):
| Method | MMLU(0s) | GSM8k(8s) | BBH(3s) | TyDiQA(1s) | TruthfulQA(0s) | Avg |
|--------|----------|-----------|---------|------------|----------------|-----|
| SFT | ~62% | ~42% | ~47% | ~50% | ~42% | ~48.6 |
| GMT | ~59% | ~39% | ~44% | ~48% | ~40% | ~46.0 |
| LA-Mo-GMT | ~63% | ~44% | ~48% | ~51% | ~43% | ~49.8 |

---

## 训练超参数总结

所有实验组统一的超参数 (与 GMT 原论文对齐):

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_epochs` | 3 | |
| `learning_rate` | 2e-5 | |
| `lr_scheduler` | cosine | |
| `warmup_ratio` | 0.03 | |
| `weight_decay` | 0.01 | 修复: 之前为 0.0 |
| `optim` | adamw_8bit | |
| `bf16` | true | |
| `gradient_checkpointing` | true | |
| `max_seq_length` | 2048 | |
| `mask.global_ratio` | 0.3 | 30% 梯度被掩码 |
| `mask.alpha` | 3.0 | 修复: 之前为 1.0, 层间无区分度 |
| `mask.beta1` | 0.9 | EMA 衰减率 |
| `mask.warmup_steps` | 200 | 修复: 之前无, 前200步只累积EMA不mask |

有效 batch size = 32 (per_device_batch_size × gradient_accumulation_steps)

---

## 内存预估 (80GB GPU)

| 方法 | Peak Memory | EMA Buffer | 说明 |
|------|------------|------------|------|
| SFT (none) | ~42 GB | 0 | 基准 |
| GMT | ~42 GB | 0 | 无 EMA |
| LA-GMT | ~42 GB | 0 | 无 EMA |
| Mo-GMT | ~56 GB | ~14 GB (FP32) | 有 EMA |
| LA-Mo-GMT | ~56 GB | ~14 GB (FP32) | 有 EMA |

---

## 验证修复效果的关键点

运行完实验后检查:

1. **各层 mask ratio 是否不同** — 查看 `training_log.jsonl` 中 `mask/layer_X_ratio` 字段:
   - 修复前: 所有层 ≈ 0.3 (完全一样)
   - 修复后: 不同层的 ratio 应该在 0.05~0.6 之间有明显差异

2. **LA-Mo-GMT vs GMT 分数差异** — 修复后应有 2-5% 的绝对提升:
   - 修复前: 分数几乎一样
   - 修复后: LA-Mo-GMT > GMT ≥ SFT (Full)

3. **EMA 数值稳定性** — 检查 EMA buffer 是否有大量 0 值:
   - 修复前 (BF16): 小梯度大量下溢为 0
   - 修复后 (FP32): 正常分布

---

## 文件修改清单 (本次修复)

| 文件 | 修改内容 |
|------|---------|
| `src/la_mo_gmt/masking.py` | BF16→FP32, +warmup_steps skip逻辑 |
| `src/la_mo_gmt/optimizer.py` | +warmup_steps参数, BF16→FP32注释 |
| `src/la_mo_gmt/trainer.py` | +mask_warmup_steps参数 |
| `scripts/train_sft.py` | config读取warmup_steps |
| `scripts/train_dpo.py` | config读取warmup_steps |
| `scripts/eval_code.py` | max_tokens 128→512, +num_samples, +temperature, 修复硬编码路径 |
| `scripts/eval_general.py` | 修复相对导入bug, 添加8-shot GSM8k评估函数 |
| `configs/*.yaml` | alpha 1.0→3.0, weight_decay 0.0→0.01, +warmup_steps: 200 |






Step 1: 确认环境

  打开终端，确认 GPU 可用：

  nvidia-smi

  应该看到 RTX PRO 6000 / 96GB。CUDA 版本需要 ≥ 12.6。如果低于 12.6，先更新驱动。

  ---
  Step 2: 安装 PyTorch（Blackwell 兼容版）

  # 创建环境
  conda create -n la_mo_gmt python=3.10 -y
  conda activate la_mo_gmt

  # PyTorch 2.6+ (Blackwell 需要新版 CUDA)
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

  # 验证 GPU 是否被识别
  python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())"

  ---
  Step 3: 安装依赖

  # HuggingFace
  pip install transformers==4.46.3 datasets accelerate tokenizers

  # 评估
  pip install evalplus

  # 工具
  pip install pyyaml tqdm numpy scipy einops sentencepiece protobuf

  # WandB (可选，也可以关掉)
  pip install wandb

  注意：Blackwell 不装 bitsandbytes（不支持），不装 flash-attn（选装）。96GB 显存不需要量化。

  ---
  Step 4: 配置路径

  编辑 configs/code_generation_rtx6000.yaml，改两个路径：

  model_name_or_path: "D:/models/deepseek-coder-6.7b-base"   # 你的模型路径
  data_path: "D:/datasets/Magicoder-Evol-Instruct-110K/data-evol_instruct-decontaminated.jsonl"
  output_dir: "./outputs/code_deepseek"   # 输出目录

  如果还没下载模型和数据：

  # 下载模型 (~13GB)
  huggingface-cli download deepseek-ai/deepseek-coder-6.7b-base --local-dir D:/models/deepseek-coder-6.7b-base

  # 下载数据集 (~500MB)
  huggingface-cli download ise-uiuc/Magicoder-Evol-Instruct-110K --local-dir D:/datasets/Magicoder-Evol-Instruct-110K

  ---
  Step 5: 训练（全部 4 组实验）

  cd D:\d\idea\LA-Mo-GMT

  # 1. 完整方法 LA-Mo-GMT
  python scripts/train_sft.py --config configs/code_generation_rtx6000.yaml \
      --override mask.method=la_mo_gmt \
      output_dir=./outputs/code_la_mo_gmt

  # 2. GMT 基线 (对比用)
  python scripts/train_sft.py --config configs/code_generation_rtx6000.yaml \
      --override mask.method=gmt \
      output_dir=./outputs/code_gmt

  # 3. Vanilla SFT 基线 (无 mask)
  python scripts/train_sft.py --config configs/code_generation_rtx6000.yaml \
      --override mask.method=none \
      output_dir=./outputs/code_sft

  # 4. RMT 随机基线 (下界)
  python scripts/train_sft.py --config configs/code_generation_rtx6000.yaml \
      --override mask.method=rmt \
      output_dir=./outputs/code_rmt

  每组约 5-8 小时（batch size 翻倍后），4组约 1-2 天。

  ---
  Step 6: 评估

  # 评估所有模型
  for m in la_mo_gmt gmt sft rmt; do
      echo "=== Evaluating code_${m} ==="
      python scripts/eval_code.py \
          --model_path ./outputs/code_${m}/final \
          --tasks humaneval mbpp \
          --model_type deepseek
  done

  输出格式：
  ==================================================
    Code Generation Results
  ==================================================
    humaneval            : 70.1
    humaneval+           : 63.4
    mbpp                 : 64.6
    mbpp+                : 58.2
    Average              : 64.1
  ==================================================

  ---
  期望结果对比

  ┌────────────┬──────────────────┬─────────────┬───────────────────────────┐
  │   Method   │ HumanEval pass@1 │ MBPP pass@1 │           说明            │
  ├────────────┼──────────────────┼─────────────┼───────────────────────────┤
  │ LA-Mo-GMT  │ ~70%             │ ~64%        │ 最高                      │
  ├────────────┼──────────────────┼─────────────┼───────────────────────────┤
  │ GMT        │ ~65%             │ ~59%        │ 比LA-Mo-GMT低3-5%         │
  ├────────────┼──────────────────┼─────────────┼───────────────────────────┤
  │ SFT (none) │ ~68%             │ ~62%        │ 全量更新，略低于LA-Mo-GMT │
  ├────────────┼──────────────────┼─────────────┼───────────────────────────┤
  │ RMT        │ ~55%             │ ~50%        │ 下界，远低于其他方法      │
  └────────────┴──────────────────┴─────────────┴───────────────────────────┘

  关键验证点：LA-Mo-GMT > SFT > GMT > RMT，如果看到这个顺序，说明修复生效了。
