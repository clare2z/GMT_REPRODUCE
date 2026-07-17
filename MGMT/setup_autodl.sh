#!/bin/bash
# ============================================================
# LA-Mo-GMT AutoDL 完整环境搭建 + 实验流程
# ============================================================
set -e

echo "============================================"
echo " LA-Mo-GMT 环境搭建 (AutoDL)"
echo "============================================"

# ============================================
# Step 1: 环境变量 — 全部指向 /root/autodl-tmp
# ============================================
echo ""
echo "[Step 1/4] 设置环境变量..."

cat >> ~/.bashrc << 'EOF'

# --- LA-Mo-GMT ---
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface/transformers
export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export TORCH_HOME=/root/autodl-tmp/torch
export WANDB_DIR=/root/autodl-tmp/wandb
export TMPDIR=/root/autodl-tmp/tmp
export XDG_CACHE_HOME=/root/autodl-tmp/xdg_cache
EOF

export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface/transformers
export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export TORCH_HOME=/root/autodl-tmp/torch
export WANDB_DIR=/root/autodl-tmp/wandb
export TMPDIR=/root/autodl-tmp/tmp
export XDG_CACHE_HOME=/root/autodl-tmp/xdg_cache

mkdir -p $HF_HOME $HF_HUB_CACHE $TRANSFORMERS_CACHE $HF_DATASETS_CACHE
mkdir -p $TORCH_HOME $WANDB_DIR $TMPDIR $XDG_CACHE_HOME
echo "  环境变量已设置，所有缓存指向 /root/autodl-tmp"

# ============================================
# Step 2: 创建 conda 环境
# ============================================
echo ""
echo "[Step 2/4] 创建 conda 环境 (la_mo_gmt, Python 3.10)..."

source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null || true

if conda env list 2>/dev/null | grep -q "^la_mo_gmt "; then
    echo "  环境已存在，删除重建..."
    conda env remove -n la_mo_gmt -y 2>/dev/null
fi

conda create -n la_mo_gmt python=3.10 -y
conda activate la_mo_gmt

echo "  Python: $(python --version)"

# ============================================
# Step 3: 安装依赖
# ============================================
echo ""
echo "[Step 3/4] 安装 Python 依赖..."

# PyTorch 2.5.1 + CUDA 12.1
echo "  [3.1] PyTorch..."
pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# HuggingFace 全家桶
echo "  [3.2] Transformers / Datasets / Accelerate..."
pip install -q \
    transformers==4.46.3 \
    datasets==2.21.0 \
    accelerate==1.2.1 \
    tokenizers==0.20.3

# bitsandbytes (8-bit Adam)
echo "  [3.3] bitsandbytes (8-bit Adam)..."
pip install -q bitsandbytes==0.44.1

# PEFT + TRL
echo "  [3.4] PEFT / TRL..."
pip install -q peft==0.13.2 trl==0.12.2

# 工具库
echo "  [3.5] Utilities..."
pip install -q pyyaml wandb tqdm numpy scipy einops sentencepiece==0.2.0 protobuf

# 可选: flash-attention (加速训练)
echo "  [3.6] Flash-Attention..."
pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "    flash-attn 跳过 (不影响运行)"

# 可选: deepspeed (13B用)
pip install -q deepspeed 2>/dev/null || echo "    deepspeed 跳过"

# ============================================
# Step 4: 验证
# ============================================
echo ""
echo "[Step 4/4] 验证环境..."

python -c "
import torch
print(f'  PyTorch    : {torch.__version__}')
print(f'  CUDA       : {torch.version.cuda}')
print(f'  GPU Count  : {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'  GPU Name   : {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import transformers
print(f'  Transformers: {transformers.__version__}')

import bitsandbytes
print(f'  bitsandbytes: {bitsandbytes.__version__}')

# 验证数据文件存在
import os
files = [
    '/root/autodl-tmp/model/deepseek-coder-6.7b-base',
    '/root/autodl-tmp/model/Mistral-7B-v0___1',
    '/root/autodl-tmp/model/llama-7b',
    '/root/autodl-tmp/dataset/MetaMathQA/MetaMathQA-395K.json',
    '/root/autodl-tmp/dataset/Magicoder-Evol-Instruct-110K/data-evol_instruct-decontaminated.jsonl',
]
import glob
tulu = glob.glob('/root/autodl-tmp/dataset/tulu-v2-sft-mixture/data/*.parquet')
print(f'  Models found   : {sum(1 for f in files if os.path.exists(f))}/{len(files)}')
print(f'  Tulu V2 files  : {len(tulu)} parquet(s)')
print('  环境验证通过!')
"

echo ""
echo "============================================"
echo "  环境搭建完成!"
echo "============================================"
echo ""
echo "  每次新开终端先激活:"
echo "    conda activate la_mo_gmt"
echo "    cd /root/autodl-tmp/MGMT"
echo ""
echo "  开始训练:"
echo "    python scripts/train_sft.py --config configs/code_generation.yaml"
echo ""
echo "  所有实验:"
echo "    bash run.sh"
