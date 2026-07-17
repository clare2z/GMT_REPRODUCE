#!/bin/bash
# ──────────────────────────────────────────
# LA-Mo-GMT: 一键运行所有对比实验
# 显存: 80GB, 使用 8-bit Adam
# ──────────────────────────────────────────
set -e

# ==============================================
# 实验 1: 代码生成 (DeepSeek-Coder-6.7B)
# Dataset: Magicoder-Evol-Instruct-110K
# ==============================================
echo "=== Code Generation: DeepSeek-Coder-6.7B ==="

# 基线: Vanilla SFT
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=none output_dir=./outputs/code_sft

# 基线: GMT (原论文方法)
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=gmt output_dir=./outputs/code_gmt

# 基线: RMT (Random Mask)
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=rmt output_dir=./outputs/code_rmt

# Ours: LA-Mo-GMT (完整方法)
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=la_mo_gmt output_dir=./outputs/code_la_mo_gmt

# 消融: Mo-GMT only
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=mo_gmt output_dir=./outputs/code_mo_gmt

# 消融: LA-GMT only
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=la_gmt output_dir=./outputs/code_la_gmt

# 评估代码生成
for m in sft gmt rmt la_mo_gmt mo_gmt la_gmt; do
    python scripts/eval_code.py --model_path "./outputs/code_${m}/final" \
        --tasks humaneval mbpp --use_8bit --model_type deepseek
done

# ==============================================
# 实验 2: 数学推理 (Mistral-7B)
# Dataset: MetaMathQA
# ==============================================
echo "=== Math Reasoning: Mistral-7B ==="

python scripts/train_sft.py --config configs/math_reasoning.yaml \
    --override mask.method=none output_dir=./outputs/math_sft

python scripts/train_sft.py --config configs/math_reasoning.yaml \
    --override mask.method=gmt output_dir=./outputs/math_gmt

python scripts/train_sft.py --config configs/math_reasoning.yaml \
    --override mask.method=rmt output_dir=./outputs/math_rmt

python scripts/train_sft.py --config configs/math_reasoning.yaml \
    --override mask.method=la_mo_gmt output_dir=./outputs/math_la_mo_gmt

python scripts/train_sft.py --config configs/math_reasoning.yaml \
    --override mask.method=mo_gmt output_dir=./outputs/math_mo_gmt

python scripts/train_sft.py --config configs/math_reasoning.yaml \
    --override mask.method=la_gmt output_dir=./outputs/math_la_gmt

# 评估数学
for m in sft gmt rmt la_mo_gmt mo_gmt la_gmt; do
    python scripts/eval_math.py --model_path "./outputs/math_${m}/final" \
        --tasks gsm8k math
done

# ==============================================
# 实验 3: 通用领域 SFT (LLaMA2-7B)
# Dataset: Tulu V2
# ==============================================
echo "=== General Domain: LLaMA2-7B ==="

python scripts/train_sft.py --config configs/general_sft.yaml \
    --override mask.method=none output_dir=./outputs/general_sft

python scripts/train_sft.py --config configs/general_sft.yaml \
    --override mask.method=gmt output_dir=./outputs/general_gmt

python scripts/train_sft.py --config configs/general_sft.yaml \
    --override mask.method=la_mo_gmt output_dir=./outputs/general_la_mo_gmt

# 评估通用领域 (需要 lm-evaluation-harness)
# python scripts/eval_general.py --model_path "./outputs/general_la_mo_gmt/final" \
#     --tasks mmlu gsm8k bbh tydiqa truthfulqa humaneval --use_harness

echo ""
echo "===== All experiments complete! ====="
echo "Results: ./outputs/"
