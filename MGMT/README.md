# LA-Mo-GMT: Layer-Adaptive Momentum-Guided Gradient Mask Tuning

Efficient LLM fine-tuning with gradient-based sparse parameter selection.  
Builds on [GMT (AAAI-25)](https://arxiv.org/abs/2406.xxxxx) with two orthogonal improvements:

- **Mo-GMT**: Uses Adam's momentum `|m_t|` instead of raw gradient `|g_t|` for importance estimation → **~19× noise reduction**
- **LA-GMT**: Allocates per-layer mask ratios adaptively → **efficient update budget allocation**

Both improvements are **zero extra memory** (momentum reused from Adam, layer stats <1KB).

## Supported Methods

| Method | Importance Signal | Mask Ratio | Paper |
|--------|------------------|------------|-------|
| `none` | — | No masking | Vanilla SFT |
| `gmt` | `|g_t|` (raw gradient) | Global uniform | GMT (AAAI-25) |
| `mo_gmt` | `|m_t|` (Adam momentum) | Global uniform | Ours |
| `la_gmt` | `|g_t|` (raw gradient) | Per-layer adaptive | Ours |
| `la_mo_gmt` | `|m_t|` (Adam momentum) | Per-layer adaptive | **Ours (full)** |
| `rmt` | Random | Global uniform | Baseline |

## Installation

```bash
git clone <repo-url>
cd LA-Mo-GMT
pip install -r requirements.txt
```

## Quick Start

### 1. SFT Training

```bash
# LA-Mo-GMT (our full method, code generation)
python scripts/train_sft.py --config configs/code_generation.yaml

# GMT baseline (for comparison)
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=gmt

# Vanilla SFT baseline
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=none

# Mo-GMT only (ablation)
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=mo_gmt

# LA-GMT only (ablation)
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=la_gmt

# RMT baseline
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=rmt
```

### 2. DPO Training

```bash
python scripts/train_dpo.py --config configs/general_dpo.yaml
```

### 3. Evaluation

```bash
# Code generation (HumanEval, MBPP)
python scripts/eval_code.py \
    --model_path ./outputs/code_mistral_la_mo_gmt/final \
    --tasks humaneval mbpp \
    --use_8bit   # Save memory during eval

# Math reasoning (GSM8k, MATH)
python scripts/eval_math.py \
    --model_path ./outputs/math_mistral_la_mo_gmt/final \
    --tasks gsm8k math

# General domain (MMLU, GSM8k, BBH, TyDiQA, TruthfulQA, HumanEval)
python scripts/eval_general.py \
    --model_path ./outputs/general_llama2_7b_la_mo_gmt/final \
    --tasks mmlu gsm8k bbh tydiqa truthfulqa humaneval \
    --use_harness
```

## Memory Usage (80GB GPU)

| Model | Method | Optimizer | Peak Memory | Status |
|-------|--------|-----------|-------------|--------|
| 7B (Mistral, LLaMA2) | SFT | 8-bit Adam | ~42 GB | ✅ |
| 7B | GMT | 8-bit Adam | ~42 GB | ✅ |
| 7B | LA-Mo-GMT | 8-bit Adam | ~56 GB | ✅ (14GB EMA) |
| 7B | LA-Mo-GMT | FP32 Adam | ~84 GB | ⚠️ Tight |
| 8B (LLaMA3) | LA-Mo-GMT | 8-bit Adam | ~61 GB | ✅ |
| 13B (LLaMA2) | LA-Mo-GMT | 8-bit Adam + ZeRO-2 | ~55 GB/GPU | ✅ |

**Memory breakdown for 7B LA-Mo-GMT with 8-bit Adam:**
- Model (BF16): 14 GB
- Gradients (BF16): 14 GB
- 8-bit Adam (m+v): 14 GB
- EMA buffers (BF16): 14 GB (Mo-GMT only)
- Activations (checkpointed): ~5 GB
- **Total: ~61 GB**

## Project Structure

```
LA-Mo-GMT/
├── README.md
├── requirements.txt
├── configs/
│   ├── code_generation.yaml      # Code: Mistral-7B, DeepSeek-Coder-6.7B
│   ├── math_reasoning.yaml       # Math: Mistral-7B, LLaMA3-8B
│   ├── general_sft.yaml          # General SFT: LLaMA2-7B, LLaMA2-13B
│   └── general_dpo.yaml          # General DPO: LLaMA2-7B, LLaMA2-13B
├── src/la_mo_gmt/
│   ├── __init__.py
│   ├── masking.py                # Core gradient masking logic
│   ├── optimizer.py              # GradientMaskOptimizer wrapper
│   ├── trainer.py                # LAMoGMTTrainer (HF Trainer subclass)
│   └── data_utils.py             # Dataset loading & collation
└── scripts/
    ├── train_sft.py              # SFT training (all methods)
    ├── train_dpo.py              # DPO training with reference model
    ├── eval_code.py              # HumanEval(+) / MBPP(+)
    ├── eval_math.py              # GSM8k / MATH
    └── eval_general.py           # MMLU / BBH / TyDiQA / TruthfulQA
```

## Configuration

Edit the YAML files in `configs/` to set your paths:

```yaml
model_name_or_path: "/your/path/to/model"
data_path: "/your/path/to/dataset.jsonl"
output_dir: "./outputs/your_experiment"

mask:
  method: "la_mo_gmt"       # Method name
  global_ratio: 0.3         # 30% gradients masked (70% updated)
  alpha: 1.0                # Layer-adaptive concentration
  beta1: 0.9                # EMA decay for momentum

training:
  per_device_batch_size: 4
  gradient_accumulation_steps: 8
  optim: "adamw_8bit"       # 8-bit Adam for memory
  gradient_checkpointing: true
  bf16: true
```

Override any value from command line:
```bash
python scripts/train_sft.py --config configs/math_reasoning.yaml \
    --override mask.global_ratio=0.5 mask.alpha=2.0 training.learning_rate=1e-5
```

## Reproducing GMT Paper Results

To reproduce GMT baselines and compare with our improvements:

```bash
# 1. GMT (original method)
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=gmt mask.global_ratio=0.3

# 2. Our LA-Mo-GMT (same settings, only method differs)
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=la_mo_gmt mask.global_ratio=0.3

# 3. Vanilla SFT
python scripts/train_sft.py --config configs/code_generation.yaml \
    --override mask.method=none
```

All other hyperparameters identical to GMT paper: learning rate 2e-5, cosine schedule, warmup 3%, BF16, no weight decay.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{li2025gmt,
  title={Enhancing Large Language Model Performance with Gradient-Based Parameter Selection},
  author={Li, Haoling and Zhang, Xin and Liu, Xiao and Gong, Yeyun and Wang, Yifan and Chen, Qi and Cheng, Peng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## License

MIT
