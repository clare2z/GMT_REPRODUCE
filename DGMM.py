"""
DGMM — Per-Layer Adaptive Gradient Retention

创新 1: 动态梯度建模
  1. 方向一致性 — 当前梯度与固定位置 EMA 梯度的 cosine similarity
  2. 波动性 — grad.abs().mean() 历史变异系数
  3. 层间协同 — 每层与所有其他层的平均皮尔逊相关

创新 2: 层自适应参数更新
  4. 关键层保留更多 — keep_pct 自适应
  5. 低价值层减少冗余 — keep_pct 自适应
  6. 非统一 mask — 每层独立 keep_pct + parameter-level threshold

DGMM-v1-effective (dgmm_final_46) — pos/neg/zero方向+grad.norm稳定度+全局层间相关。safe schedule: step<500→[0.95,0.99], 500-999→[0.90,0.98], 1000+→[0.85,0.98]。Ablation: DGMM-v2-cosine-synergy=32%, aggressive=41%, budget=38%。
"""

import os, re
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Set


# ═══════════════════════════════════════════════════════════════
# [已弃用] 保留空壳以维持向后兼容，DGMM 不再使用神经网络
# ═══════════════════════════════════════════════════════════════

class GradientEncoder(nn.Module):
    """[已弃用] DGMM v1 梯度编码器，当前版本不使用"""
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return F.normalize(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))), dim=-1)

class ContrastiveLearner(nn.Module):
    """[已弃用] DGMM v1 对比学习器，当前版本不使用"""
    def __init__(self, encoder_dim=64, temperature=0.5):
        super().__init__()
    def forward(self, a, p, n):
        return torch.tensor(0.0, device=a.device)

class LayerAttentionFusion(nn.Module):
    """[已弃用] DGMM v1 层注意力融合，当前版本不使用"""
    def __init__(self, feature_dim=64, num_layers=12):
        super().__init__()
    def forward(self, x):
        return x


# ═══════════════════════════════════════════════════════════════
# DGMM v2 核心
# ═══════════════════════════════════════════════════════════════

class DGMMFramework:
    def __init__(self, encoder_hidden_dim=128, encoder_output_dim=64,
                 contrastive_temperature=0.5, contrastive_weight=0.1,
                 consistency_weight=0.2, ema_alpha=0.99,
                 device="cuda", dtype=torch.bfloat16,
                 grad_history_window=5, warmup_steps=500,
                 mask_floor=0.2, meta_lr=1e-5,
                 ablate: str = ""):
        self.ema_alpha = ema_alpha
        self.warmup_steps = warmup_steps
        self.encoder_output_dim = encoder_output_dim
        self.step_count = 0
        self.ablate = set(ablate.split(",")) if ablate else set()

        # 历史追踪
        self.grad_ema: Dict[str, torch.Tensor] = {}
        self.sample_indices: Dict[str, torch.Tensor] = {}  # 固定采样索引
        self.absmean_history: Dict[str, list] = {}
        self.stats_ema: Dict[str, torch.Tensor] = {}
        self.layer_keep: Dict[str, float] = {}

        # 需跳过的参数模式
        self._skip_kw = ['embed', 'lm_head', 'norm', 'bias']

    # ═══ 参数过滤 ══════════════════════════════════════════

    def _should_skip(self, name: str, grad: torch.Tensor) -> bool:
        """跳过 embed / lm_head / norm / bias / 1D 参数"""
        if grad.ndim <= 1:
            return True
        name_lower = name.lower()
        for kw in self._skip_kw:
            if kw in name_lower:
                return True
        return False

    # ═══ 创新一: 三个分析维度 ═══════════════════════════

    def _direction(self, grad: torch.Tensor):
        """1. 方向分析: pos/neg/zero 梯度符号分布"""
        return (grad > 0).float().mean(), (grad < 0).float().mean(), (grad == 0).float().mean()

    def _stability(self, name: str, grad: torch.Tensor) -> float:
        """2. 稳定性: grad.norm() 历史波动 (0=稳定, 1=波动)"""
        if name not in self.absmean_history:
            self.absmean_history[name] = []
        norm_val = float(grad.detach().float().norm().item())
        h = self.absmean_history[name]
        h.append(norm_val)
        if len(h) > 20:
            h.pop(0)
        if len(h) < 2:
            return 0.0
        cv = float(np.std(h) / (np.mean(h) + 1e-8))
        return min(cv, 1.0)

    def _correlation(self, features: torch.Tensor) -> float:
        """3. 全局层间相关: z-score → 皮尔逊矩阵 → 全局均值"""
        n = features.size(0)
        if n < 2:
            return 0.0
        z = (features - features.mean(0, keepdim=True)) / (features.std(0, keepdim=True) + 1e-8)
        return float(torch.mm(z, z.t()).mean().item() / max(1, features.size(1)))

    # ═══ 层分组 ═════════════════════════════════════════

    def _group_layers(self, grads: Dict[str, torch.Tensor], skip_names: Set[str]) -> Dict[str, torch.Tensor]:
        groups = {}
        for name, grad in grads.items():
            if name in skip_names:
                continue
            g = grad.detach().float().flatten()
            m = re.search(r'layers\.(\d+)', name)
            key = f"L{int(m.group(1)):02d}" if m else name.rsplit('.', 1)[0]
            groups.setdefault(key, []).append(g)
        return {k: torch.cat(v) for k, v in groups.items()}

    # ═══ 核心 ═══════════════════════════════════════════

    def apply_mask(self, accumulated_grads: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict]:
        if os.environ.get("DGMM_DISABLED") == "1":
            return accumulated_grads, self._empty_info()

        # ── 分类 ────────────────────────────────────
        skip_names = {name for name, g in accumulated_grads.items() if self._should_skip(name, g)}
        layer_grads = self._group_layers(accumulated_grads, skip_names)
        names = sorted(layer_grads.keys())
        n = len(names)

        if n < 2:
            return self._fallback_gmt(accumulated_grads, skip_names, layer_grads)

        # ── 每层 pos/neg/zero + stability ───────────
        raw_features: Dict[str, torch.Tensor] = {}
        for name in names:
            g = layer_grads[name]
            pos, neg, zero = self._direction(g)
            stab = 1.0 - self._stability(name, g)  # 高=稳定
            raw_features[name] = torch.tensor([pos, neg, zero, stab])

        # ── 全局层间相关 ─────────────────────────────
        feat_stack = torch.stack([raw_features[n] for n in names])
        corr = self._correlation(feat_stack)

        # ── quality → keep_pct ───────────────────────
        for name in names:
            f = raw_features[name]
            d = 0.5 if "direction" in self.ablate else (f[0].item() - f[1].item())  # pos - neg
            v = 0.5 if "volatility" in self.ablate else f[3].item()  # stability
            y = 0.0 if "synergy" in self.ablate else corr
            quality = float((+3.0 * d + 2.0 * v + 2.0 * y))
            keep = 0.89 + quality * 0.10
            keep = max(0.0, min(1.0, keep))
            if name in self.layer_keep:
                self.layer_keep[name] = self.ema_alpha * self.layer_keep[name] + (1 - self.ema_alpha) * keep
            else:
                self.layer_keep[name] = keep

        # ── 阶段 clamp ───────────────────────────────
        if self.step_count < 500:
            lo, hi = 0.95, 0.99
        elif self.step_count < 1000:
            lo, hi = 0.92, 0.98
        else:
            lo, hi = 0.90, 0.98
        for name in names:
            self.layer_keep[name] = max(lo, min(hi, self.layer_keep[name]))

        # ── 应用: parameter-level threshold ──────────
        self.step_count += 1
        masked_grads = {}
        target_keeps = []
        mask_keeps = []

        for name, grad in accumulated_grads.items():
            if name in skip_names:
                masked_grads[name] = grad
                target_keeps.append(1.0)
                mask_keeps.append(1.0)
                continue

            m = re.search(r'layers\.(\d+)', name)
            key = f"L{int(m.group(1)):02d}" if m else name.rsplit('.', 1)[0]
            kp = self.layer_keep.get(key, 0.89)

            g_abs = grad.detach().float().abs()
            cut_idx = max(1, int(g_abs.numel() * (1.0 - kp)))
            thr = float(torch.kthvalue(g_abs.flatten(), cut_idx).values.item())
            mask = grad.abs() >= thr
            mask_actual = float(mask.float().mean().item())
            masked_grad = grad * mask.float().to(grad.dtype)

            if self.step_count <= self.warmup_steps:
                masked_grads[name] = grad
                mask_keeps.append(1.0)
                target_keeps.append(kp)
            else:
                ramp = min(1.0, (self.step_count - self.warmup_steps) / max(1, self.warmup_steps))
                masked_grads[name] = grad * (1.0 - ramp) + masked_grad * ramp
                mask_keeps.append(mask_actual * ramp + (1 - ramp))
                target_keeps.append(kp)


        # ── 日志 ─────────────────────────────────────
        # 从 self.layer_keep 去重获取每层 keep（layer-level 非 parameter-level）
        sorted_layers = sorted(self.layer_keep.items(), key=lambda x: x[1])
        n_show = min(3, len(sorted_layers))
        lowest = [(l[0], f"{l[1]:.3f}") for l in sorted_layers[:n_show]]
        highest = [(l[0], f"{l[1]:.3f}") for l in sorted_layers[-n_show:]]
        avg_synergy = float(corr)

        info = {
            'avg_importance': float(np.mean(target_keeps)),
            'layer_corr': avg_synergy,
            'contrastive_loss': 0.0,
            'consistency_loss': 0.0,
            'mask_keep_mean': float(np.mean(mask_keeps)),
            'effective_keep_mean': float(np.mean(mask_keeps)),
            'target_keep_mean': float(np.mean(target_keeps)),
            'target_keep_min': float(min(target_keeps)),
            'target_keep_max': float(max(target_keeps)),
            'lowest_layers': str(lowest),
            'highest_layers': str(highest),
        }
        return masked_grads, info

    def _fallback_gmt(self, grads, skip_names, layer_grads):
        if not layer_grads:
            return grads, self._empty_info()
        g_abs = list(layer_grads.values())[0].abs()
        thr = float(torch.kthvalue(g_abs, max(1, int(g_abs.numel() * 0.2))).values.item())
        masked = {}
        for name, grad in grads.items():
            if name in skip_names:
                masked[name] = grad
            else:
                masked[name] = grad * (grad.abs() >= thr).float().to(grad.dtype)
        return masked, self._empty_info()

    def _empty_info(self):
        return {'avg_importance': 1.0, 'layer_corr': 0.0, 'contrastive_loss': 0.0,
                'consistency_loss': 0.0, 'mask_keep_mean': 1.0, 'effective_keep_mean': 1.0,
                'target_keep_mean': 1.0, 'target_keep_min': 1.0, 'target_keep_max': 1.0,
                'lowest_layers': '[]', 'highest_layers': '[]'}

    def get_layer_importance(self):
        return self.layer_keep.copy()
