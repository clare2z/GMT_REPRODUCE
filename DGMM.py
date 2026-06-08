"""
DGMM — 每层自适应梯度保留 (Per-Layer Adaptive Gradient Retention)

核心理念:
  GMT: 全局固定保留 80% → 二元 mask，所有层一样
  DGMM: 每层独立保留比例 (70%-90%) → 关键层多留，冗余层多砍

创新 1: 梯度方向/稳定性/层间相关 → 决定每层留多少
创新 2: 关键层留更多(↓砍)、低价值层砍更多(↓留)，每层独立

为什么 AdamW 下有效: 梯度设为零不受 AdamW 抵消 (零永远是零)
"""

import os, re
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class GradientEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=64):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return F.normalize(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))), dim=-1)

class ContrastiveLearner(nn.Module):
    def __init__(self, encoder_dim=64, temperature=0.5):
        super().__init__()
    def forward(self, a, p, n): return torch.tensor(0.0, device=a.device)

class LayerAttentionFusion(nn.Module):
    def __init__(self, feature_dim=64, num_layers=12):
        super().__init__()
    def forward(self, x): return x


class DGMMFramework:
    def __init__(self, encoder_hidden_dim=128, encoder_output_dim=64,
                 contrastive_temperature=0.5, contrastive_weight=0.1,
                 consistency_weight=0.2, ema_alpha=0.99,
                 device="cuda", dtype=torch.bfloat16,
                 grad_history_window=5, warmup_steps=500,
                 mask_floor=0.2, meta_lr=1e-5):
        self.device = device
        self.dtype = dtype
        self.ema_alpha = ema_alpha
        self.warmup_steps = warmup_steps
        self.encoder_output_dim = encoder_output_dim
        self.grad_history: Dict[str, list] = {}
        self.step_count = 0
        self.layer_keep: Dict[str, float] = {}

    # ═══ 创新一: 三个分析维度 ═══════════════════════════

    def _direction(self, grad):
        """梯度方向: 正=向前学, 负=反向调, 零=收敛"""
        return (grad > 0).float().mean(), (grad < 0).float().mean(), (grad == 0).float().mean()

    def _stability(self, name, grad):
        """稳定性: 标准差(频率) + 波动幅度 + 动量(趋势)"""
        if name not in self.grad_history:
            self.grad_history[name] = []
        std, diff, mom = 0.0, 0.0, 0.0
        norm = float(grad.norm().item())
        h = self.grad_history[name]
        if h:
            std = float(np.std(h))
            if len(h) >= 2:
                diff = float(abs(h[-1] - h[-2]))
                if len(h) >= 3:
                    mom = float(diff / (abs(h[-2] - h[-3]) + 1e-8))
        h.append(norm)
        if len(h) > 5: h.pop(0)
        return (torch.tensor(std, device=self.device),
                torch.tensor(diff, device=self.device),
                torch.tensor(mom, device=self.device))

    def _correlation(self, features):
        """层间协同: z-score → 皮尔逊矩阵 → 全局均值"""
        n = features.size(0)
        if n < 2: return torch.tensor(0.0, device=self.device)
        z = (features - features.mean(0, keepdim=True)) / (features.std(0, keepdim=True) + 1e-8)
        return torch.mm(z, z.t()).mean() / max(1, features.size(1))

    # ═══ 层分组 ═════════════════════════════════════════

    def _group_layers(self, grads):
        groups = {}
        for name, grad in grads.items():
            m = re.search(r'layers\.(\d+)', name)
            key = f"L{m.group(1):02d}" if m else name.split('.')[-1]
            groups.setdefault(key, []).append(grad.to(self.device).flatten())
        return {k: torch.cat(v) for k, v in groups.items()}

    # ═══ 核心 ═══════════════════════════════════════════

    def apply_mask(self, accumulated_grads):
        if os.environ.get("DGMM_DISABLED") == "1":
            return accumulated_grads, {'avg_importance': 1.0, 'layer_corr': 0.0,
                                        'contrastive_loss': 0.0, 'consistency_loss': 0.0}

        layer_grads = self._group_layers(accumulated_grads)
        names = sorted(layer_grads.keys())
        n = len(names)
        if n < 2:
            # 只有一层, 按 GMT 处理
            threshold = float(torch.kthvalue(
                list(layer_grads.values())[0].abs(),
                max(1, int(list(layer_grads.values())[0].numel() * 0.2))
            ).values.item())
            return {name: grad * (grad.abs() >= threshold).float().to(grad.dtype)
                    for name, grad in accumulated_grads.items()}, \
                   {'avg_importance': 0.8, 'layer_corr': 0.0, 'contrastive_loss': 0.0, 'consistency_loss': 0.0}

        # ── 每层 6 维统计 ──────────────────────────
        stats_list = []
        for name in names:
            g = layer_grads[name]
            pos, neg, zero = self._direction(g)
            std, diff, mom = self._stability(name, g)
            stats_list.append(torch.stack([
                pos, neg, zero, std, diff,
                torch.tensor(min(mom.item(), 3.0), device=self.device)
            ]))
        stats = torch.stack(stats_list)

        # ── 跨层 z-score + 层间相关 ─────────────────
        stats_norm = (stats - stats.mean(0, keepdim=True)) / (stats.std(0, keepdim=True) + 1e-8)
        corr = self._correlation(stats_norm)

        # ── 创新二: 每层独立 keep_pct ───────────────
        # 活跃+稳定+高协同 → 多留; 挣扎+波动+孤立 → 多砍
        w = torch.tensor([4.0, -3.0, -1.0, -2.0, 1.5, 1.5], device=self.device)
        quality = (stats_norm * w).sum(dim=1) + corr * 2.0
        keep_pct = 0.80 + quality * 0.08  # 基准 80% ± 调整
        keep_pct = torch.clamp(keep_pct, 0.65, 0.92)  # [65%, 92%]

        # EMA 追踪
        for i, name in enumerate(names):
            kv = float(keep_pct[i].item())
            self.layer_keep[name] = self.ema_alpha * self.layer_keep.get(name, kv) + (1 - self.ema_alpha) * kv

        # ── 每层阈值 + 掩码 ────────────────────────
        thresholds = {}
        for name in names:
            g_abs = layer_grads[name].abs()
            kp = self.layer_keep[name]
            cut = max(1, int(g_abs.numel() * (1.0 - kp)))
            thresholds[name] = float(torch.kthvalue(g_abs, cut).values.item())

        self.step_count += 1
        masked_grads = {}
        for name, grad in accumulated_grads.items():
            m = re.search(r'layers\.(\d+)', name)
            key = f"L{m.group(1):02d}" if m else name.split('.')[-1]
            thr = thresholds.get(key)
            if thr is not None:
                mask = grad.abs() >= thr
                masked_grad = grad * mask.float().to(grad.dtype)
                if self.step_count <= self.warmup_steps:
                    masked_grads[name] = grad
                else:
                    ramp = min(1.0, (self.step_count - self.warmup_steps) / max(1, self.warmup_steps))
                    masked_grads[name] = grad * (1.0 - ramp) + masked_grad * ramp
            else:
                masked_grads[name] = grad

        info = {'avg_importance': float(keep_pct.mean().item()),
                'layer_corr': float(corr.item()),
                'contrastive_loss': 0.0, 'consistency_loss': 0.0}
        return masked_grads, info

    def get_layer_importance(self):
        return self.layer_keep.copy()
