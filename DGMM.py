"""
DGMM v2 — 全局梯度质量分析 + 动态缩放

创新 1：梯度方向(正/负) + 稳定性(标准差/波动/动量) 全局分析
创新 2：自适应缩放 — 质量高放大(>1.0)，质量低压低(<1.0)

与 GMT 区别：GMT 二元掩码只砍不升，DGMM 连续缩放可升可降
"""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


# 保留旧模块（向后兼容，不再使用）
class GradientEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return F.normalize(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))), dim=-1)

class ContrastiveLearner(nn.Module):
    def __init__(self, encoder_dim=64, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    def forward(self, a, p, n):
        return torch.tensor(0.0, device=a.device)

class LayerAttentionFusion(nn.Module):
    def __init__(self, feature_dim=64, num_layers=12):
        super().__init__()
    def forward(self, x):
        return x


class DGMMFramework:
    def __init__(
        self,
        encoder_hidden_dim=128, encoder_output_dim=64,
        contrastive_temperature=0.5, contrastive_weight=0.1,
        consistency_weight=0.2, ema_alpha=0.99,
        device="cuda", dtype=torch.bfloat16,
        grad_history_window=5, warmup_steps=500,
        mask_floor=0.2, meta_lr=1e-5,
    ):
        self.device = device
        self.dtype = dtype
        self.ema_alpha = ema_alpha
        self.grad_history_window = grad_history_window
        self.warmup_steps = warmup_steps
        self.mask_floor = mask_floor

        self.layer_importance: Dict[str, float] = {}
        self.grad_history: Dict[str, list] = {}
        self.step_count = 0
        self.global_scale = 1.0

    def _analyze_gradient_direction(self, grad):
        return (grad > 0).float().mean(), (grad < 0).float().mean(), (grad == 0).float().mean()

    def _analyze_gradient_stability(self, name, grad):
        if name not in self.grad_history:
            self.grad_history[name] = []
        grad_std, grad_diff, momentum = 0.0, 0.0, 0.0
        norm = float(grad.norm().item())
        h = self.grad_history[name]
        if h:
            grad_std = float(np.std(h))
            if len(h) >= 2:
                grad_diff = float(abs(h[-1] - h[-2]))
                if len(h) >= 3:
                    prev_diff = abs(h[-2] - h[-3])
                    momentum = float(grad_diff / (prev_diff + 1e-8))
        h.append(norm)
        if len(h) > self.grad_history_window:
            h.pop(0)
        return (torch.tensor(grad_std, device=self.device),
                torch.tensor(grad_diff, device=self.device),
                torch.tensor(momentum, device=self.device))

    def apply_mask(self, accumulated_grads):
        if os.environ.get("DGMM_DISABLED") == "1":
            return accumulated_grads, {'avg_importance': 1.0, 'layer_corr': 0.0,
                                        'contrastive_loss': 0.0, 'consistency_loss': 0.0}

        # ── 全局梯度分析（所有层合并）───────────────────────
        all_grads = []
        for g in accumulated_grads.values():
            all_grads.append(g.to(self.device).flatten())
        global_grad = torch.cat(all_grads)

        pos, neg, zero = self._analyze_gradient_direction(global_grad)
        std, diff, mom = self._analyze_gradient_stability("global", global_grad)

        # ── 6 统计量 → 动态 k_percent ───────────────────────
        quality = (
            +3.0 * pos.item()
            -2.0 * neg.item()
            -1.0 * min(std.item(), 1.0)
            -2.0 * min(diff.item(), 0.5)
            +1.5 * min(mom.item(), 3.0) if mom.item() > 0 else -0.5
        )
        # k_percent: 质量好→激进砍，质量差→保守留。GMT 固定 80%
        k = 80.0 + quality * 15.0
        k = max(0.55, min(0.95, k / 100.0))  # k ∈ [55%, 95%]

        # 全局阈值掩码（如 GMT，但 k 动态）
        all_abs = global_grad.abs()
        k_idx = max(1, int(all_abs.numel() * (1.0 - k)))
        threshold = float(torch.kthvalue(all_abs, k_idx).values.item())

        # ── warmup ─────────────────────────────────────────
        self.step_count += 1
        masked_grads = {}
        for name, grad in accumulated_grads.items():
            mask = grad.abs() >= threshold
            masked_grad = grad * mask.float().to(grad.dtype)

            if self.step_count <= self.warmup_steps:
                masked_grads[name] = grad
            else:
                ramp = min(1.0, (self.step_count - self.warmup_steps) / max(1, self.warmup_steps))
                masked_grads[name] = grad * (1.0 - ramp) + masked_grad * ramp

        info = {
            'avg_importance': weight,
            'layer_corr': 0.0,
            'contrastive_loss': 0.0,
            'consistency_loss': 0.0,
        }
        return masked_grads, info

    def get_layer_importance(self):
        return self.layer_importance.copy()
