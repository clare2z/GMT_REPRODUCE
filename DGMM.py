"""
DGMM — 全创新点版本

创新一：动态梯度建模
  1. 梯度方向(正/负/零) — _analyze_gradient_direction
  2. 稳定性(标准差/波动/动量) — _analyze_gradient_stability
  3. 层间协同(皮尔逊相关) — _analyze_layer_correlation

创新二：层自适应参数更新
  4. 关键层保留更多 — per-layer boost, 活跃层放大梯度
  5. 低价值层减少冗余 — per-layer k_percent, 低价值层砍更多
  6. 非统一 mask — 每层独立 k_percent + 独立 boost
"""

import os, re
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


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
        self.grad_history_window = grad_history_window
        self.warmup_steps = warmup_steps
        self.mask_floor = mask_floor
        self.encoder_output_dim = encoder_output_dim
        self.layer_importance: Dict[str, float] = {}
        self.grad_history: Dict[str, list] = {}
        self.step_count = 0

    # ═══ 创新一：三个分析函数 ══════════════════════════════

    def _analyze_gradient_direction(self, grad):
        """1. 梯度方向：正=向前学, 负=反向调, 零=收敛"""
        return (grad > 0).float().mean(), (grad < 0).float().mean(), (grad == 0).float().mean()

    def _analyze_gradient_stability(self, name, grad):
        """2. 稳定性：标准差(频率) + 变化幅度 + 动量(趋势)"""
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
                    prev_diff = abs(h[-2] - h[-3])
                    mom = float(diff / (prev_diff + 1e-8))
        h.append(norm)
        if len(h) > self.grad_history_window:
            h.pop(0)
        return (torch.tensor(std, device=self.device),
                torch.tensor(diff, device=self.device),
                torch.tensor(mom, device=self.device))

    def _analyze_layer_correlation(self, features):
        """3. 层间协同：z-score → 皮尔逊矩阵 → 全局均值"""
        n = features.size(0)
        if n < 2:
            return torch.tensor(0.0, device=self.device)
        z = (features - features.mean(0, keepdim=True)) / (features.std(0, keepdim=True) + 1e-8)
        corr = torch.mm(z, z.t()) / max(1, features.size(1))
        return corr.mean()

    # ═══ 层分组 ════════════════════════════════════════════

    def _group_by_layer(self, grads):
        groups = {}
        for name, grad in grads.items():
            m = re.search(r'layers\.(\d+)', name)
            key = f"L{m.group(1)}" if m else (name.split('.')[1] if '.' in name else name.split('.')[0])
            groups.setdefault(key, []).append(grad.to(self.device).flatten())
        return {k: torch.cat(v) for k, v in groups.items()}

    # ═══ 核心 ══════════════════════════════════════════════

    def apply_mask(self, accumulated_grads):
        if os.environ.get("DGMM_DISABLED") == "1":
            return accumulated_grads, {'avg_importance': 1.0, 'layer_corr': 0.0,
                                        'contrastive_loss': 0.0, 'consistency_loss': 0.0}

        layer_grads = self._group_by_layer(accumulated_grads)
        names = sorted(layer_grads.keys())

        # ── 每层收集 6 个统计量 ─────────────────────────
        stats_list = []  # (n_layers, 6)
        for name in names:
            g = layer_grads[name]
            pos, neg, zero = self._analyze_gradient_direction(g)
            std, diff, mom = self._analyze_gradient_stability(name, g)
            stats_list.append(torch.stack([pos, neg, zero, std, diff,
                torch.tensor(min(mom.item(), 3.0), device=self.device)]))
        stats = torch.stack(stats_list)  # (N, 6)

        # ── 跨层 z-score 归一化 + 层间相关性 ────────────
        stats_norm = (stats - stats.mean(0, keepdim=True)) / (stats.std(0, keepdim=True) + 1e-8)
        padded = F.pad(stats_norm, (0, max(0, self.encoder_output_dim - 6)))
        corr = self._analyze_layer_correlation(padded)

        # ── 每层质量 → per-layer boost + k_percent ──────
        # 创新二：每层独立 boost(升/降) + k_percent(保留比例)
        w = torch.tensor([3.0, -2.0, -1.0, -2.0, 1.5, 1.5], device=self.device)
        quality = (stats_norm * w).sum(dim=1) + corr * 2.0  # (N,)

        # per-layer boost: 重要层放大，不重要层压低 → [0.5, 1.5]
        boost = 1.0 + torch.tanh(quality * 0.5) * 0.5

        # per-layer k_percent: 重要层保留多，不重要砍多 → [60%, 90%]
        k_pct = 0.80 + quality * 0.10
        k_pct = torch.clamp(k_pct, 0.60, 0.90)

        # EMA 追踪层重要性
        for i, name in enumerate(names):
            imp = float(boost[i].item())
            if name in self.layer_importance:
                self.layer_importance[name] = self.ema_alpha * self.layer_importance[name] + (1 - self.ema_alpha) * imp
            else:
                self.layer_importance[name] = imp

        # ── 预先算每层的阈值和boost ──────────────────
        layer_info = {}  # key → (threshold, boost)
        for i, name in enumerate(names):
            grad = layer_grads[name]
            g_abs = grad.abs()
            cut_idx = max(1, int(g_abs.numel() * (1.0 - k_pct[i].item())))
            thr = float(torch.kthvalue(g_abs, cut_idx).values.item())
            layer_info[name] = (thr, boost[i].item())

        # ── 按参数逐一应用 ──────────────────────────
        self.step_count += 1
        masked_grads = {}
        for name, grad in accumulated_grads.items():
            m = re.search(r'layers\.(\d+)', name)
            key = f"L{m.group(1)}" if m else (name.split('.')[1] if '.' in name else name.split('.')[0])
            info = layer_info.get(key)
            if info:
                thr, bst = info
                mask = grad.abs() >= thr
                masked_grad = grad * mask.float().to(grad.dtype) * bst
                if self.step_count <= self.warmup_steps:
                    masked_grads[name] = grad
                else:
                    ramp = min(1.0, (self.step_count - self.warmup_steps) / max(1, self.warmup_steps))
                    masked_grads[name] = grad * (1.0 - ramp) + masked_grad * ramp
            else:
                masked_grads[name] = grad

        info = {'avg_importance': float(boost.mean().item()),
                'layer_corr': corr.item(),
                'contrastive_loss': 0.0, 'consistency_loss': 0.0}
        return masked_grads, info

    def get_layer_importance(self):
        return self.layer_importance.copy()
