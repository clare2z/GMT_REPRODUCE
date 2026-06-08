"""
DGMM — 每层自适应梯度保留 (Per-Layer Adaptive Gradient Retention)

创新 1: 动态梯度建模
  1. 方向一致性(cosine similarity with EMA gradient)
  2. 稳定性(grad.abs().mean() 历史波动)
  3. 层间协同(per-layer synergy score)

创新 2: 层自适应参数更新
  4. 关键层保留更多 ← keep_pct 自适应
  5. 低价值层减少冗余 ← keep_pct 自适应
  6. 非统一 mask ← 每层独立 keep_pct + parameter-level threshold

特殊处理: embed/lm_head/norm/bias/1D 参数原样通过

AdamW 兼容: 只做设零(梯度为零不受 AdamW 抵消)
"""

import os, re
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Set


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
        self.step_count = 0

        # 历史追踪
        self.grad_ema: Dict[str, torch.Tensor] = {}      # per-layer: EMA of gradient values
        self.absmean_history: Dict[str, list] = {}       # per-layer: grad.abs().mean() history
        self.stats_ema: Dict[str, torch.Tensor] = {}     # per-layer: 6-dim stats EMA (for synergy)
        self.layer_keep: Dict[str, float] = {}           # per-layer: EMA keep_pct

        # 不需处理的参数名模式
        self._skip_patterns = ['embed', 'lm_head', 'norm', 'bias']

    def _should_skip(self, name: str) -> bool:
        """跳过 embed/lm_head/norm/bias/1D 参数"""
        # 1D parameters
        if name.endswith('weight') and 'weight' in name:
            pass  # need to check dim later
        for p in self._skip_patterns:
            if p in name:
                return True
        return False

    def _is_1d_or_bias(self, name: str, grad: torch.Tensor) -> bool:
        return grad.ndim <= 1 or 'bias' in name

    # ═══ 创新一: 三个分析维度 ═══════════════════════════

    def _direction_consistency(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        """1. 方向一致性: 当前梯度与历史 EMA 梯度的 cosine similarity"""
        g_flat = grad.flatten().to(self.dtype)
        # 截断到统一长度
        max_len = 50000
        if g_flat.size(0) > max_len:
            idx = torch.randperm(g_flat.size(0), device=self.device)[:max_len]
            g_flat = g_flat[idx]

        if name not in self.grad_ema:
            self.grad_ema[name] = g_flat.detach().clone()
            return torch.tensor(1.0, device=self.device)  # 第一步 = 完全一致

        ema = self.grad_ema[name]
        # cosine similarity
        cos = torch.dot(g_flat, ema) / (g_flat.norm() * ema.norm() + 1e-8)
        # update EMA
        self.grad_ema[name] = self.ema_alpha * ema + (1 - self.ema_alpha) * g_flat.detach()
        return cos.clamp(-1.0, 1.0)

    def _stability(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        """2. 稳定性: grad.abs().mean() 的历史波动"""
        if name not in self.absmean_history:
            self.absmean_history[name] = []

        abs_mean = float(grad.abs().mean().item())
        h = self.absmean_history[name]
        h.append(abs_mean)
        if len(h) > 20:
            h.pop(0)

        if len(h) < 2:
            return torch.tensor(0.0, device=self.device)  # 没历史 = 完全稳定

        # 变异系数: std / mean (归一化波动)
        mean_val = np.mean(h)
        std_val = np.std(h)
        cv = std_val / (mean_val + 1e-8)
        # 映射到 [0, 1]: 0=完全稳定, 1=极度波动
        return torch.tensor(min(float(cv), 1.0), device=self.device)

    def _layer_synergy(self, name: str, all_stats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """3. 层间协同: 该层与所有其他层的平均相关性"""
        if len(all_stats) < 2:
            return torch.tensor(0.0, device=self.device)

        this_feat = all_stats[name]
        synergies = []
        for other_name, other_feat in all_stats.items():
            if other_name == name:
                continue
            # Pearson correlation between this layer's stats and other layer's stats
            corr = torch.dot(this_feat, other_feat) / (this_feat.norm() * other_feat.norm() + 1e-8)
            synergies.append(corr.item())
        return torch.tensor(np.mean(synergies), device=self.device)

    # ═══ 层分组 ═════════════════════════════════════════

    def _group_layers(self, grads: Dict[str, torch.Tensor], skip_names: Set[str]) -> Dict[str, torch.Tensor]:
        """把梯度的 transfer 层分组合并（跳过特殊参数）"""
        groups = {}
        for name, grad in grads.items():
            if name in skip_names:
                continue
            m = re.search(r'layers\.(\d+)', name)
            key = f"L{int(m.group(1)):02d}" if m else name.rsplit('.', 1)[0]
            groups.setdefault(key, []).append(grad.to(self.device).flatten())
        return {k: torch.cat(v) for k, v in groups.items()}

    # ═══ 核心 ═══════════════════════════════════════════

    def apply_mask(self, accumulated_grads: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict]:
        if os.environ.get("DGMM_DISABLED") == "1":
            return accumulated_grads, {'avg_importance': 1.0, 'layer_corr': 0.0,
                                        'contrastive_loss': 0.0, 'consistency_loss': 0.0}

        # ── 分类参数 ────────────────────────────────
        skip_names = set()
        for name, grad in accumulated_grads.items():
            if self._should_skip(name) or self._is_1d_or_bias(name, grad):
                skip_names.add(name)

        layer_grads = self._group_layers(accumulated_grads, skip_names)
        names = sorted(layer_grads.keys())
        n = len(names)

        # 只有一层时退化为 GMT
        if n < 2:
            return self._fallback_gmt(accumulated_grads, list(layer_grads.values())[0])

        # ── 每层 3 维信号 ──────────────────────────
        all_stats = {}  # name → 3-dim feature (cosine, stability, synergy_placeholder)
        raw_features = {}
        for name in names:
            g = layer_grads[name]
            cos = self._direction_consistency(name, g)        # [0, 1]
            stab = self._stability(name, g)                    # [0, 1], lower=better
            raw_features[name] = torch.stack([cos, stab])

        # synergy: 用 (cos, stability) 作为特征向量算层间相关性
        for name in names:
            syn = self._layer_synergy(name, raw_features)      # [-1, 1]
            all_stats[name] = torch.stack([
                raw_features[name][0],  # cosine: 高=方向一致
                raw_features[name][1],  # stability: 低=稳定
                syn,                    # synergy: 高=协同
            ])
            # EMA 平滑 stats（用于跨步稳定）
            if name in self.stats_ema:
                self.stats_ema[name] = self.ema_alpha * self.stats_ema[name] + (1 - self.ema_alpha) * all_stats[name]
            else:
                self.stats_ema[name] = all_stats[name]

        # ── 每层 quality → keep_pct ─────────────────
        # cos↑ keep↑  stab↑(不稳定) keep↓  syn↑ keep↑
        keep_raw = {}
        for i, name in enumerate(names):
            s = self.stats_ema[name]
            quality = float((+2.0 * s[0] - 1.5 * s[1] + 1.5 * s[2]).item())
            keep = 0.89 + quality * 0.06
            keep_raw[name] = keep
            self.layer_keep[name] = self.ema_alpha * self.layer_keep.get(name, keep) + (1 - self.ema_alpha) * keep

        # ── 稳定性保护：分阶段 clamp ──────────────────
        if self.step_count < 500:
            lo, hi = 0.95, 0.99
        elif self.step_count < 1000:
            lo, hi = 0.90, 0.98
        else:
            lo, hi = 0.85, 0.98
        for name in names:
            self.layer_keep[name] = max(lo, min(hi, self.layer_keep[name]))

        # ── 应用: parameter-level threshold ──────────
        self.step_count += 1
        masked_grads = {}
        all_target = []
        all_actual = []
        all_layer_keeps = []

        for name, grad in accumulated_grads.items():
            if name in skip_names:
                masked_grads[name] = grad
                all_target.append(1.0)
                all_actual.append(1.0)
                continue

            m = re.search(r'layers\.(\d+)', name)
            key = f"L{int(m.group(1)):02d}" if m else name.rsplit('.', 1)[0]
            kp = self.layer_keep.get(key, 0.89)

            # parameter-level threshold
            g_abs = grad.abs()
            cut_idx = max(1, int(g_abs.numel() * (1.0 - kp)))
            thr = float(torch.kthvalue(g_abs.flatten(), cut_idx).values.item())
            mask = g_abs >= thr
            actual_keep = float(mask.float().mean().item())
            masked_grad = grad * mask.float().to(grad.dtype)

            if self.step_count <= self.warmup_steps:
                masked_grads[name] = grad
                all_target.append(kp)
                all_actual.append(1.0)  # warmup期间全保留
            else:
                ramp = min(1.0, (self.step_count - self.warmup_steps) / max(1, self.warmup_steps))
                masked_grads[name] = grad * (1.0 - ramp) + masked_grad * ramp
                all_target.append(kp)
                all_actual.append(kp * ramp + (1 - ramp))  # ramp混合后的真实保留率

            all_layer_keeps.append((key, kp, actual_keep))

        # ── 详细日志 ─────────────────────────────────
        sorted_layers = sorted(all_layer_keeps, key=lambda x: x[1])
        n_show = min(3, len(sorted_layers))
        lowest = sorted_layers[:n_show]
        highest = sorted_layers[-n_show:]

        info = {
            'avg_importance': float(np.mean(all_target)),
            'layer_corr': float(np.mean(all_actual)),
            'contrastive_loss': float(np.mean(all_actual)),
            'consistency_loss': 0.0,
            # 新增字段（供日志查看）
            'actual_keep_mean': float(np.mean(all_actual)),
            'target_keep_min': float(min(all_target)),
            'target_keep_max': float(max(all_target)),
            'lowest_layers': str([(l[0], f"{l[1]:.3f}") for l in lowest]),
            'highest_layers': str([(l[0], f"{l[1]:.3f}") for l in highest]),
        }
        return masked_grads, info

    def _fallback_gmt(self, grads, layer_grad):
        """单层退化为 GMT"""
        g_abs = layer_grad.abs()
        thr = float(torch.kthvalue(g_abs, max(1, int(g_abs.numel() * 0.2))).values.item())
        masked = {}
        for name, grad in grads.items():
            if self._should_skip(name) or self._is_1d_or_bias(name, grad):
                masked[name] = grad
            else:
                masked[name] = grad * (grad.abs() >= thr).float().to(grad.dtype)
        return masked, {'avg_importance': 0.8, 'layer_corr': 0.0,
                        'contrastive_loss': 0.0, 'consistency_loss': 0.0}

    def get_layer_importance(self):
        return self.layer_keep.copy()
