"""
DGMM (Dynamic Gradient Manifold Masking) — 独立模块，挂接到 MGMT 框架

权限约定: 不修改 MGMT 原有代码, 只通过 apply_gradient_mask() 的 dgmm 分支接入
"""

import os, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Set


class DGMMFramework:
    """DGMM 核心 — 梯度方向/稳定性/层间相关分析 + per-layer 自适应 keep ratio"""

    def __init__(
        self,
        encoder_output_dim: int = 64,
        ema_alpha: float = 0.99,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        warmup_steps: int = 500,
        soft_alpha: float = 0.0,
        late_start: int = 0,
        keep_update_interval: int = 1,
        ablate: str = "",
    ):
        self.ema_alpha = ema_alpha
        self.warmup_steps = warmup_steps
        self.encoder_output_dim = encoder_output_dim
        self.soft_alpha = soft_alpha
        self.late_start = late_start
        self.step_count = 0
        self.ablate = set(ablate.split(",")) if ablate else set()
        self.keep_update_interval = keep_update_interval
        self.device = device

        # 历史追踪
        self.grad_ema: Dict[str, torch.Tensor] = {}
        self.sample_indices: Dict[str, torch.Tensor] = {}
        self.absmean_history: Dict[str, list] = {}
        self.stats_ema: Dict[str, torch.Tensor] = {}
        self.layer_keep: Dict[str, float] = {}
        self._last_keeps: Dict[str, float] = {}
        self._skip_kw = ['embed', 'lm_head', 'norm', 'bias']

    def _should_skip(self, name: str, grad: torch.Tensor) -> bool:
        if grad.ndim <= 1:
            return True
        for kw in self._skip_kw:
            if kw in name.lower():
                return True
        return False

    def _direction(self, grads: list):
        total_elems = 0
        pos_count = 0
        neg_count = 0
        for g in grads:
            gf = g.detach().float()
            total_elems += gf.numel()
            pos_count += (gf > 0).sum().item()
            neg_count += (gf < 0).sum().item()
        if total_elems == 0:
            return 0.0, 0.0, 0.0
        return pos_count / total_elems, neg_count / total_elems, (total_elems - pos_count - neg_count) / total_elems

    def _stability(self, name: str, grads: list) -> float:
        if name not in self.absmean_history:
            self.absmean_history[name] = []
        norm_sq = 0.0
        for g in grads:
            norm_sq += float(g.detach().float().norm().item()) ** 2
        norm_val = norm_sq ** 0.5
        h = self.absmean_history[name]
        h.append(norm_val)
        if len(h) > 20:
            h.pop(0)
        if len(h) < 2:
            return 0.0
        cv = float(np.std(h) / (np.mean(h) + 1e-8))
        return min(cv, 1.0)

    def _correlation(self, features: torch.Tensor) -> float:
        n = features.size(0)
        if n < 2:
            return 0.0
        z = (features - features.mean(0, keepdim=True)) / (features.std(0, keepdim=True) + 1e-8)
        return float(torch.mm(z, z.t()).mean().item() / max(1, features.size(1)))

    def _group_layers(self, grads: Dict[str, torch.Tensor], skip_names: Set[str]) -> Dict[str, list]:
        groups = {}
        for name, grad in grads.items():
            if name in skip_names:
                continue
            m = re.search(r'layers\.(\d+)', name)
            key = f"L{int(m.group(1)):02d}" if m else name.rsplit('.', 1)[0]
            groups.setdefault(key, []).append(grad)
        return groups

    def apply_mask(self, accumulated_grads: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict]:
        import time

        if os.environ.get("DGMM_DISABLED") == "1":
            return accumulated_grads, {"avg_importance": 1.0}

        skip_names = {name for name, g in accumulated_grads.items() if self._should_skip(name, g)}
        layer_grads = self._group_layers(accumulated_grads, skip_names)
        names = sorted(layer_grads.keys())
        n = len(names)

        if n < 2:
            return self._fallback_gmt(accumulated_grads, skip_names, layer_grads)

        corr = 0.0
        do_update = (self.step_count % self.keep_update_interval == 0)

        if do_update:
            raw_features: Dict[str, torch.Tensor] = {}
            for name in names:
                grads = layer_grads[name]
                pos, neg, zero = self._direction(grads)
                stab = 1.0 - self._stability(name, grads)
                raw_features[name] = torch.tensor([pos, neg, zero, stab])

            feat_stack = torch.stack([raw_features[n] for n in names])
            corr = self._correlation(feat_stack)

            for name in names:
                f = raw_features[name]
                d = 0.5 if "direction" in self.ablate else (f[0].item() - f[1].item())
                v = 0.5 if "volatility" in self.ablate else f[3].item()
                y = 0.0 if "synergy" in self.ablate else corr
                quality = float((+3.0 * d + 2.0 * v + 2.0 * y))
                keep = 0.89 + quality * 0.10
                keep = max(0.0, min(1.0, keep))
                if name in self.layer_keep:
                    self.layer_keep[name] = self.ema_alpha * self.layer_keep[name] + (1 - self.ema_alpha) * keep
                else:
                    self.layer_keep[name] = keep

            if self.step_count <= self.warmup_steps:
                lo, hi = 0.99, 1.00
            elif self.step_count < 500:
                lo, hi = 0.90, 0.99
            elif self.step_count < 1000:
                lo, hi = 0.85, 0.99
            else:
                lo, hi = 0.80, 0.99
            for name in names:
                self.layer_keep[name] = max(lo, min(hi, self.layer_keep[name]))
            self._last_keeps = dict(self.layer_keep)

        # 应用 mask
        self.step_count += 1
        masked_grads = {}

        layer_thresholds: Dict[str, float] = {}
        for name in names:
            grads = layer_grads[name]
            kp = self.layer_keep.get(name, 0.89)
            if kp >= 1.0:
                continue
            g0_abs = grads[0].detach().float().abs().flatten()
            n_sample = min(100000, g0_abs.numel())
            if n_sample < g0_abs.numel():
                idx = torch.randint(0, g0_abs.numel(), (n_sample,), device=g0_abs.device)
                sampled = g0_abs[idx]
            else:
                sampled = g0_abs
            cut_idx = max(1, int(sampled.numel() * (1.0 - kp)))
            layer_thresholds[name] = float(torch.kthvalue(sampled, cut_idx).values.item())

        for name, grad in accumulated_grads.items():
            if name in skip_names:
                masked_grads[name] = grad
                continue
            m = re.search(r'layers\.(\d+)', name)
            key = f"L{int(m.group(1)):02d}" if m else name.rsplit('.', 1)[0]
            kp = self.layer_keep.get(key, 0.89)
            thr = layer_thresholds.get(key, 0.0)
            mask = grad.abs() >= thr

            if self.soft_alpha > 0:
                mf = mask.float().to(grad.dtype)
                scale = mf + (1.0 - mf) * self.soft_alpha
                masked_grads[name] = grad * scale
            else:
                masked_grads[name] = grad * mask.float().to(grad.dtype)

        kvals = list(self.layer_keep.values())
        info = {
            "avg_importance": float(np.mean(kvals)) if kvals else 0.89,
            "keep_min": float(min(kvals)) if kvals else 0.89,
            "keep_max": float(max(kvals)) if kvals else 0.89,
        }
        return masked_grads, info

    def _fallback_gmt(self, grads, skip_names, layer_grads):
        if not layer_grads:
            return grads, {"avg_importance": 0.8}
        first_list = list(layer_grads.values())[0]
        g_abs = first_list[0].detach().float().abs().flatten()
        thr = float(torch.kthvalue(g_abs, max(1, int(g_abs.numel() * 0.2))).values.item())
        masked = {}
        for name, grad in grads.items():
            if name in skip_names:
                masked[name] = grad
            else:
                masked[name] = grad * (grad.abs() >= thr).float().to(grad.dtype)
        return masked, {"avg_importance": 0.8}
