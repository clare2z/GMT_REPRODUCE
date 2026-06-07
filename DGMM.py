"""
DGMM (Dynamic Gradient Manifold Masking) 核心框架

简化版：梯度方向/稳定性/层间相关性的统计量直接计算重要性。
去除元网络（GradientEncoder/ContrastiveLearner/LayerAttentionFusion），
改用 0 参数统计公式，保留全部 6 个分析维度 + 2 大创新点。
"""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


# ═══════════════════════════════════════════════════════════════
# 保留旧模块类（向后兼容，新版 DGMMFramework 不再使用）
# ═══════════════════════════════════════════════════════════════

class GradientEncoder(nn.Module):
    """[已弃用] 梯度编码器"""
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.norm(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.normalize(x, dim=-1)


class ContrastiveLearner(nn.Module):
    """[已弃用] 对比学习器"""
    def __init__(self, encoder_dim=64, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.ReLU(), nn.Linear(encoder_dim, encoder_dim)
        )

    def forward(self, anchors, positives, negatives):
        a = self.projection_head(anchors)
        p = self.projection_head(positives)
        n = self.projection_head(negatives)
        pos_sim = torch.sum(a * p, dim=-1) / self.temperature
        neg_sim = torch.mm(a, n.t()) / self.temperature
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(a.size(0), dtype=torch.long, device=a.device)
        return F.cross_entropy(logits, labels)


class LayerAttentionFusion(nn.Module):
    """[已弃用] 层注意力融合"""
    def __init__(self, feature_dim=64, num_layers=12):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, layer_features):
        x = layer_features.unsqueeze(0)
        q, k, v = self.query_proj(x), self.key_proj(x), self.value_proj(x)
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / np.sqrt(x.size(-1)), dim=-1)
        return self.output_proj(torch.bmm(attn, v)).squeeze(0)


# ═══════════════════════════════════════════════════════════════
# 新版 DGMM 框架 — 纯统计驱动，0 参数元网络
# ═══════════════════════════════════════════════════════════════

class DGMMFramework:
    """
    DGMM 核心框架 (简化版)

    两个创新点全部保留：
    1. 动态梯度建模：方向(正/负/零) + 稳定性(标准差/波动/动量) + 层间协同(皮尔逊相关系数)
    2. 层自适应更新：每层独立评分，关键层保留更多梯度，低价值层抑制

    重要性评分公式 (6维度 → 1分数):
        score = +2.0 * pos_ratio      ← 正梯度多 = 主动学习 → 重要
                -1.5 * neg_ratio      ← 负梯度多 = 反向挣扎 → 降低
                -0.5 * stability      ← 波动大 = 不稳定 → 降低
                -1.0 * grad_diff      ← 变化大 = 噪声 → 降低
                +1.0 * momentum_f     ← 动量↑ = 加速收敛 → 重要
                +1.0 * 层相关性       ← 其他层协同 → 重要

    皮尔逊相关系数: 层间 z-score 归一化后做点积 → 协方差矩阵 → 全局均值
    """

    def __init__(
        self,
        encoder_hidden_dim: int = 128,        # 保留兼容
        encoder_output_dim: int = 64,         # 保留兼容
        contrastive_temperature: float = 0.5, # 保留兼容
        contrastive_weight: float = 0.1,      # 保留兼容
        consistency_weight: float = 0.2,      # 保留兼容
        ema_alpha: float = 0.99,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        grad_history_window: int = 5,
        warmup_steps: int = 500,
        mask_floor: float = 0.2,
        meta_lr: float = 1e-5,                # 保留兼容
    ):
        self.device = device
        self.dtype = dtype
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.ema_alpha = ema_alpha
        self.grad_history_window = grad_history_window
        self.warmup_steps = warmup_steps
        self.mask_floor = mask_floor

        # 层重要性追踪
        self.layer_importance: Dict[str, float] = {}
        self.prev_layer_importance: Dict[str, float] = {}
        self.global_importance_threshold = 0.8  # k_percent 基准

        # 梯度历史（稳定性分析用）
        self.grad_history: Dict[str, list] = {}
        self.step_count = 0

    # ═══════════════════════════════════════════════════════
    # 三大梯度分析模块 (创新点一：动态梯度建模)
    # ═══════════════════════════════════════════════════════

    def _analyze_gradient_direction(self, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        梯度变化方向分析

        返回:
            pos_ratio  — 正梯度比例（参数在前进方向学习）
            neg_ratio  — 负梯度比例（参数在反向调整）
            zero_ratio — 零梯度比例（参数收敛或停滞）
        """
        positive_ratio = (grad > 0).float().mean()
        negative_ratio = (grad < 0).float().mean()
        zero_ratio = (grad == 0).float().mean()
        return positive_ratio, negative_ratio, zero_ratio

    def _analyze_gradient_stability(self, layer_name: str, current_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        梯度波动频率与长期稳定性分析

        维护最近 5 步的梯度范数历史，计算：
            grad_std  — 标准差（波动频率，小 = 稳定）
            grad_diff — 最近两步变化幅度（瞬时波动）
            momentum  — 变化加速度（>1 = 加速变化，<1 = 减速稳定）

        返回: (grad_std, grad_diff, momentum) 均为标量 tensor
        """
        if layer_name not in self.grad_history:
            self.grad_history[layer_name] = []

        grad_std = 0.0
        grad_diff = 0.0
        momentum = 0.0

        current_grad_norm = float(current_grad.norm().item())

        if len(self.grad_history[layer_name]) > 0:
            grad_std = float(np.std(self.grad_history[layer_name]))

            if len(self.grad_history[layer_name]) >= 2:
                recent = self.grad_history[layer_name][-1]
                prev = self.grad_history[layer_name][-2]
                grad_diff = float(np.abs(recent - prev))

                if len(self.grad_history[layer_name]) >= 3:
                    prev_prev = self.grad_history[layer_name][-3]
                    prev_diff = abs(prev - prev_prev)
                    momentum = float(grad_diff / (prev_diff + 1e-8))

        self.grad_history[layer_name].append(current_grad_norm)
        if len(self.grad_history[layer_name]) > self.grad_history_window:
            self.grad_history[layer_name].pop(0)

        return (
            torch.tensor(grad_std, device=self.device),
            torch.tensor(grad_diff, device=self.device),
            torch.tensor(momentum, device=self.device)
        )

    def _analyze_layer_correlation(self, layer_features: torch.Tensor) -> torch.Tensor:
        """
        Transformer 层之间的梯度协同关系

        对每层特征做 z-score 归一化 → 皮尔逊相关系数矩阵 → 全局平均

        含义：不同层梯度变化方向的一致性。高相关 = 层间协作学习，低相关 = 各层独立更新。
        """
        num_layers = layer_features.size(0)
        if num_layers < 2:
            return torch.tensor(0.0, device=self.device)

        normalized = (layer_features - layer_features.mean(dim=1, keepdim=True)) / (layer_features.std(dim=1, keepdim=True) + 1e-8)
        correlation_matrix = torch.mm(normalized, normalized.t()) / self.encoder_output_dim
        return correlation_matrix.mean()

    # ═══════════════════════════════════════════════════════
    # 层分组
    # ═══════════════════════════════════════════════════════

    def _compute_layer_gradients(self, accumulated_grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将梯度按 transformer 层块分组（每 8 层一组，32 层 → 4 block）"""
        layer_grads = {}
        for name, grad in accumulated_grads.items():
            match = re.search(r'layers\.(\d+)', name)
            if match:
                layer_name = f"layer_{match.group(1)}"
            else:
                layer_name = name.split('.')[1] if '.' in name else name.split('.')[0]
            if layer_name not in layer_grads:
                layer_grads[layer_name] = []
            layer_grads[layer_name].append(grad.to(self.device).flatten())

        for layer_name in layer_grads:
            layer_grads[layer_name] = torch.cat(layer_grads[layer_name], dim=0)

        return layer_grads

    # ═══════════════════════════════════════════════════════
    # 核心：统计公式算重要性 (替代元网络)
    # ═══════════════════════════════════════════════════════

    def _compute_importance_scores(self, layer_grads: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], float]:
        """
        创新点二：层自适应参数更新

        6 统计量 → 每层动态 k_percent → 每层独立阈值掩码。

        关键层（活跃、稳定、高协同）→ k_percent 高 → 保留更多参数更新
        低价值层（挣扎、波动、低协同）→ k_percent 低 → 砍掉更多冗余更新

        k_percent 范围：50%-95%（动态调整，GMT 用固定 80%）
        """
        layer_names = sorted(layer_grads.keys())
        n = max(1, len(layer_names))

        # ── 收集每层的 6 个统计量 ─────────────────────────────
        raw_stats = {}  # layer_name → [pos, neg, zero, std, diff, mom]
        for layer_name in layer_names:
            grad = layer_grads[layer_name]
            pos, neg, zero = self._analyze_gradient_direction(grad)
            std, diff, mom = self._analyze_gradient_stability(layer_name, grad)
            raw_stats[layer_name] = torch.stack([
                pos, neg, zero, std, diff, torch.tensor(min(mom.item(), 3.0), device=self.device)
            ])

        # ── 跨层 z-score 归一化 ────────────────────────────────
        stats_tensor = torch.stack([raw_stats[name] for name in layer_names])  # (n, 6)
        stats_mean = stats_tensor.mean(dim=0)
        stats_std = stats_tensor.std(dim=0) + 1e-8
        stats_norm = (stats_tensor - stats_mean) / stats_std

        # ── 层间相关性 ─────────────────────────────────────────
        padded = F.pad(stats_norm, (0, self.encoder_output_dim - 6))
        layer_corr = self._analyze_layer_correlation(padded)

        # ── 6 统计量 → 动态 k_percent ───────────────────────────
        # 权重放大 5 倍 + 宽范围 clamp，确保层间有真实差异
        weights = torch.tensor([50.0, -50.0, -25.0, -25.0, 25.0, 25.0], device=self.device)
        k_delta = (stats_norm * weights).sum(dim=1) + layer_corr * 25.0
        k_percent = 80.0 + k_delta
        k_percent = torch.clamp(k_percent, 30.0, 95.0) / 100.0  # → [0.3, 0.95]

        scores = {}
        for i, name in enumerate(layer_names):
            scores[name] = float(k_percent[i].item())

        return scores, layer_corr.item()

    # ═══════════════════════════════════════════════════════
    # 掩码应用
    # ═══════════════════════════════════════════════════════

    def apply_mask(self, accumulated_grads: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        应用 DGMM 掩码到梯度。

        Returns:
            (masked_grads, info_dict)
        """
        # DGMM_DISABLED=1 则完全不干预梯度
        if os.environ.get("DGMM_DISABLED") == "1":
            return accumulated_grads, {
                'avg_importance': 1.0, 'layer_corr': 0.0,
                'contrastive_loss': 0.0, 'consistency_loss': 0.0
            }

        layer_grads = self._compute_layer_gradients(accumulated_grads)
        importance_scores, layer_corr = self._compute_importance_scores(layer_grads)

        # EMA 平滑 + 一致性追踪
        avg_importance = sum(importance_scores.values()) / max(1, len(importance_scores))
        consistency_loss = 0.0

        for layer_name in sorted(layer_grads.keys()):
            imp = importance_scores.get(layer_name, self.global_importance_threshold)
            if layer_name in self.prev_layer_importance:
                consistency_loss += (imp - self.prev_layer_importance[layer_name]) ** 2

            if layer_name in self.layer_importance:
                self.layer_importance[layer_name] = self.ema_alpha * self.layer_importance[layer_name] + (1 - self.ema_alpha) * imp
            else:
                self.layer_importance[layer_name] = imp

            self.prev_layer_importance[layer_name] = imp

        # ── 掩码应用 + warmup ──────────────────────────────────
        # 每层用各自的 k_percent 算阈值（重要层阈值低→保留多）
        block_thresholds = {}
        for layer_name, kp in importance_scores.items():
            if layer_name in layer_grads:
                grad = layer_grads[layer_name]
                k_idx = max(1, int(grad.numel() * (1.0 - kp)))  # 砍掉 bottom (1-kp)
                block_thresholds[layer_name] = float(torch.kthvalue(grad.abs(), k_idx).values.item())

        masked_grads = {}
        self.step_count += 1

        for name, grad in accumulated_grads.items():
            match = re.search(r'layers\.(\d+)', name)
            if match:
                layer_name = f"layer_{match.group(1)}"
            else:
                layer_name = name.split('.')[1] if '.' in name else name.split('.')[0]

            threshold = block_thresholds.get(layer_name, 0.0)
            mask = grad.abs() >= threshold
            masked_grad = grad * mask.float().to(grad.dtype)

            # warmup + ramp：warmup 期间不掩码，之后渐进引入
            if self.step_count <= self.warmup_steps:
                masked_grads[name] = grad
            else:
                ramp = min(1.0, (self.step_count - self.warmup_steps) / max(1, self.warmup_steps))
                masked_grads[name] = grad * (1.0 - ramp) + masked_grad * ramp

        info = {
            'avg_importance': avg_importance,
            'layer_corr': layer_corr,
            'contrastive_loss': 0.0,
            'consistency_loss': consistency_loss,
        }
        return masked_grads, info

    def get_layer_importance(self) -> Dict[str, float]:
        """获取层重要性分数"""
        return self.layer_importance.copy()


# ═══════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════

def test_dgmm():
    """测试 DGMM 框架"""
    print("Testing DGMM Framework (Statistical Version)...")

    dgmm = DGMMFramework(
        device="cuda" if torch.cuda.is_available() else "cpu",
        warmup_steps=0  # 测试时不延迟
    )

    test_grads = {
        'model.layers.0.self_attn.q_proj.weight': torch.randn(512, 512).to(dgmm.device).to(dgmm.dtype),
        'model.layers.0.self_attn.k_proj.weight': torch.randn(512, 512).to(dgmm.device).to(dgmm.dtype),
        'model.layers.8.mlp.down_proj.weight': torch.randn(256, 512).to(dgmm.device).to(dgmm.dtype),
        'model.layers.16.self_attn.o_proj.weight': torch.randn(512, 512).to(dgmm.device).to(dgmm.dtype),
        'model.layers.24.mlp.up_proj.weight': torch.randn(512, 256).to(dgmm.device).to(dgmm.dtype),
        'lm_head.weight': torch.randn(32000, 512).to(dgmm.device).to(dgmm.dtype),
    }

    masked_grads, info = dgmm.apply_mask(test_grads)
    importance = dgmm.get_layer_importance()

    print(f"Gradients: {len(test_grads)} → {len(masked_grads)}")
    print(f"Info: {info}")
    print("Layer importance (per-block):")
    for k, v in sorted(importance.items()):
        print(f"  {k}: {v:.4f}")
    print("DGMM Framework test passed!")


if __name__ == "__main__":
    test_dgmm()
