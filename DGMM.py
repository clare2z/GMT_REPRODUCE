"""
DGMM (Dynamic Gradient Manifold Masking) 核心框架

基于对比学习和自适应掩码的梯度训练方法。
包含梯度方向分析、稳定性追踪、层间相关性分析、warmup 机制。
"""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class GradientEncoder(nn.Module):
    """梯度编码器：将梯度信息编码到特征空间"""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.norm(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.normalize(x, dim=-1)


class ContrastiveLearner(nn.Module):
    """对比学习器：学习梯度流形结构"""
    def __init__(self, encoder_dim: int = 64, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim)
        )

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        anchors = self.projection_head(anchors)
        positives = self.projection_head(positives)
        negatives = self.projection_head(negatives)

        pos_sim = torch.sum(anchors * positives, dim=-1) / self.temperature
        neg_sim = torch.mm(anchors, negatives.t()) / self.temperature

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(anchors.size(0), dtype=torch.long, device=anchors.device)
        return F.cross_entropy(logits, labels)


class LayerAttentionFusion(nn.Module):
    """层注意力融合：整合多层梯度特征"""
    def __init__(self, feature_dim: int = 64, num_layers: int = 12):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, layer_features: torch.Tensor) -> torch.Tensor:
        layer_features = layer_features.unsqueeze(0)
        queries = self.query_proj(layer_features)
        keys = self.key_proj(layer_features)
        values = self.value_proj(layer_features)

        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(layer_features.size(-1))
        attn_weights = F.softmax(attn_scores, dim=-1)

        fused = torch.bmm(attn_weights, values)
        fused = self.output_proj(fused)

        return fused.squeeze(0)


class DGMMFramework:
    """DGMM 核心框架 — 含梯度方向分析、稳定性追踪、warmup 机制"""

    def __init__(
        self,
        encoder_hidden_dim: int = 128,
        encoder_output_dim: int = 64,
        contrastive_temperature: float = 0.5,
        contrastive_weight: float = 0.1,
        consistency_weight: float = 0.2,
        ema_alpha: float = 0.9,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        grad_history_window: int = 5,
        warmup_steps: int = 500,
        mask_floor: float = 0.3,
    ):
        self.device = device
        self.dtype = dtype
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight
        self.ema_alpha = ema_alpha
        self.grad_history_window = grad_history_window
        self.warmup_steps = warmup_steps
        self.mask_floor = mask_floor

        self.gradient_encoder = GradientEncoder(
            input_dim=encoder_hidden_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim
        ).to(device).to(dtype)

        self.contrastive_learner = ContrastiveLearner(
            encoder_dim=encoder_output_dim,
            temperature=contrastive_temperature
        ).to(device).to(dtype)

        self.layer_attention = LayerAttentionFusion(
            feature_dim=encoder_output_dim,
            num_layers=12
        ).to(device).to(dtype)

        # 特征融合层：将 64 维编码特征 + 6 维统计特征 映射回 64 维
        self.feature_fusion = nn.Linear(encoder_output_dim + 6, encoder_output_dim).to(device).to(dtype)

        self.meta_optimizer = torch.optim.AdamW(
            list(self.gradient_encoder.parameters()) +
            list(self.contrastive_learner.parameters()) +
            list(self.layer_attention.parameters()) +
            list(self.feature_fusion.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )

        self.layer_importance: Dict[str, float] = {}
        self.prev_layer_importance: Dict[str, float] = {}
        self.global_importance_threshold = 0.5

        self.grad_history: Dict[str, list] = {}
        self.step_count = 0

    # ── 梯度分析 ──────────────────────────────────────────────

    def _analyze_gradient_direction(self, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """分析梯度方向：正/负/零比例"""
        positive_ratio = (grad > 0).float().mean()
        negative_ratio = (grad < 0).float().mean()
        zero_ratio = (grad == 0).float().mean()
        return positive_ratio, negative_ratio, zero_ratio

    def _analyze_gradient_stability(self, layer_name: str, current_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """分析梯度稳定性：标准差/波动/动量"""
        if layer_name not in self.grad_history:
            self.grad_history[layer_name] = []

        grad_std = 0.0
        grad_diff = 0.0
        momentum = 0.0

        current_grad_norm = float(current_grad.norm().item())

        if len(self.grad_history[layer_name]) > 0:
            grad_std = float(np.std(self.grad_history[layer_name]))

            if len(self.grad_history[layer_name]) >= 2:
                recent_grad_norm = self.grad_history[layer_name][-1]
                prev_grad_norm = self.grad_history[layer_name][-2]
                grad_diff = float(np.abs(recent_grad_norm - prev_grad_norm))

                if len(self.grad_history[layer_name]) >= 3:
                    prev_prev_grad_norm = self.grad_history[layer_name][-3]
                    prev_diff = np.abs(prev_grad_norm - prev_prev_grad_norm)
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
        """计算层间皮尔逊相关系数"""
        num_layers = layer_features.size(0)
        if num_layers < 2:
            return torch.tensor(0.0, device=self.device)

        normalized_features = (layer_features - layer_features.mean(dim=1, keepdim=True)) / (layer_features.std(dim=1, keepdim=True) + 1e-8)
        correlation_matrix = torch.mm(normalized_features, normalized_features.t()) / self.encoder_output_dim
        avg_correlation = correlation_matrix.mean()
        return avg_correlation

    # ── 核心流程 ──────────────────────────────────────────────

    def _compute_layer_gradients(self, accumulated_grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将梯度按 transformer 层号分组（如 layer_0, layer_1, ..., embed, norm, head）"""
        layer_grads = {}
        for name, grad in accumulated_grads.items():
            # 提取 transformer 层号: model.layers.5.xxx → layer_5
            match = re.search(r'layers\.(\d+)', name)
            if match:
                layer_name = f"layer_{match.group(1)}"
            else:
                # embed_tokens、norm、lm_head 等顶层模块
                layer_name = name.split('.')[1] if '.' in name else name.split('.')[0]
            if layer_name not in layer_grads:
                layer_grads[layer_name] = []
            layer_grads[layer_name].append(grad.to(self.device).flatten())

        for layer_name in layer_grads:
            layer_grads[layer_name] = torch.cat(layer_grads[layer_name], dim=0)

        return layer_grads

    def _extract_layer_features(self, layer_grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        """提取层特征：编码梯度 + 统计特征融合"""
        layer_features = []

        for layer_name in sorted(layer_grads.keys()):
            grad = layer_grads[layer_name]

            # 方向特征(3) + 稳定性特征(3) → 6 维统计向量
            pos_ratio, neg_ratio, zero_ratio = self._analyze_gradient_direction(grad)
            stability, grad_diff, momentum = self._analyze_gradient_stability(layer_name, grad)

            stats = torch.stack([
                pos_ratio.detach(), neg_ratio.detach(), zero_ratio.detach(),
                stability.detach(), grad_diff.detach(), momentum.detach()
            ])

            # 截断/填充到固定维度
            if grad.size(0) < self.encoder_hidden_dim:
                grad = F.pad(grad, (0, self.encoder_hidden_dim - grad.size(0)))
            elif grad.size(0) > self.encoder_hidden_dim:
                grad = grad[:self.encoder_hidden_dim]

            # 编码 + 融合
            base_features = self.gradient_encoder(grad.unsqueeze(0).to(self.dtype))
            fused = self.feature_fusion(torch.cat([base_features.squeeze(0), stats.to(self.dtype)], dim=0))

            layer_features.append(fused.unsqueeze(0))

        return torch.cat(layer_features, dim=0)

    def _build_contrastive_samples(self, layer_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建对比学习样本"""
        anchors = layer_features
        positives = layer_features.roll(1, dims=0)
        negatives = layer_features[torch.randperm(layer_features.size(0))]
        return anchors, positives, negatives

    def apply_mask(self, accumulated_grads: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        应用 DGMM 掩码到梯度。

        Returns:
            (masked_grads, info_dict)
        """
        # 调试开关：DGMM_DISABLED=1 则完全不干预梯度
        if os.environ.get("DGMM_DISABLED") == "1":
            return accumulated_grads, {
                'avg_importance': 1.0, 'layer_corr': 0.0,
                'contrastive_loss': 0.0, 'consistency_loss': 0.0
            }

        layer_grads = self._compute_layer_gradients(accumulated_grads)
        layer_features = self._extract_layer_features(layer_grads)

        layer_correlation = self._analyze_layer_correlation(layer_features)

        anchors, positives, negatives = self._build_contrastive_samples(layer_features)
        contrastive_loss = self.contrastive_learner(anchors, positives, negatives)

        fused_features = self.layer_attention(layer_features)
        importance_scores = torch.sigmoid(torch.mean(fused_features, dim=-1))

        avg_importance = importance_scores.mean().item()

        consistency_loss = 0.0
        for i, layer_name in enumerate(sorted(layer_grads.keys())):
            importance = importance_scores[i].item()
            if layer_name in self.prev_layer_importance:
                consistency_loss += (importance - self.prev_layer_importance[layer_name]) ** 2

            if layer_name in self.layer_importance:
                self.layer_importance[layer_name] = self.ema_alpha * self.layer_importance[layer_name] + (1 - self.ema_alpha) * importance
            else:
                self.layer_importance[layer_name] = importance

            self.prev_layer_importance[layer_name] = importance

        total_meta_loss = (
            self.contrastive_weight * contrastive_loss +
            self.consistency_weight * consistency_loss -
            0.05 * layer_correlation
        )

        self.meta_optimizer.zero_grad()
        total_meta_loss.backward()
        self.meta_optimizer.step()

        # ── 掩码应用 + warmup ──────────────────────────────────
        masked_grads = {}
        self.step_count += 1

        for name, grad in accumulated_grads.items():
            layer_name = name.split('.')[0]
            importance = self.layer_importance.get(layer_name, self.global_importance_threshold)

            weight = max(self.mask_floor, min(1.0, importance))

            # warmup: 前 warmup_steps 步不干扰，之后渐进引入
            if self.step_count <= self.warmup_steps:
                weight = 1.0
            else:
                # ramp 长度 = warmup_steps，比例自适应
                ramp = min(1.0, (self.step_count - self.warmup_steps) / max(1, self.warmup_steps))
                weight = 1.0 - ramp * (1.0 - weight)

            masked_grads[name] = grad * weight

        info = {
            'avg_importance': avg_importance,
            'layer_corr': layer_correlation.item(),
            'contrastive_loss': contrastive_loss.item(),
            'consistency_loss': consistency_loss,
        }
        return masked_grads, info

    def get_layer_importance(self) -> Dict[str, float]:
        """获取层重要性分数"""
        return self.layer_importance.copy()


def test_dgmm():
    """测试 DGMM 框架"""
    print("Testing DGMM Framework...")

    dgmm = DGMMFramework(
        encoder_hidden_dim=128,
        encoder_output_dim=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    test_grads = {
        'layer1.weight': torch.randn(512, 512).to(dgmm.device).to(dgmm.dtype),
        'layer1.bias': torch.randn(512).to(dgmm.device).to(dgmm.dtype),
        'layer2.weight': torch.randn(256, 512).to(dgmm.device).to(dgmm.dtype),
        'layer2.bias': torch.randn(256).to(dgmm.device).to(dgmm.dtype),
    }

    masked_grads, info = dgmm.apply_mask(test_grads)

    print(f"Original gradients: {len(test_grads)}")
    print(f"Masked gradients: {len(masked_grads)}")
    print(f"Layer importance: {dgmm.get_layer_importance()}")
    print(f"Info: {info}")
    print("DGMM Framework test passed!")


if __name__ == "__main__":
    test_dgmm()
