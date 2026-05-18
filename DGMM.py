"""
DGMM (Dynamic Gradient Manifold Masking) 核心框架

基于对比学习和自适应掩码的梯度训练方法
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
from typing import Dict, Tuple, List, Optional


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
    """DGMM核心框架"""
    
    def __init__(
        self,
        encoder_hidden_dim: int = 128,
        encoder_output_dim: int = 64,
        contrastive_temperature: float = 0.5,
        contrastive_weight: float = 0.1,
        consistency_weight: float = 0.2,
        ema_alpha: float = 0.9,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        self.device = device
        self.dtype = dtype
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight
        self.ema_alpha = ema_alpha
        
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

        self.meta_optimizer = torch.optim.AdamW(
            list(self.gradient_encoder.parameters()) + 
            list(self.contrastive_learner.parameters()) + 
            list(self.layer_attention.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        self.layer_importance: Dict[str, float] = {}
        self.prev_layer_importance: Dict[str, float] = {}
        self.global_importance_threshold = 0.5

    def _compute_layer_gradients(self, accumulated_grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将梯度按层组织"""
        layer_grads = {}
        for name, grad in accumulated_grads.items():
            layer_name = name.split('.')[0]
            if layer_name not in layer_grads:
                layer_grads[layer_name] = []
            layer_grads[layer_name].append(grad.to(self.device).flatten())
        
        for layer_name in layer_grads:
            layer_grads[layer_name] = torch.cat(layer_grads[layer_name], dim=0)
        
        return layer_grads

    def _extract_layer_features(self, layer_grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        """提取层特征"""
        layer_features = []
        for layer_name in sorted(layer_grads.keys()):
            grad = layer_grads[layer_name]
            if grad.size(0) < self.encoder_hidden_dim:
                grad = F.pad(grad, (0, self.encoder_hidden_dim - grad.size(0)))
            elif grad.size(0) > self.encoder_hidden_dim:
                grad = grad[:self.encoder_hidden_dim]
            
            features = self.gradient_encoder(grad.unsqueeze(0).to(self.dtype))
            layer_features.append(features)
        
        return torch.cat(layer_features, dim=0)

    def _build_contrastive_samples(self, layer_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建对比学习样本"""
        anchors = layer_features
        positives = layer_features.roll(1, dims=0)
        negatives = layer_features[torch.randperm(layer_features.size(0))]
        return anchors, positives, negatives

    def apply_mask(self, accumulated_grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用DGMM掩码到梯度"""
        layer_grads = self._compute_layer_gradients(accumulated_grads)
        layer_features = self._extract_layer_features(layer_grads)
        
        anchors, positives, negatives = self._build_contrastive_samples(layer_features)
        contrastive_loss = self.contrastive_learner(anchors, positives, negatives)
        
        fused_features = self.layer_attention(layer_features)
        importance_scores = torch.sigmoid(torch.mean(fused_features, dim=-1))
        
        if hasattr(self, 'use_adaptive_threshold') and self.use_adaptive_threshold:
            self.global_importance_threshold = torch.mean(importance_scores).item()
        
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
        
        total_meta_loss = self.contrastive_weight * contrastive_loss + self.consistency_weight * consistency_loss
        
        self.meta_optimizer.zero_grad()
        total_meta_loss.backward()
        self.meta_optimizer.step()
        
        masked_grads = {}
        for name, grad in accumulated_grads.items():
            layer_name = name.split('.')[0]
            importance = self.layer_importance.get(layer_name, self.global_importance_threshold)
            
            mask = torch.rand(grad.size(), device=self.device) < importance
            masked_grads[name] = grad * mask.to(self.dtype)
        
        return masked_grads

    def get_layer_importance(self) -> Dict[str, float]:
        """获取层重要性分数"""
        return self.layer_importance.copy()


def test_dgmm():
    """测试DGMM框架"""
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
    
    masked_grads = dgmm.apply_mask(test_grads)
    
    print(f"Original gradients: {len(test_grads)}")
    print(f"Masked gradients: {len(masked_grads)}")
    print(f"Layer importance: {dgmm.get_layer_importance()}")
    print("DGMM Framework test passed!")


if __name__ == "__main__":
    test_dgmm()
