import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import sys
if sys.version_info >= (3, 13):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias
from typing import Dict, Tuple, Optional, List
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    PreTrainedModel, 
    PreTrainedTokenizer,
    BitsAndBytesConfig
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class GradientEncoder(nn.Module):
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
    def __init__(self, feature_dim: int = 64, num_layers: int = 12):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, layer_features: torch.Tensor) -> torch.Tensor:
        queries = self.query_proj(layer_features)
        keys = self.key_proj(layer_features)
        values = self.value_proj(layer_features)
        
        attn_scores = torch.bmm(queries.unsqueeze(1), keys.unsqueeze(2)).squeeze() / np.sqrt(layer_features.size(-1))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        fused = torch.einsum('bld,bd->bl', layer_features, attn_weights)
        return self.output_proj(fused)


class DynamicGradientManifoldTrainer:
    def __init__(
        self,
        model_name: str = "/Data/zhengtingyu/models/gpt2",
        device: str = "cuda",
        accumulation_steps: int = 8,
        learning_rate: float = 2e-5,
        use_quantization: bool = False,
        load_in_4bit: bool = True,
        num_epochs: int = 3,
        
        encoder_hidden_dim: int = 128,
        encoder_output_dim: int = 64,
        contrastive_temperature: float = 0.5,
        contrastive_weight: float = 0.1,
        consistency_weight: float = 0.2,
        ema_alpha: float = 0.9,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.learning_rate = learning_rate
        self.use_quantization = use_quantization
        self.load_in_4bit = load_in_4bit
        self.num_epochs = num_epochs
        
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight
        self.ema_alpha = ema_alpha
        
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
        self.gradient_encoder: Optional[GradientEncoder] = None
        self.contrastive_learner: Optional[ContrastiveLearner] = None
        self.layer_attention: Optional[LayerAttentionFusion] = None
        self.meta_optimizer: Optional[torch.optim.Optimizer] = None
        
        self.layer_importance: Dict[str, float] = {}
        self.prev_layer_importance: Dict[str, float] = {}
        
        self._validate_parameters()
        self._initialize_components()
    
    def _validate_parameters(self) -> None:
        if self.accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {self.accumulation_steps}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not (0 <= self.ema_alpha <= 1):
            raise ValueError(f"ema_alpha must be in [0, 1], got {self.ema_alpha}")
        if not (0 <= self.contrastive_weight <= 1):
            raise ValueError(f"contrastive_weight must be in [0, 1], got {self.contrastive_weight}")
        
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.use_quantization = False
    
    def _initialize_components(self) -> None:
        logger.info(f"===== Initializing DGMM Trainer =====")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Quantization: {'4-bit' if self.load_in_4bit else 'None'}")
        logger.info(f"Gradient accumulation steps: {self.accumulation_steps}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Encoder hidden dim: {self.encoder_hidden_dim}")
        logger.info(f"Encoder output dim: {self.encoder_output_dim}")
        logger.info(f"Contrastive temperature: {self.contrastive_temperature}")
        logger.info(f"Contrastive weight: {self.contrastive_weight}")
        logger.info(f"Consistency weight: {self.consistency_weight}")
        logger.info(f"EMA alpha: {self.ema_alpha}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
        
        quantization_config = None
        if self.use_quantization and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            logger.info("Loaded 4-bit quantization config")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Number of parameters: {self.model.num_parameters():,}")
        
        self._initialize_meta_components()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        logger.info("Optimizer initialized")
    
    def _initialize_meta_components(self) -> None:
        self.gradient_encoder = GradientEncoder(
            input_dim=self.encoder_hidden_dim,
            hidden_dim=self.encoder_hidden_dim,
            output_dim=self.encoder_output_dim
        ).to(self.device)
        
        self.contrastive_learner = ContrastiveLearner(
            encoder_dim=self.encoder_output_dim,
            temperature=self.contrastive_temperature
        ).to(self.device)
        
        self.layer_attention = LayerAttentionFusion(
            feature_dim=self.encoder_output_dim,
            num_layers=12
        ).to(self.device)
        
        self.meta_optimizer = torch.optim.AdamW(
            list(self.gradient_encoder.parameters()) + 
            list(self.contrastive_learner.parameters()) +
            list(self.layer_attention.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )
        logger.info("DGMM components (GradientEncoder, ContrastiveLearner, LayerAttention) initialized")
    
    def _accumulate_gradients(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        accumulated_grads = {}
        self.model.zero_grad()
        
        for i, text in enumerate(texts):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss / self.accumulation_steps
            loss.backward()
            
            if (i + 1) % self.accumulation_steps == 0 or i == len(texts) - 1:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if name not in accumulated_grads:
                            accumulated_grads[name] = param.grad.clone().detach()
                        else:
                            accumulated_grads[name] += param.grad.clone().detach()
        
        return accumulated_grads
    
    def _compute_layer_gradients(self, accumulated_grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        layer_features = []
        
        for layer_name in sorted(layer_grads.keys()):
            grad = layer_grads[layer_name]
            if grad.size(0) < self.encoder_hidden_dim:
                grad = F.pad(grad, (0, self.encoder_hidden_dim - grad.size(0)))
            elif grad.size(0) > self.encoder_hidden_dim:
                grad = grad[:self.encoder_hidden_dim]
            
            features = self.gradient_encoder(grad.unsqueeze(0))
            layer_features.append(features)
        
        return torch.cat(layer_features, dim=0)
    
    def _build_contrastive_samples(self, layer_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors = layer_features
        positives = layer_features.roll(1, dims=0)
        negatives = layer_features[torch.randperm(layer_features.size(0))]
        
        return anchors, positives, negatives
    
    def _apply_dgmm_mask(self, accumulated_grads: Dict[str, torch.Tensor]) -> None:
        layer_grads = self._compute_layer_gradients(accumulated_grads)
        layer_features = self._extract_layer_features(layer_grads)
        
        anchors, positives, negatives = self._build_contrastive_samples(layer_features)
        contrastive_loss = self.contrastive_learner(anchors, positives, negatives)
        
        fused_features = self.layer_attention(layer_features)
        
        importance_scores = torch.sigmoid(torch.mean(fused_features, dim=-1))
        
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
        
        for name, param in self.model.named_parameters():
            if name in accumulated_grads and accumulated_grads[name] is not None:
                layer_name = name.split('.')[0]
                importance = self.layer_importance.get(layer_name, 0.5)
                
                mask = torch.rand(param.grad.size(), device=self.device) < importance
                param.grad = param.grad * mask.float()
        
        logger.debug(f"Contrastive loss: {contrastive_loss.item():.4f}, "
                    f"Consistency loss: {consistency_loss:.4f}, "
                    f"Mean importance: {torch.mean(importance_scores).item():.4f}")
    
    def compute_gradient_energy_retention(self, accumulated_grads: Dict[str, torch.Tensor]) -> float:
        if not accumulated_grads:
            return 0.0
        
        layer_grads = self._compute_layer_gradients(accumulated_grads)
        layer_features = self._extract_layer_features(layer_grads)
        
        fused_features = self.layer_attention(layer_features)
        importance_scores = torch.sigmoid(torch.mean(fused_features, dim=-1))
        
        total_energy_before = 0.0
        total_energy_after = 0.0
        
        layer_names = sorted(layer_grads.keys())
        for i, name in enumerate(layer_names):
            if name in accumulated_grads and accumulated_grads[name] is not None:
                grad = accumulated_grads[name]
                total_energy_before += torch.sum(grad ** 2).item()
                
                importance = importance_scores[i].item()
                mask = torch.rand(grad.size(), device=self.device) < importance
                total_energy_after += torch.sum((grad * mask.float()) ** 2).item()
        
        return total_energy_after / (total_energy_before + 1e-10) if total_energy_before > 0 else 0.0
    
    def compute_mask_stability(self) -> float:
        if not self.layer_importance or not self.prev_layer_importance:
            return 0.0
        
        stability_sum = 0.0
        count = 0
        
        for layer_name in self.layer_importance:
            if layer_name in self.prev_layer_importance:
                stability_sum += 1 - abs(self.layer_importance[layer_name] - self.prev_layer_importance[layer_name])
                count += 1
        
        return stability_sum / count if count > 0 else 0.0
    
    def compute_layer_update_imbalance(self) -> float:
        if not self.layer_importance:
            return 0.0
        
        importances = np.array(list(self.layer_importance.values()))
        return np.var(importances)
    
    def train(self, texts: List[str]) -> Dict[str, List[float]]:
        history = {
            'gradient_energy_retention': [],
            'mask_stability': [],
            'layer_update_imbalance': [],
            'loss': []
        }
        
        logger.info(f"===== Starting DGMM Training =====")
        
        for epoch in range(self.num_epochs):
            logger.info(f"\n=== Epoch {epoch + 1}/{self.num_epochs} ===")
            
            accumulated_grads = self._accumulate_gradients(texts)
            
            energy_retention = self.compute_gradient_energy_retention(accumulated_grads)
            mask_stability = self.compute_mask_stability()
            layer_imbalance = self.compute_layer_update_imbalance()
            
            self._apply_dgmm_mask(accumulated_grads)
            
            self.optimizer.step()
            self.model.zero_grad()
            
            history['gradient_energy_retention'].append(energy_retention)
            history['mask_stability'].append(mask_stability)
            history['layer_update_imbalance'].append(layer_imbalance)
            
            logger.info(f"Gradient Energy Retention: {energy_retention:.4f}")
            logger.info(f"Mask Stability: {mask_stability:.4f}")
            logger.info(f"Layer Update Imbalance: {layer_imbalance:.6f}")
        
        logger.info("\n===== Training Complete =====")
        return history


if __name__ == "__main__":
    trainer = DynamicGradientManifoldTrainer(
        model_name="/Data/zhengtingyu/models/gpt2",
        device="cuda",
        accumulation_steps=8,
        learning_rate=2e-5,
        encoder_hidden_dim=128,
        encoder_output_dim=64,
        contrastive_temperature=0.5,
        contrastive_weight=0.1,
        consistency_weight=0.2,
        ema_alpha=0.9
    )
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Gradient masking techniques improve training efficiency."
    ]
    
    history = trainer.train(sample_texts)
    
    print("\nTraining History:")
    print(f"Gradient Energy Retention: {history['gradient_energy_retention']}")
    print(f"Mask Stability: {history['mask_stability']}")
    print(f"Layer Update Imbalance: {history['layer_update_imbalance']}")