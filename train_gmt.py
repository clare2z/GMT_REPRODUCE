import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
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


class MetaNetwork(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, grad: torch.Tensor, param_value: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        x = torch.cat([grad.unsqueeze(-1), param_value.unsqueeze(-1), history.unsqueeze(-1)], dim=-1)
        if x.dim() > 2:
            x = x.flatten(0, -2)
        x = F.relu(self.fc1(x))
        x = self.norm(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.norm(x)
        x = self.dropout(x)
        raw_score = self.fc3(x)
        return raw_score


class HistoryEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
    
    def forward(self, grad_history: torch.Tensor, update_history: torch.Tensor) -> torch.Tensor:
        x = torch.cat([grad_history, update_history], dim=-1)
        x = F.relu(self.fc(x))
        _, (h_n, _) = self.lstm(x.unsqueeze(0))
        return h_n.squeeze(0)


class DIGMTrainer:
    def __init__(
        self,
        model_name: str = "/Data/zhengtingyu/models/gpt2",
        device: str = "cuda",
        accumulation_steps: int = 8,
        learning_rate: float = 2e-5,
        use_quantization: bool = False,
        load_in_4bit: bool = True,
        num_epochs: int = 3,
        
        meta_hidden_dim: int = 64,
        history_window: int = 5,
        beta: float = 0.9,
        tau: float = 0.5,
        meta_weight: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.learning_rate = learning_rate
        self.use_quantization = use_quantization
        self.load_in_4bit = load_in_4bit
        self.num_epochs = num_epochs
        
        self.meta_hidden_dim = meta_hidden_dim
        self.history_window = history_window
        self.beta = beta
        self.tau = tau
        self.meta_weight = meta_weight
        
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
        self.meta_network: Optional[MetaNetwork] = None
        self.history_encoder: Optional[HistoryEncoder] = None
        self.meta_optimizer: Optional[torch.optim.Optimizer] = None
        
        self.param_history: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        
        self._validate_parameters()
        self._initialize_components()
    
    def _validate_parameters(self) -> None:
        if self.accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {self.accumulation_steps}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not (0 <= self.beta <= 1):
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        if not (0 <= self.tau <= 1):
            raise ValueError(f"tau must be in [0, 1], got {self.tau}")
        
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.use_quantization = False
    
    def _initialize_components(self) -> None:
        logger.info(f"===== Initializing DIGM Trainer =====")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Quantization: {'4-bit' if self.load_in_4bit else 'None'}")
        logger.info(f"Gradient accumulation steps: {self.accumulation_steps}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Meta network hidden dim: {self.meta_hidden_dim}")
        logger.info(f"History window size: {self.history_window}")
        logger.info(f"EMA beta: {self.beta}")
        logger.info(f"Threshold tau: {self.tau}")
        logger.info(f"Meta loss weight: {self.meta_weight}")
        
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
        
        self._initialize_param_history()
        self._initialize_meta_network()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        logger.info("Optimizer initialized")
    
    def _initialize_param_history(self) -> None:
        for name, param in self.model.named_parameters():
            self.param_history[name] = []
    
    def _initialize_meta_network(self) -> None:
        self.meta_network = MetaNetwork(
            input_dim=3,
            hidden_dim=self.meta_hidden_dim,
            output_dim=1
        ).to(self.device)
        
        self.history_encoder = HistoryEncoder(
            input_dim=2,
            hidden_dim=self.meta_hidden_dim // 2
        ).to(self.device)
        
        self.meta_optimizer = torch.optim.AdamW(
            list(self.meta_network.parameters()) + 
            list(self.history_encoder.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )
        logger.info("Meta network and history encoder initialized")
    
    def _update_param_history(self, accumulated_grads: Dict[torch.Tensor, torch.Tensor]) -> None:
        for name, param in self.model.named_parameters():
            if param in accumulated_grads and accumulated_grads[param] is not None:
                grad_norm = float(torch.norm(accumulated_grads[param]))
                update_norm = float(torch.norm(accumulated_grads[param] / self.accumulation_steps))
                
                self.param_history[name].append({
                    'grad_norm': grad_norm,
                    'update_norm': update_norm,
                    'step': len(self.param_history[name])
                })
                
                if len(self.param_history[name]) > self.history_window:
                    self.param_history[name].pop(0)
    
    def _compute_history_representation(self, param_name: str) -> torch.Tensor:
        history = self.param_history.get(param_name, [])
        
        if len(history) == 0:
            return torch.tensor(0.0, device=self.device)
        
        grad_norms = torch.tensor([h['grad_norm'] for h in history], device=self.device)
        update_norms = torch.tensor([h['update_norm'] for h in history], device=self.device)
        
        if len(history) == 1:
            return grad_norms.mean()
        
        ema_grad = torch.tensor(0.0, device=self.device)
        ema_update = torch.tensor(0.0, device=self.device)
        
        for i, h in enumerate(history):
            ema_grad = self.beta * ema_grad + (1 - self.beta) * h['grad_norm']
            ema_update = self.beta * ema_update + (1 - self.beta) * h['update_norm']
        
        return torch.tensor(ema_grad, device=self.device)
    
    def _predict_importance(self, accumulated_grads: Dict[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        importance_dict = {}
        
        for name, param in self.model.named_parameters():
            if param in accumulated_grads and accumulated_grads[param] is not None:
                grad = accumulated_grads[param]
                grad_norm = torch.norm(grad)
                
                param_value_norm = torch.norm(param.data)
                
                history_rep = self._compute_history_representation(name)
                
                importance = self.meta_network(
                    grad_norm,
                    param_value_norm,
                    history_rep
                )
                
                importance_dict[name] = importance.squeeze()
        
        return importance_dict
    
    def _apply_digm_mask(self, accumulated_grads: Dict[torch.Tensor, torch.Tensor]) -> None:
        importance = self._predict_importance(accumulated_grads)
        
        if not importance:
            return
        
        all_scores = torch.tensor(list(importance.values()), device=self.device)
        adaptive_threshold = torch.median(all_scores)
        
        for name, param in self.model.named_parameters():
            if param not in accumulated_grads or accumulated_grads[param] is None:
                continue
            
            imp = importance.get(name, adaptive_threshold)
            
            if hasattr(param, 'prev_importance'):
                smoothed_imp = self.beta * param.prev_importance + (1 - self.beta) * imp
            else:
                smoothed_imp = imp
            
            param.prev_importance = imp
            
            mask = smoothed_imp >= adaptive_threshold
            
            if hasattr(param, 'gmt_mask') and param.gmt_mask is not None:
                param.prev_gmt_mask = param.gmt_mask.clone()
            else:
                param.prev_gmt_mask = None
            
            param.gmt_mask = mask
            param.grad = param.grad * mask.float()
        
        self._update_param_history(accumulated_grads)
        logger.debug(f"Adaptive threshold: {adaptive_threshold.item():.4f}, "
                    f"Mean score: {all_scores.mean().item():.4f}, "
                    f"Std score: {all_scores.std().item():.4f}")
    
    def compute_gradient_energy_retention(self, accumulated_grads: Dict[torch.Tensor, torch.Tensor]) -> float:
        if not accumulated_grads:
            return 0.0
        
        importance = self._predict_importance(accumulated_grads)
        
        if not importance:
            return 0.0
        
        all_scores = torch.tensor(list(importance.values()), device=self.device)
        adaptive_threshold = torch.median(all_scores)
        
        total_energy_before = 0.0
        total_energy_after = 0.0
        
        for name, param in self.model.named_parameters():
            if param in accumulated_grads and accumulated_grads[param] is not None:
                grad = accumulated_grads[param]
                grad_abs = grad.abs()
                total_energy_before += torch.sum(grad_abs ** 2).item()
                
                imp = importance.get(name, adaptive_threshold)
                mask = imp >= adaptive_threshold
                total_energy_after += torch.sum((grad * mask.float()) ** 2).item()
        
        return total_energy_after / (total_energy_before + 1e-10) if total_energy_before > 0 else 0.0
    
    def compute_mask_stability(self) -> float:
        total_similarity = 0.0
        total_elements = 0
        
        for param in self.model.parameters():
            if not hasattr(param, 'gmt_mask') or param.gmt_mask is None:
                continue
            if not hasattr(param, 'prev_gmt_mask') or param.prev_gmt_mask is None:
                continue
            
            curr_mask = param.gmt_mask.flatten()
            prev_mask = param.prev_gmt_mask.flatten()
            
            intersection = torch.sum(curr_mask & prev_mask).item()
            union = torch.sum(curr_mask | prev_mask).item()
            
            if union > 0:
                total_similarity += (intersection / union) * curr_mask.numel()
                total_elements += curr_mask.numel()
        
        return total_similarity / total_elements if total_elements > 0 else 0.0
    
    def compute_layer_update_imbalance(self) -> Tuple[Dict[str, float], float, float]:
        layer_updates: Dict[str, float] = {}
        total_update = 0.0
        
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            
            layer_name = name.split('.')[0] if '.' in name else name
            update_norm = float(torch.norm(param.grad))
            
            if layer_name not in layer_updates:
                layer_updates[layer_name] = 0.0
            layer_updates[layer_name] += update_norm
            total_update += update_norm
        
        if total_update == 0:
            return layer_updates, 0.0, 0.0
        
        for layer in layer_updates:
            layer_updates[layer] /= total_update
        
        update_values = np.array(list(layer_updates.values()))
        variance = float(np.var(update_values))
        min_update = float(np.min(update_values))
        max_update = float(np.max(update_values))
        max_ratio = max_update / (min_update + 1e-10)
        
        return layer_updates, variance, max_ratio
    
    def _tokenize_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=128,
        )
        return inputs["input_ids"].to(self.device), inputs["attention_mask"].to(self.device)
    
    def train(self, texts: List[str], num_epochs: Optional[int] = None) -> dict:
        if not texts:
            raise ValueError("texts cannot be empty")
        
        epochs_to_run = num_epochs if num_epochs is not None else self.num_epochs
        
        logger.info(f"\n===== Starting DIGM Training =====")
        logger.info(f"Number of samples: {len(texts)}")
        logger.info(f"Number of epochs: {epochs_to_run}")
        
        training_history = {
            'loss': [],
            'gradient_energy_retention': [],
            'mask_stability': [],
            'layer_update_variance': [],
            'max_update_ratio': [],
            'layer_distributions': [],
        }
        
        for epoch in range(epochs_to_run):
            accumulated_grads: Dict[torch.Tensor, torch.Tensor] = {}
            step_count = 0
            total_loss = 0.0
            epoch_energy_retention = []
            epoch_mask_stabilities = []
            batches_per_epoch = len(texts) // self.accumulation_steps
            
            logger.info(f"\n--- Epoch {epoch + 1}/{epochs_to_run} ---")
            
            for text_idx, text in enumerate(texts):
                input_ids, attention_mask = self._tokenize_text(text)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss / self.accumulation_steps
                loss.backward()
                
                for param in self.model.parameters():
                    if param.grad is not None:
                        if param not in accumulated_grads:
                            accumulated_grads[param] = torch.zeros_like(param.grad)
                        accumulated_grads[param] += param.grad
                        param.grad = None
                
                step_count += 1
                total_loss += loss.item() * self.accumulation_steps
                
                if step_count % self.accumulation_steps == 0:
                    for param in self.model.parameters():
                        if param in accumulated_grads and accumulated_grads[param] is not None:
                            param.grad = accumulated_grads[param] / self.accumulation_steps
                    
                    energy_retention = self.compute_gradient_energy_retention(accumulated_grads)
                    epoch_energy_retention.append(energy_retention)
                    
                    self._apply_digm_mask(accumulated_grads)
                    
                    mask_stability = self.compute_mask_stability()
                    epoch_mask_stabilities.append(mask_stability)
                    
                    self.meta_optimizer.step()
                    self.meta_optimizer.zero_grad()
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accumulated_grads = {}
                    
                    batch_idx = step_count // self.accumulation_steps
                    logger.info(f"  Batch {batch_idx}/{batches_per_epoch}: "
                              f"Loss={total_loss/(text_idx+1):.4f}, "
                              f"EnergyRetention={energy_retention:.4f}, "
                              f"MaskStability={mask_stability:.4f}")
            
            avg_loss = total_loss / len(texts)
            avg_energy_retention = np.mean(epoch_energy_retention) if epoch_energy_retention else 0.0
            avg_mask_stability = np.mean(epoch_mask_stabilities) if epoch_mask_stabilities else 0.0
            layer_updates, update_variance, max_update_ratio = self.compute_layer_update_imbalance()
            
            training_history['loss'].append(avg_loss)
            training_history['gradient_energy_retention'].append(avg_energy_retention)
            training_history['mask_stability'].append(avg_mask_stability)
            training_history['layer_update_variance'].append(update_variance)
            training_history['max_update_ratio'].append(max_update_ratio)
            training_history['layer_distributions'].append(layer_updates)
            
            self._log_epoch_results(epoch + 1, avg_loss, avg_energy_retention, avg_mask_stability,
                                   layer_updates, update_variance, max_update_ratio)
        
        self._log_training_summary(training_history)
        logger.info("\n===== Training Completed =====")
        
        return training_history
    
    def _log_epoch_results(self, epoch: int, avg_loss: float, avg_energy_retention: float,
                          avg_mask_stability: float, layer_updates: Dict[str, float],
                          update_variance: float, max_update_ratio: float) -> None:
        logger.info(f"\n=== Epoch {epoch} Results ===")
        logger.info(f"Loss = {avg_loss:.4f}")
        logger.info(f"【指标1】梯度能量保留率: {avg_energy_retention:.4f}")
        logger.info(f"【指标2】Mask稳定性: {avg_mask_stability:.4f}")
        logger.info(f"【指标3】层级更新不平衡:")
        logger.info(f"        - 更新方差: {update_variance:.6f}")
        logger.info(f"        - 最大/最小更新比: {max_update_ratio:.2f}")
        logger.info(f"        - 各层更新分布:")
        for layer, ratio in sorted(layer_updates.items()):
            logger.info(f"          {layer}: {ratio:.4f}")
    
    def _log_training_summary(self, history: dict) -> None:
        logger.info("\n===== Training Summary =====")
        
        logger.info("\n[Loss]")
        for i, loss in enumerate(history['loss'], 1):
            logger.info(f"  Epoch {i}: {loss:.4f}")
        
        logger.info("\n[Gradient Energy Retention]")
        for i, val in enumerate(history['gradient_energy_retention'], 1):
            logger.info(f"  Epoch {i}: {val:.4f}")
        
        logger.info("\n[Mask Stability]")
        for i, val in enumerate(history['mask_stability'], 1):
            logger.info(f"  Epoch {i}: {val:.4f}")
        
        logger.info("\n[Layer Update Variance]")
        for i, val in enumerate(history['layer_update_variance'], 1):
            logger.info(f"  Epoch {i}: {val:.6f}")
        
        logger.info("\n[Max/Min Update Ratio]")
        for i, val in enumerate(history['max_update_ratio'], 1):
            logger.info(f"  Epoch {i}: {val:.2f}")


def main():
    logger.info("===== DIGM Training =====")
    
    trainer = DIGMTrainer(
        model_name="/Data/zhengtingyu/models/gpt2",
        device="cuda",
        accumulation_steps=8,
        learning_rate=2e-5,
        use_quantization=False,
        load_in_4bit=False,
        num_epochs=3,
        
        meta_hidden_dim=64,
        history_window=5,
        beta=0.9,
        tau=0.5,
        meta_weight=0.1,
    )
    
    texts = [
        "Life is short, be happy.",
        "Knowledge is power.",
        "Practice makes perfect.",
        "Stay hungry, stay foolish.",
    ] * 100
    
    trainer.train(texts)
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()