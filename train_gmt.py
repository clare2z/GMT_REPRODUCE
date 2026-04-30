import torch
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class GMTTrainer:
    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: str = "cpu",
        k_percent: int = 50,
        accumulation_steps: int = 4,
        learning_rate: float = 5e-5,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.k_percent = k_percent
        self.accumulation_steps = accumulation_steps
        self.learning_rate = learning_rate
        
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
        self._validate_parameters()
        self._initialize_components()
    
    def _validate_parameters(self) -> None:
        if not (0 < self.k_percent <= 100):
            raise ValueError(f"k_percent must be in (0, 100], got {self.k_percent}")
        
        if self.accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {self.accumulation_steps}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        
        if self.device not in ["cpu", "cuda"]:
            raise ValueError(f"device must be 'cpu' or 'cuda', got {self.device}")
        
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
    
    def _initialize_components(self) -> None:
        logger.info(f"Initializing GMT trainer with model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        logger.info(f"Model loaded to {self.device}")
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        logger.info(f"Optimizer initialized with lr={self.learning_rate}")
    
    def _compute_threshold(self, accumulated_grads: Dict[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        all_grad_values = []
        for param, grad in accumulated_grads.items():
            if grad is not None:
                grad_abs = grad.abs()
                all_grad_values.append(grad_abs.flatten())
        
        if not all_grad_values:
            raise RuntimeError("No gradients found in accumulated_grads")
        
        all_grads_flat = torch.cat(all_grad_values)
        num_elements = len(all_grads_flat)
        
        k = int(num_elements * self.k_percent / 100)
        k = max(k, 1)
        
        threshold = torch.kthvalue(all_grads_flat, num_elements - k + 1).values
        return threshold
    
    def compute_gradient_energy_retention(
        self,
        accumulated_grads: Dict[torch.Tensor, torch.Tensor]
    ) -> float:
        if not accumulated_grads:
            return 0.0
        
        threshold = self._compute_threshold(accumulated_grads)
        
        total_energy_before = 0.0
        total_energy_after = 0.0
        
        for param, grad in accumulated_grads.items():
            if grad is not None:
                grad_abs = grad.abs()
                total_energy_before += torch.sum(grad_abs ** 2).item()
                
                mask = grad_abs >= threshold
                total_energy_after += torch.sum((grad * mask) ** 2).item()
        
        if total_energy_before == 0:
            return 0.0
        
        return total_energy_after / total_energy_before
    
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
    
    def _apply_gmt_mask(
        self,
        accumulated_grads: Dict[torch.Tensor, torch.Tensor],
        threshold: torch.Tensor
    ) -> None:
        for param in self.model.parameters():
            if param not in accumulated_grads or accumulated_grads[param] is None:
                continue
            
            grad_abs = accumulated_grads[param].abs()
            mask = grad_abs >= threshold
            
            if hasattr(param, 'gmt_mask') and param.gmt_mask is not None:
                param.prev_gmt_mask = param.gmt_mask.clone()
            else:
                param.prev_gmt_mask = None
            
            param.gmt_mask = mask
            param.grad = param.grad * mask
    
    def _tokenize_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )
        return inputs["input_ids"].to(self.device), inputs["attention_mask"].to(self.device)
    
    def train(self, texts: List[str], num_epochs: int = 2) -> None:
        if not texts:
            raise ValueError("texts cannot be empty")
        
        logger.info(f"Starting training with {len(texts)} samples, {num_epochs} epochs")
        logger.info(f"GMT configuration: k={self.k_percent}%, accumulation_steps={self.accumulation_steps}")
        
        for epoch in range(num_epochs):
            accumulated_grads: Dict[torch.Tensor, torch.Tensor] = {}
            step_count = 0
            total_loss = 0.0
            epoch_energy_retention = []
            epoch_mask_stabilities = []
            
            for text in texts:
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
                    
                    threshold = self._compute_threshold(accumulated_grads)
                    self._apply_gmt_mask(accumulated_grads, threshold)
                    
                    mask_stability = self.compute_mask_stability()
                    epoch_mask_stabilities.append(mask_stability)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    accumulated_grads = {}
            
            avg_loss = total_loss / len(texts)
            avg_energy_retention = np.mean(epoch_energy_retention) if epoch_energy_retention else 0.0
            avg_mask_stability = np.mean(epoch_mask_stabilities) if epoch_mask_stabilities else 0.0
            layer_updates, update_variance, max_update_ratio = self.compute_layer_update_imbalance()
            
            self._log_epoch_results(
                epoch + 1,
                avg_loss,
                avg_energy_retention,
                avg_mask_stability,
                layer_updates,
                update_variance,
                max_update_ratio
            )
        
        logger.info("Training completed successfully")
    
    def _log_epoch_results(
        self,
        epoch: int,
        avg_loss: float,
        avg_energy_retention: float,
        avg_mask_stability: float,
        layer_updates: Dict[str, float],
        update_variance: float,
        max_update_ratio: float,
    ) -> None:
        logger.info(f"\n=== Epoch {epoch} ===")
        logger.info(f"Loss = {avg_loss:.4f}")
        logger.info(f"【指标1】梯度能量保留率 (Gradient Energy Retention): {avg_energy_retention:.4f}")
        logger.info(f"【指标2】Mask稳定性 (Mask Stability): {avg_mask_stability:.4f}")
        logger.info(f"【指标3】层级更新不平衡 (Layer-wise Update Imbalance):")
        logger.info(f"        - 更新方差: {update_variance:.6f}")
        logger.info(f"        - 最大/最小更新比: {max_update_ratio:.2f}")
        logger.info(f"        - 各层更新分布:")
        for layer, ratio in sorted(layer_updates.items()):
            logger.info(f"          {layer}: {ratio:.4f}")


def main() -> None:
    logger.info("程序开始")
    
    trainer = GMTTrainer(
        model_name="distilgpt2",
        device="cpu",
        k_percent=50,
        accumulation_steps=4,
        learning_rate=5e-5,
    )
    
    texts = [
        "Life is short, be happy.",
        "Knowledge is power.",
        "Practice makes perfect.",
        "Stay hungry, stay foolish.",
    ] * 20
    
    trainer.train(texts, num_epochs=2)
    
    logger.info("程序结束")


if __name__ == "__main__":
    main()