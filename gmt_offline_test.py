import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=500, d_model=128, n_head=2, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=512, batch_first=True)
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
    
    def forward(self, x, attention_mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc(x)
        return x


class GMTTrainer:
    def __init__(self, model, k_percent=50, accumulation_steps=4, lr=1e-4):
        self.model = model
        self.k_percent = k_percent
        self.accumulation_steps = accumulation_steps
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = next(model.parameters()).device
        
    def _compute_threshold(self, accumulated_grads):
        all_grad_values = []
        for param, grad in accumulated_grads.items():
            if grad is not None:
                all_grad_values.append(grad.abs().flatten())
        
        if not all_grad_values:
            return 0.0
        
        all_grads_flat = torch.cat(all_grad_values)
        k = int(len(all_grads_flat) * self.k_percent / 100)
        k = max(k, 1)
        threshold = torch.kthvalue(all_grads_flat, len(all_grads_flat) - k + 1).values
        return threshold
    
    def compute_gradient_energy_retention(self, accumulated_grads):
        if not accumulated_grads:
            return 0.0
        
        threshold = self._compute_threshold(accumulated_grads)
        total_before = 0.0
        total_after = 0.0
        
        for param, grad in accumulated_grads.items():
            if grad is not None:
                grad_abs = grad.abs()
                total_before += torch.sum(grad_abs ** 2).item()
                mask = grad_abs >= threshold
                total_after += torch.sum((grad * mask) ** 2).item()
        
        return total_after / (total_before + 1e-10)
    
    def compute_mask_stability(self, current_masks, previous_masks):
        if not current_masks or not previous_masks:
            return 0.0
        
        total_sim = 0.0
        total_elem = 0
        
        for name in current_masks:
            if name not in previous_masks:
                continue
            
            curr = current_masks[name].flatten()
            prev = previous_masks[name].flatten()
            intersection = torch.sum(curr & prev).item()
            union = torch.sum(curr | prev).item()
            
            if union > 0:
                total_sim += (intersection / union) * curr.numel()
                total_elem += curr.numel()
        
        return total_sim / total_elem if total_elem > 0 else 0.0
    
    def compute_layer_update_imbalance(self):
        layer_updates = {}
        total_update = 0.0
        
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            
            layer_name = name.split('.')[0]
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
    
    def train(self, data_loader, num_epochs=3):
        logger.info(f"\n===== Starting GMT Training =====")
        logger.info(f"Device: {self.device}")
        logger.info(f"k-percent: {self.k_percent}%")
        logger.info(f"Accumulation steps: {self.accumulation_steps}")
        logger.info(f"Learning rate: {self.lr}")
        
        previous_masks = {}
        
        for epoch in range(num_epochs):
            self.model.train()
            accumulated_grads = {}
            step_count = 0
            total_loss = 0.0
            
            logger.info(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            
            for step, (x, y) in enumerate(data_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), y.view(-1))
                loss = loss / self.accumulation_steps
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
                        if param in accumulated_grads:
                            param.grad = accumulated_grads[param] / self.accumulation_steps
                    
                    energy_retention = self.compute_gradient_energy_retention(accumulated_grads)
                    threshold = self._compute_threshold(accumulated_grads)
                    
                    current_masks = {}
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            mask = param.grad.abs() >= threshold
                            current_masks[name] = mask
                            param.grad = param.grad * mask
                    
                    mask_stability = self.compute_mask_stability(current_masks, previous_masks)
                    previous_masks = current_masks
                    
                    layer_updates, update_variance, max_ratio = self.compute_layer_update_imbalance()
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accumulated_grads = {}
                    
                    logger.info(f"  Step {step_count}: Loss={total_loss/(step+1):.4f}, "
                              f"EnergyRetention={energy_retention:.4f}, "
                              f"MaskStability={mask_stability:.4f}, "
                              f"UpdateVariance={update_variance:.6f}, "
                              f"MaxRatio={max_ratio:.2f}")
            
            avg_loss = total_loss / len(data_loader)
            logger.info(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")
        
        logger.info("\n===== Training Completed =====")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    vocab_size = 500
    seq_len = 32
    batch_size = 4
    
    model = SimpleTransformer(vocab_size=vocab_size).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 生成模拟数据
    data = [(torch.randint(0, vocab_size, (batch_size, seq_len)),
             torch.randint(0, vocab_size, (batch_size, seq_len)))
            for _ in range(20)]
    
    trainer = GMTTrainer(model, k_percent=50, accumulation_steps=4, lr=1e-4)
    trainer.train(data, num_epochs=3)


if __name__ == "__main__":
    main()
