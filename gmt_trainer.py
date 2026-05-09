import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import logging
import csv
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentCSVLogger:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.results = []
        self.headers = ["timestamp", "epoch", "step", "loss", 
                        "gradient_energy_retention", "mask_stability", 
                        "layer_update_variance", "max_update_ratio"]
        
    def log(self, epoch, step, loss, energy_retention, mask_stability, variance, max_ratio):
        self.results.append({
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "gradient_energy_retention": energy_retention,
            "mask_stability": mask_stability,
            "layer_update_variance": variance,
            "max_update_ratio": max_ratio
        })
    
    def save(self):
        filename = f"results/{self.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # 创建目录
        import os
        os.makedirs("results", exist_ok=True)
        
        with open(filename, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(self.results)
        
        logger.info(f"Results saved to {filename}")
        return filename


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=500, d_model=128, n_head=2, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=512, batch_first=True)
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc(x)
        return x


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 初始化模型
    vocab_size = 500
    model = SimpleTransformer(vocab_size=vocab_size).to(device)
    
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 初始化CSV记录器
    csv_logger = ExperimentCSVLogger("gmt_experiment")
    
    # 模拟数据
    seq_len = 32
    batch_size = 4
    data = [(torch.randint(0, vocab_size, (batch_size, seq_len)),
             torch.randint(0, vocab_size, (batch_size, seq_len)))
            for _ in range(20)]
    
    # 训练参数
    k_percent = 50
    accumulation_steps = 4
    
    # 训练循环
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        accumulated_grads = {}
        step_count = 0
        total_loss = 0.0
        
        logger.info(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        for step, (x, y) in enumerate(data):
            x, y = x.to(device), y.to(device)
            
            outputs = model(x)
            loss = F.cross_entropy(outputs.view(-1, vocab_size), y.view(-1))
            loss = loss / accumulation_steps
            loss.backward()
            
            # 累积梯度
            for param in model.parameters():
                if param.grad is not None:
                    if param not in accumulated_grads:
                        accumulated_grads[param] = torch.zeros_like(param.grad)
                    accumulated_grads[param] += param.grad
                    param.grad = None
            
            step_count += 1
            total_loss += loss.item() * accumulation_steps
            
            if step_count % accumulation_steps == 0:
                # 计算平均梯度
                for param in model.parameters():
                    if param in accumulated_grads:
                        param.grad = accumulated_grads[param] / accumulation_steps
                
                # 计算阈值
                all_grad_values = []
                for param, grad in accumulated_grads.items():
                    if grad is not None:
                        all_grad_values.append(grad.abs().flatten())
                all_grads_flat = torch.cat(all_grad_values)
                k = int(len(all_grads_flat) * k_percent / 100)
                threshold = torch.kthvalue(all_grads_flat, len(all_grads_flat) - k + 1).values
                
                # 计算能量保留率
                total_before = sum(torch.sum(g.abs()**2).item() for g in accumulated_grads.values() if g is not None)
                total_after = 0
                for param, grad in accumulated_grads.items():
                    if grad is not None:
                        mask = grad.abs() >= threshold
                        total_after += torch.sum((grad * mask)**2).item()
                energy_retention = total_after / (total_before + 1e-10)
                
                # 应用mask
                for param in model.parameters():
                    if param.grad is not None:
                        mask = param.grad.abs() >= threshold
                        param.grad = param.grad * mask
                
                # 计算层级更新不平衡
                layer_updates = {}
                total_update = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        layer_name = name.split('.')[0]
                        update_norm = float(torch.norm(param.grad))
                        layer_updates[layer_name] = layer_updates.get(layer_name, 0) + update_norm
                        total_update += update_norm
                
                update_values = np.array(list(layer_updates.values())) if layer_updates else np.array([0])
                variance = float(np.var(update_values))
                max_ratio = float(np.max(update_values)) / (float(np.min(update_values)) + 1e-10)
                
                # 记录到CSV
                csv_logger.log(
                    epoch=epoch+1,
                    step=step_count,
                    loss=total_loss/(step+1),
                    energy_retention=energy_retention,
                    mask_stability=0.0,  # 需要前一次mask才能计算
                    variance=variance,
                    max_ratio=max_ratio
                )
                
                optimizer.step()
                optimizer.zero_grad()
                accumulated_grads = {}
                
                logger.info(f"  Step {step_count}: Loss={total_loss/(step+1):.4f}, "
                          f"EnergyRetention={energy_retention:.4f}, "
                          f"UpdateVariance={variance:.6f}, "
                          f"MaxRatio={max_ratio:.2f}")
        
        logger.info(f"Epoch {epoch+1} completed.")
    
    # 保存CSV
    csv_logger.save()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
