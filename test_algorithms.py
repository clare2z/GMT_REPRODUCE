import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import csv
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    def __init__(self, vocab_size=500, d_model=128, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.fc(x)
        return x


class SFTTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        count = 0
        for x, y in dataloader:
            outputs = self.model(x)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), y.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        return total_loss / count if count > 0 else 0.0


class DropTrainer:
    def __init__(self, model, drop_rate=0.1, lr=1e-4):
        self.model = model
        self.drop_rate = drop_rate
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        count = 0
        for x, y in dataloader:
            outputs = self.model(x)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), y.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            
            for param in self.model.parameters():
                if param.grad is not None:
                    mask = torch.rand_like(param.grad) > self.drop_rate
                    param.grad = param.grad * mask
            
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        return total_loss / count if count > 0 else 0.0


class HFTTrainer:
    def __init__(self, model, top_k=50, lr=1e-4):
        self.model = model
        self.top_k = top_k
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        count = 0
        for x, y in dataloader:
            outputs = self.model(x)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), y.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            
            all_grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads.append(param.grad.abs().flatten())
            
            if all_grads:
                all_grads_flat = torch.cat(all_grads)
                k = int(len(all_grads_flat) * self.top_k / 100)
                threshold = torch.kthvalue(all_grads_flat, len(all_grads_flat) - k + 1).values
                
                for param in self.model.parameters():
                    if param.grad is not None:
                        mask = param.grad.abs() >= threshold
                        param.grad = param.grad * mask.float()
            
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        return total_loss / count if count > 0 else 0.0


class RMTTrainer:
    def __init__(self, model, momentum=0.9, lr=1e-4):
        self.model = model
        self.momentum = momentum
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.grad_momentum = {}
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        count = 0
        for x, y in dataloader:
            outputs = self.model(x)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), y.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name not in self.grad_momentum:
                        self.grad_momentum[name] = torch.zeros_like(param.grad)
                    self.grad_momentum[name] = self.momentum * self.grad_momentum[name] + (1 - self.momentum) * param.grad
                    param.grad = self.grad_momentum[name]
            
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        return total_loss / count if count > 0 else 0.0


class GMTTrainer:
    def __init__(self, model, k_percent=50, accumulation_steps=2, lr=1e-4):
        self.model = model
        self.k_percent = k_percent
        self.accumulation_steps = accumulation_steps
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        count = 0
        accumulated_grads = {}
        step_count = 0
        
        for x, y in dataloader:
            outputs = self.model(x)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), y.view(-1)) / self.accumulation_steps
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name not in accumulated_grads:
                        accumulated_grads[name] = torch.zeros_like(param.grad)
                    accumulated_grads[name] += param.grad
                    param.grad = None
            
            total_loss += loss.item() * self.accumulation_steps
            count += 1
            step_count += 1
            
            if step_count % self.accumulation_steps == 0:
                all_grad_values = []
                for grad in accumulated_grads.values():
                    all_grad_values.append(grad.abs().flatten())
                
                if all_grad_values:
                    all_grads_flat = torch.cat(all_grad_values)
                    k = int(len(all_grads_flat) * self.k_percent / 100)
                    threshold = torch.kthvalue(all_grads_flat, len(all_grads_flat) - k + 1).values
                    
                    for name, param in self.model.named_parameters():
                        if name in accumulated_grads:
                            param.grad = accumulated_grads[name] / self.accumulation_steps
                            mask = param.grad.abs() >= threshold
                            param.grad = param.grad * mask
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulated_grads = {}
        
        return total_loss / count if count > 0 else 0.0


class DGMMTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.layer_importance = {}
        self.prev_layer_importance = {}
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        count = 0
        
        for x, y in dataloader:
            outputs = self.model(x)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), y.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            
            accumulated_grads = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    accumulated_grads[name] = param.grad.clone().detach()
            
            if accumulated_grads:
                layer_importance = {}
                for name in accumulated_grads:
                    layer_name = name.split('.')[0]
                    grad_norm = torch.norm(accumulated_grads[name])
                    layer_importance[layer_name] = (layer_importance.get(layer_name, 0) + grad_norm.item()) / 2
                
                for layer_name in layer_importance:
                    if layer_name in self.prev_layer_importance:
                        layer_importance[layer_name] = 0.9 * self.prev_layer_importance[layer_name] + 0.1 * layer_importance[layer_name]
                
                for name, param in self.model.named_parameters():
                    if name in accumulated_grads:
                        layer_name = name.split('.')[0]
                        importance = layer_importance.get(layer_name, 0.5)
                        mask = torch.rand(param.grad.size()) < importance
                        param.grad = param.grad * mask.float()
                
                self.prev_layer_importance = layer_importance
            
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        
        return total_loss / count if count > 0 else 0.0


def generate_sample_data(vocab_size=500, seq_len=32, batch_size=4, num_samples=20):
    data = []
    for _ in range(num_samples):
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))
        data.append((x, y))
    return data


def main():
    logger.info("===== Testing Algorithms =====")
    
    vocab_size = 500
    num_epochs = 3
    
    data = generate_sample_data(vocab_size=vocab_size)
    
    algorithms = ["SFT", "Drop", "HFT", "RMT", "GMT", "DGMM"]
    results = []
    
    for algorithm in algorithms:
        logger.info(f"\n--- Testing {algorithm} ---")
        model = SimpleModel(vocab_size=vocab_size)
        
        if algorithm == "SFT":
            trainer = SFTTrainer(model)
        elif algorithm == "Drop":
            trainer = DropTrainer(model)
        elif algorithm == "HFT":
            trainer = HFTTrainer(model)
        elif algorithm == "RMT":
            trainer = RMTTrainer(model)
        elif algorithm == "GMT":
            trainer = GMTTrainer(model)
        elif algorithm == "DGMM":
            trainer = DGMMTrainer(model)
        else:
            continue
        
        for epoch in range(num_epochs):
            loss = trainer.train_epoch(data)
            logger.info(f"  Epoch {epoch+1}/{num_epochs}: Loss = {loss:.4f}")
        
        results.append({"Algorithm": algorithm, "Final Loss": loss})
    
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"results/test_results_{timestamp}.csv"
    
    with open(csv_filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Final Loss"])
        for result in results:
            writer.writerow([result["Algorithm"], f"{result['Final Loss']:.4f}"])
    
    logger.info(f"\n===== Results saved to {csv_filename} =====")
    print("\nResults Summary:")
    for result in results:
        print(f"{result['Algorithm']}: Final Loss = {result['Final Loss']:.4f}")


if __name__ == "__main__":
    main()