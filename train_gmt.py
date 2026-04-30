print("程序开始")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

model_name = "distilgpt2"
device = "cpu"
k_percent = 50  # 保留前k%重要参数
accumulation_steps = 4  # N: 累积N个batch

print("使用设备:", device)
print(f"GMT参数: 保留前{k_percent}%重要参数, 累积{accumulation_steps}个batch")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

texts = [
    "Life is short, be happy.",
    "Knowledge is power.",
    "Practice makes perfect.",
    "Stay hungry, stay foolish.",
] * 20

def compute_gradient_energy_retention(model, accumulated_grads):
    total_energy_before = 0.0
    total_energy_after = 0.0
    
    all_grad_values = []
    for p in model.parameters():
        if p in accumulated_grads and accumulated_grads[p] is not None:
            grad_abs = accumulated_grads[p].abs()
            all_grad_values.append(grad_abs.flatten())
    
    if not all_grad_values:
        return 0.0
    
    all_grads_flat = torch.cat(all_grad_values)
    k = int(len(all_grads_flat) * k_percent / 100)
    k = max(k, 1)
    threshold = torch.kthvalue(all_grads_flat, len(all_grads_flat) - k + 1).values
    
    for p in model.parameters():
        if p in accumulated_grads and accumulated_grads[p] is not None:
            grad_abs = accumulated_grads[p].abs()
            total_energy_before += torch.sum(grad_abs ** 2).item()
            mask = grad_abs >= threshold
            total_energy_after += torch.sum((accumulated_grads[p] * mask) ** 2).item()
    
    return total_energy_after / (total_energy_before + 1e-10) if total_energy_before > 0 else 0.0

def compute_mask_stability(model):
    total_similarity = 0.0
    total_elements = 0
    for p in model.parameters():
        if hasattr(p, 'gmt_mask') and p.gmt_mask is not None:
            if hasattr(p, 'prev_gmt_mask') and p.prev_gmt_mask is not None:
                curr_mask = p.gmt_mask.flatten()
                prev_mask = p.prev_gmt_mask.flatten()
                intersection = torch.sum(curr_mask & prev_mask).item()
                union = torch.sum(curr_mask | prev_mask).item()
                if union > 0:
                    total_similarity += intersection / union * curr_mask.numel()
                    total_elements += curr_mask.numel()
    return total_similarity / total_elements if total_elements > 0 else 0.0

def compute_layer_update_imbalance(model):
    layer_updates = {}
    total_update = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            layer_name = name.split('.')[0] if '.' in name else name
            update_norm = torch.norm(p.grad).item()
            if layer_name not in layer_updates:
                layer_updates[layer_name] = 0.0
            layer_updates[layer_name] += update_norm
            total_update += update_norm
    if total_update == 0:
        return layer_updates, 0.0, 0.0
    for layer in layer_updates:
        layer_updates[layer] /= total_update
    update_values = np.array(list(layer_updates.values()))
    variance = np.var(update_values)
    max_ratio = np.max(update_values) / (np.min(update_values) + 1e-10)
    return layer_updates, variance, max_ratio

print("开始训练")

accumulated_grads = {}
step_count = 0

for epoch in range(2):
    total_loss = 0
    epoch_energy_retention = []
    epoch_mask_stabilities = []
    
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
        )

        loss = outputs.loss / accumulation_steps
        loss.backward()
        
        for p in model.parameters():
            if p.grad is not None:
                if p not in accumulated_grads:
                    accumulated_grads[p] = torch.zeros_like(p.grad)
                accumulated_grads[p] += p.grad
                p.grad = None
        
        step_count += 1
        total_loss += loss.item() * accumulation_steps
        
        if step_count % accumulation_steps == 0:
            for p in model.parameters():
                if p in accumulated_grads and accumulated_grads[p] is not None:
                    p.grad = accumulated_grads[p] / accumulation_steps
            
            energy_retention = compute_gradient_energy_retention(model, accumulated_grads)
            epoch_energy_retention.append(energy_retention)
            
            all_grad_values = []
            all_params = []
            for p in model.parameters():
                if p in accumulated_grads and accumulated_grads[p] is not None:
                    grad_abs = accumulated_grads[p].abs()
                    all_grad_values.append(grad_abs.flatten())
                    all_params.append(p)
            
            if all_grad_values:
                all_grads_flat = torch.cat(all_grad_values)
                k = int(len(all_grads_flat) * k_percent / 100)
                k = max(k, 1)
                threshold = torch.kthvalue(all_grads_flat, len(all_grads_flat) - k + 1).values
                
                for p in all_params:
                    grad_abs = accumulated_grads[p].abs()
                    mask = grad_abs >= threshold
                    
                    if hasattr(p, 'gmt_mask') and p.gmt_mask is not None:
                        p.prev_gmt_mask = p.gmt_mask.clone()
                    else:
                        p.prev_gmt_mask = None
                    
                    p.gmt_mask = mask
                    p.grad = p.grad * mask
            
            mask_stability = compute_mask_stability(model)
            epoch_mask_stabilities.append(mask_stability)
            
            optimizer.step()
            optimizer.zero_grad()
            
            accumulated_grads = {}
    
    avg_loss = total_loss / len(texts)
    avg_energy_retention = sum(epoch_energy_retention) / len(epoch_energy_retention) if epoch_energy_retention else 0.0
    avg_mask_stability = sum(epoch_mask_stabilities) / len(epoch_mask_stabilities) if epoch_mask_stabilities else 0.0
    
    layer_updates, update_variance, max_update_ratio = compute_layer_update_imbalance(model)
    
    print(f"\n=== Epoch {epoch + 1} ===")
    print(f"Loss = {avg_loss:.4f}")
    print(f"【指标1】梯度能量保留率 (Gradient Energy Retention): {avg_energy_retention:.4f}")
    print(f"【指标2】Mask稳定性 (Mask Stability): {avg_mask_stability:.4f}")
    print(f"【指标3】层级更新不平衡 (Layer-wise Update Imbalance):")
    print(f"        - 更新方差: {update_variance:.6f}")
    print(f"        - 最大/最小更新比: {max_update_ratio:.2f}")
    print(f"        - 各层更新分布:")
    for layer, ratio in sorted(layer_updates.items()):
        print(f"          {layer}: {ratio:.4f}")

print("\n训练结束")