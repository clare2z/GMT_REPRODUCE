print("程序开始")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

model_name = "distilgpt2"
device = "cpu"

print("使用设备:", device)

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

def compute_gradient_energy_retention(model):
    total_energy_before = 0.0
    total_energy_after = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_abs = p.grad.abs()
            total_energy_before += torch.sum(grad_abs ** 2).item()
            mask = grad_abs >= torch.quantile(grad_abs.float().flatten(), 0.5)
            total_energy_after += torch.sum((p.grad * mask) ** 2).item()
    return total_energy_after / (total_energy_before + 1e-10) if total_energy_before > 0 else 0.0

def compute_mask_stability(current_masks, previous_masks):
    if previous_masks is None or len(current_masks) != len(previous_masks):
        return 0.0
    total_similarity = 0.0
    total_elements = 0
    for (name, curr_mask), (prev_name, prev_mask) in zip(current_masks.items(), previous_masks.items()):
        if name == prev_name:
            curr_flat = curr_mask.flatten()
            prev_flat = prev_mask.flatten()
            intersection = torch.sum(curr_flat & prev_flat).item()
            union = torch.sum(curr_flat | prev_flat).item()
            if union > 0:
                total_similarity += intersection / union * curr_flat.numel()
                total_elements += curr_flat.numel()
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

def apply_gmt_and_collect_masks(model, quantile=0.5):
    masks = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_abs = p.grad.abs()
            flat = grad_abs.float().flatten()
            sample_size = min(10000, flat.numel())
            indices = torch.randint(0, flat.numel(), (sample_size,))
            sample = flat[indices]
            threshold = torch.quantile(sample, quantile)
            mask = grad_abs >= threshold
            masks[name] = mask
            p.grad = p.grad * mask
    return masks

print("开始训练")

previous_masks = None

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

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        energy_retention = compute_gradient_energy_retention(model)
        epoch_energy_retention.append(energy_retention)
        
        current_masks = apply_gmt_and_collect_masks(model)
        
        if previous_masks is not None:
            mask_stability = compute_mask_stability(current_masks, previous_masks)
            epoch_mask_stabilities.append(mask_stability)
        
        previous_masks = current_masks
        
        optimizer.step()
        total_loss += loss.item()

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