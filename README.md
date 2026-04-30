# GMT (Gradient Mask Tuning) Training Code

基于梯度掩码调优的大模型训练框架，通过选择性更新重要参数实现高效训练。

## 目录

- [算法原理](#算法原理)
- [核心代码讲解](#核心代码讲解)
- [测评指标详解](#测评指标详解)
- [代码结构](#代码结构)
- [使用方法](#使用方法)
- [参考文献](#参考文献)

## 算法原理

### 核心思想

使用梯度的大小来判断参数重要性，只更新"重要参数"，忽略不重要的参数，从而减少冗余更新并提升性能。

### 三个核心公式

**1. 梯度累积**
$$\Gamma_{ij} = \frac{1}{N} \sum_{n=1}^{N} \nabla_{\theta_{ij}} L(\Theta, B_n)$$

**2. 掩码生成（核心）**
$$M(\Gamma_{ij}, k) = \{g_{ij} \mid |g_{ij}| \geq T_k\}$$

**3. 参数更新**
$$\theta_{ij}^{(t+1)} = \theta_{ij}^{(t)} - \eta \cdot M(\Gamma_{ij}, k)$$

### 理论基础
$$s_{ij} = |\nabla_{\theta_{ij}} L(\Theta; D)|$$
梯度绝对值 = 参数重要性

## 核心代码讲解

### 1. GMTTrainer 类结构

```python
class GMTTrainer:
    def __init__(self, model_name, device, k_percent, accumulation_steps, learning_rate):
        # 初始化参数验证和组件加载
    
    def _validate_parameters(self):
        # 参数范围检查
    
    def _initialize_components(self):
        # 加载tokenizer、model和optimizer
    
    def train(self, texts, num_epochs):
        # 主训练循环
```

### 2. 梯度累积实现

```python
# 累积N个batch的梯度
for param in self.model.parameters():
    if param.grad is not None:
        if param not in accumulated_grads:
            accumulated_grads[param] = torch.zeros_like(param.grad)
        accumulated_grads[param] += param.grad
        param.grad = None

# 取平均
for param in self.model.parameters():
    if param in accumulated_grads:
        param.grad = accumulated_grads[param] / accumulation_steps
```

### 3. 阈值计算（核心算法）

```python
def _compute_threshold(self, accumulated_grads):
    # 收集所有梯度绝对值
    all_grad_values = []
    for param, grad in accumulated_grads.items():
        if grad is not None:
            grad_abs = grad.abs()
            all_grad_values.append(grad_abs.flatten())
    
    # 计算前k%的阈值
    all_grads_flat = torch.cat(all_grad_values)
    k = int(len(all_grads_flat) * self.k_percent / 100)
    threshold = torch.kthvalue(all_grads_flat, len(all_grads_flat) - k + 1).values
    return threshold
```

### 4. 掩码应用

```python
def _apply_gmt_mask(self, accumulated_grads, threshold):
    for param in self.model.parameters():
        if param in accumulated_grads:
            grad_abs = accumulated_grads[param].abs()
            mask = grad_abs >= threshold
            param.gmt_mask = mask
            param.grad = param.grad * mask  # 只保留重要参数的梯度
```

## 测评指标详解

### 1. 梯度能量保留率 (Gradient Energy Retention)

**公式：**
$$\text{Energy Retention} = \frac{\sum_i (g_i \cdot m_i)^2}{\sum_i g_i^2}$$

**代码实现：**
```python
def compute_gradient_energy_retention(self, accumulated_grads):
    total_energy_before = 0.0
    total_energy_after = 0.0
    
    for param, grad in accumulated_grads.items():
        if grad is not None:
            grad_abs = grad.abs()
            total_energy_before += torch.sum(grad_abs ** 2).item()
            
            mask = grad_abs >= threshold
            total_energy_after += torch.sum((grad * mask) ** 2).item()
    
    return total_energy_after / total_energy_before
```

**含义：**
- 衡量被保留的梯度能量占总梯度能量的比例
- 值越接近1，说明大部分能量被保留
- 值越接近0.5（当k=50%），说明约一半能量被保留
- **用途**：验证GMT是否成功筛选出重要参数

---

### 2. Mask稳定性 (Mask Stability)

**公式：**
$$\text{Stability} = \frac{\text{Intersection}(mask_t, mask_{t-1})}{\text{Union}(mask_t, mask_{t-1})}$$

**代码实现：**
```python
def compute_mask_stability(self):
    total_similarity = 0.0
    total_elements = 0
    
    for param in self.model.parameters():
        if hasattr(param, 'gmt_mask') and hasattr(param, 'prev_gmt_mask'):
            curr_mask = param.gmt_mask.flatten()
            prev_mask = param.prev_gmt_mask.flatten()
            
            intersection = torch.sum(curr_mask & prev_mask).item()
            union = torch.sum(curr_mask | prev_mask).item()
            
            if union > 0:
                total_similarity += (intersection / union) * curr_mask.numel()
                total_elements += curr_mask.numel()
    
    return total_similarity / total_elements
```

**含义：**
- 使用Jaccard相似度计算相邻两次mask的重合程度
- 值在0~1之间，越接近1表示稳定性越高
- **用途**：评估GMT选择参数的一致性

---

### 3. 层级更新不平衡 (Layer-wise Update Imbalance)

**指标公式：**
$$\text{Variance} = \frac{1}{n} \sum_{i=1}^{n} (update_i - \mu)^2$$
$$\text{Max/Min Ratio} = \frac{\text{max}(update_i)}{\text{min}(update_i)}$$

**代码实现：**
```python
def compute_layer_update_imbalance(self):
    layer_updates = {}
    total_update = 0.0
    
    for name, param in self.model.named_parameters():
        if param.grad is not None:
            layer_name = name.split('.')[0]
            update_norm = float(torch.norm(param.grad))
            layer_updates[layer_name] += update_norm
            total_update += update_norm
    
    for layer in layer_updates:
        layer_updates[layer] /= total_update
    
    update_values = np.array(list(layer_updates.values()))
    variance = float(np.var(update_values))
    max_ratio = float(np.max(update_values)) / (float(np.min(update_values)) + 1e-10)
    
    return layer_updates, variance, max_ratio
```

**含义：**
- **更新方差**：衡量各层更新量的离散程度
- **最大/最小更新比**：衡量更新最集中和最稀疏层的差距
- **用途**：分析不同层参数更新量的分布情况

---

## 算法流程图

```
┌─────────────────────────────────────────────────────────────┐
│ 输入: 模型参数 Θ, 训练数据 D                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 梯度累积                                           │
│ Γ = 0                                                      │
│ for n = 1 to N:                                            │
│     Γ += ∇Θ L(Θ, B_n)                                      │
│ Γ = Γ / N                                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 计算阈值                                           │
│ T_k = 所有梯度绝对值的第k百分位数                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 掩码筛选                                           │
│ for 每个参数 θ_ij:                                         │
│     if |Γ_ij| >= T_k:                                      │
│         θ_ij = θ_ij - η * Γ_ij  (更新)                      │
│     else:                                                  │
│         不更新                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 输出: 更新后的参数 Θ'                                       │
└─────────────────────────────────────────────────────────────┘
```

## 代码结构

```
├── train_gmt.py          # 主训练脚本（GMTTrainer类）
├── README.md             # 项目说明文档
└── test.py               # 测试脚本
```

## 使用方法

### 安装依赖
```bash
pip install torch transformers numpy
```

### 运行训练
```bash
python train_gmt.py
```

### 自定义配置
```python
from train_gmt import GMTTrainer

trainer = GMTTrainer(
    model_name="distilgpt2",
    device="cpu",           # "cpu" or "cuda"
    k_percent=50,          # 保留前k%重要参数
    accumulation_steps=4,  # 累积N个batch后更新
    learning_rate=5e-5,    # 学习率
)

trainer.train(texts, num_epochs=5)
```

## 输出示例

### 训练配置信息
```
===== Training Configuration =====
Model: distilgpt2
Device: cpu
Number of samples: 80
Number of epochs: 5
GMT k-percent: 50%
Gradient accumulation steps: 4
Learning rate: 5e-05
====================================
```

### Batch级别日志
```
--- Epoch 1/5 ---
  Batch 1/20: Loss=3.2156, EnergyRetention=0.7852, MaskStability=0.6234
  Batch 2/20: Loss=3.1823, EnergyRetention=0.7910, MaskStability=0.6512
  Batch 3/20: Loss=3.1456, EnergyRetention=0.8023, MaskStability=0.6875
  ...
```

### Epoch结果
```
=== Epoch 1 ===
Loss = 2.8567
【指标1】梯度能量保留率 (Gradient Energy Retention): 0.7852
【指标2】Mask稳定性 (Mask Stability): 0.6234
【指标3】层级更新不平衡 (Layer-wise Update Imbalance):
        - 更新方差: 0.012345
        - 最大/最小更新比: 3.25
        - 各层更新分布:
          decoder: 0.4521
          lm_head: 0.3210
          transformer: 0.2269
```

### 训练总结
```
===== Training Summary =====
Metrics evolution across epochs:

[Loss]
  Epoch 1: 2.8567
  Epoch 2: 2.6234
  Epoch 3: 2.4512
  Epoch 4: 2.3156
  Epoch 5: 2.2034

[Gradient Energy Retention]
  Epoch 1: 0.7852
  Epoch 2: 0.8012
  Epoch 3: 0.8156
  Epoch 4: 0.8234
  Epoch 5: 0.8312

[Mask Stability]
  Epoch 1: 0.6234
  Epoch 2: 0.6875
  Epoch 3: 0.7234
  Epoch 4: 0.7567
  Epoch 5: 0.7890

===== Training Completed =====
```

## 📄 Reference

- Li, H., Zhang, X., Liu, X., Gong, Y., Wang, Y., Chen, Q., & Cheng, P. (2025).  
  **Enhancing Large Language Model Performance with Gradient-Based Parameter Selection**.  
  *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2025).*  
  🔗 `https://arxiv.org/abs/2406.15330`

## License
MIT License