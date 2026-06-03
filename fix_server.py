#!/usr/bin/env python3
import re

# 读取文件
with open('code_generation_DGMM.py', 'r') as f:
    lines = f.readlines()

# 找到 _analyze_gradient_stability 函数并替换
output_lines = []
i = 0
while i < len(lines):
    if 'def _analyze_gradient_stability' in lines[i]:
        # 开始替换
        output_lines.append(lines[i])
        i += 1
        
        # 跳过旧函数内容直到遇到下一个 def 或空行
        while i < len(lines):
            if (i > 0 and lines[i].strip().startswith('def ') and not 'self.' in lines[i]) or \
               (i > 0 and i + 1 < len(lines) and lines[i].strip() == '' and lines[i+1].strip().startswith('def ')):
                break
            i += 1
        
        # 插入新的函数实现
        new_func = '''    def _analyze_gradient_stability(self, layer_name: str, current_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_name not in self.grad_history:
            self.grad_history[layer_name] = []
        
        grad_std = 0.0
        grad_diff = 0.0
        momentum = 0.0
        
        current_grad_norm = float(current_grad.norm().item())
        
        if len(self.grad_history[layer_name]) > 0:
            grad_std = float(np.std(self.grad_history[layer_name]))
            
            if len(self.grad_history[layer_name]) >= 2:
                recent_grad_norm = self.grad_history[layer_name][-1]
                prev_grad_norm = self.grad_history[layer_name][-2]
                grad_diff = float(np.abs(recent_grad_norm - prev_grad_norm))
                
                if len(self.grad_history[layer_name]) >= 3:
                    prev_prev_grad_norm = self.grad_history[layer_name][-3]
                    prev_diff = np.abs(prev_grad_norm - prev_prev_grad_norm)
                    momentum = float(grad_diff / (prev_diff + 1e-8))
        
        self.grad_history[layer_name].append(current_grad_norm)
        if len(self.grad_history[layer_name]) > self.grad_history_window:
            self.grad_history[layer_name].pop(0)
        
        return (
            torch.tensor(grad_std, device=self.device),
            torch.tensor(grad_diff, device=self.device),
            torch.tensor(momentum, device=self.device)
        )\n'''
        output_lines.append(new_func)
        continue
    
    output_lines.append(lines[i])
    i += 1

# 写入文件
with open('code_generation_DGMM.py', 'w') as f:
    f.writelines(output_lines)

print("✅ Fixed _analyze_gradient_stability!")

# 验证
with open('code_generation_DGMM.py', 'r') as f:
    content = f.read()

if 'torch.stack(self.grad_history[layer_name])' not in content:
    print("✅ torch.stack removed successfully!")
else:
    print("❌ torch.stack still exists!")
