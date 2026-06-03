#!/usr/bin/env python3

with open('code_generation_DGMM.py', 'r') as f:
    content = f.read()

# 添加函数定义
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
        )\n\n'''

# 在 _analyze_gradient_direction 后插入函数
insert_pos = content.find('def _analyze_layer_correlation')
if insert_pos != -1:
    content = content[:insert_pos] + new_func + content[insert_pos:]

# 修复函数调用（将4个返回值改为3个）
content = content.replace(
    'stability, grad_diff, momentum, direction_flip = self._analyze_gradient_stability(layer_name, grad)',
    'stability, grad_diff, momentum = self._analyze_gradient_stability(layer_name, grad)'
)

# 写入文件
with open('code_generation_DGMM.py', 'w') as f:
    f.write(content)

print("✅ 修复完成！")

# 验证语法
import py_compile
try:
    py_compile.compile('code_generation_DGMM.py')
    print("✅ Syntax OK!")
except py_compile.PyCompileError as e:
    print(f"❌ Syntax error: {e}")

# 验证函数存在
with open('code_generation_DGMM.py', 'r') as f:
    content = f.read()
if '_analyze_gradient_stability' in content:
    print("✅ 函数已存在！")
