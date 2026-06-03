#!/bin/bash

python3 << 'PYTHON_SCRIPT'
with open('code_generation_DGMM.py', 'r') as f:
    content = f.read()

old_func = '''    def _analyze_gradient_stability(self, layer_name: str, current_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_name not in self.grad_history:
            self.grad_history[layer_name] = []
        
        grad_std = torch.tensor(0.0, device='cpu')
        grad_diff = torch.tensor(0.0, device='cpu')
        momentum = torch.tensor(0.0, device='cpu')
        
        current_grad_cpu = current_grad.detach().clone().cpu()
        
        if len(self.grad_history[layer_name]) > 0:
            grad_history = torch.stack(self.grad_history[layer_name])
            grad_std = grad_history.std(dim=0).mean()
            
            if len(self.grad_history[layer_name]) >= 2:
                recent_grads = grad_history[-2:]
                grad_diff = torch.abs(recent_grads[1] - recent_grads[0]).mean()
                
                if len(self.grad_history[layer_name]) >= 3:
                    prev_diff = torch.abs(recent_grads[0] - grad_history[-3]).mean()
                    momentum = grad_diff / (prev_diff + 1e-8)
        
        self.grad_history[layer_name].append(current_grad_cpu)
        if len(self.grad_history[layer_name]) > self.grad_history_window:
            self.grad_history[layer_name].pop(0)
        
        return grad_std.to(self.device), grad_diff.to(self.device), momentum.to(self.device)'''

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
        )'''

content = content.replace(old_func, new_func)

with open('code_generation_DGMM.py', 'w') as f:
    f.write(content)

print("✅ Fixed!")
PYTHON_SCRIPT

# 验证
grep -n "torch.stack" code_generation_DGMM.py || echo "✅ torch.stack removed!"

# 运行
source ~/pytorch_env/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
python3 code_generation_DGMM.py
