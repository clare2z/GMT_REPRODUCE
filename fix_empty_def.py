#!/usr/bin/env python3

with open('code_generation_DGMM.py', 'r') as f:
    lines = f.readlines()

# 找到空函数定义并删除
for i in range(len(lines)-1):
    if 'def _analyze_gradient_stability' in lines[i] and 'Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]' in lines[i]:
        # 检查下一行是否是空行或下一个函数定义
        if i + 1 < len(lines) and (lines[i+1].strip() == '' or lines[i+1].strip().startswith('def ')):
            # 删除这一行空函数定义
            del lines[i]
            print(f"✅ 删除了空函数定义在第 {i+1} 行")
            break

# 写入文件
with open('code_generation_DGMM.py', 'w') as f:
    f.writelines(lines)

print("✅ 修复完成！")

# 验证语法
import py_compile
try:
    py_compile.compile('code_generation_DGMM.py')
    print("✅ Syntax OK!")
except py_compile.PyCompileError as e:
    print(f"❌ Syntax error: {e}")
