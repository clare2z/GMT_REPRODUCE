#!/usr/bin/env python3

with open('code_generation_DGMM.py', 'r') as f:
    content = f.read()

# 使用正则表达式找到第二个定义并删除
import re

# 找到第一个定义的结束位置
first_def = re.search(r'\n    def _analyze_gradient_stability.*?(?=\n    def |\Z)', content, re.DOTALL)
if first_def:
    # 找到第二个定义的位置
    second_def_start = content.find('def _analyze_gradient_stability', first_def.end())
    if second_def_start != -1:
        # 找到第二个定义的结束位置（下一个def或文件结束）
        next_def = re.search(r'\n    def ', content[second_def_start+1:])
        if next_def:
            second_def_end = second_def_start + 1 + next_def.start()
        else:
            second_def_end = len(content)
        
        # 删除第二个定义
        new_content = content[:second_def_start] + content[second_def_end:]
        
        with open('code_generation_DGMM.py', 'w') as f:
            f.write(new_content)
        
        print("✅ 删除了重复定义！")
    else:
        print("❌ 没有找到第二个定义")
else:
    print("❌ 没有找到第一个定义")
