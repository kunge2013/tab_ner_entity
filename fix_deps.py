#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys

print("修复依赖包版本冲突...")

# 检查当前版本
result = subprocess.run([sys.executable, "-m", "pip", "show", "transformers"], capture_output=True, text=True)
transformers_version = None
for line in result.stdout.split('\n'):
    if line.startswith('Version:'):
        transformers_version = line.split(':')[1].strip()
        break

result = subprocess.run([sys.executable, "-m", "pip", "show", "tokenizers"], capture_output=True, text=True)
tokenizers_version = None
for line in result.stdout.split('\n'):
    if line.startswith('Version:'):
        tokenizers_version = line.split(':')[1].strip()
        break

print(f"当前 transformers 版本: {transformers_version}")
print(f"当前 tokenizers 版本: {tokenizers_version}")

# 修复方案：使用兼容的版本组合
if transformers_version and "4.56" in transformers_version:
    print("\n检测到 transformers 4.56.x，需要升级...")
    print("正在安装兼容的版本...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "transformers", "tokenizers"], check=False)
else:
    print("\n尝试安装兼容版本...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.30.2", "tokenizers>=0.11.1,<0.14.0"], check=False)

print("\n依赖修复完成！")
print("现在可以运行: python train.py")
