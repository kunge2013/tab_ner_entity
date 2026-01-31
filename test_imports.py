#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速测试脚本 - 验证所有导入和基本功能
"""

import sys

print("="*60)
print("导入测试")
print("="*60)

try:
    print("\n1. 测试基础库导入...")
    import torch
    import numpy as np
    print("   ✓ torch, numpy")

    print("\n2. 测试transformers导入...")
    from transformers import BertTokenizer, BertConfig
    print("   ✓ BertTokenizer, BertConfig")

    print("\n3. 测试自定义模块导入...")
    from config import Config
    print(f"   ✓ Config (模型: {Config.MODEL_NAME})")

    print("\n4. 测试数据处理器...")
    from data_processor import load_data_splits
    train_dataloader, dev_dataloader, test_dataloader, tokenizer = load_data_splits()
    print(f"   ✓ 数据加载成功")
    print(f"     - 训练批次: {len(train_dataloader)}")
    print(f"     - 验证批次: {len(dev_dataloader)}")
    print(f"     - 测试批次: {len(test_dataloader)}")

    print("\n5. 测试模型初始化...")
    bert_config = BertConfig.from_pretrained(Config.MODEL_NAME)
    from model import BertForNER
    model = BertForNER.from_pretrained(Config.MODEL_NAME, config=bert_config)
    print("   ✓ 模型初始化成功")

    print("\n6. 测试一个batch...")
    batch = next(iter(train_dataloader))
    print(f"   ✓ Batch加载成功")
    print(f"     - input_ids shape: {batch['input_ids'].shape}")
    print(f"     - labels shape: {batch['labels'].shape}")

    print("\n7. 测试前向传播...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
    print(f"   ✓ 前向传播成功")
    print(f"     - loss: {outputs['loss'].item():.4f}")

    print("\n" + "="*60)
    print("✓ 所有测试通过！代码可以正常运行")
    print("="*60)

    print("\n现在可以运行完整训练:")
    print("  python train.py")

except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
