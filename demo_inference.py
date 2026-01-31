#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速演示 - 使用已训练模型进行推理
"""

import os
import sys
import torch

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*70)
print("数据库字段属性实体识别 - 推理演示")
print("="*70)

try:
    from transformers import BertTokenizer
    from model import BertForNER
    from utils import extract_entities, format_output
    from config import Config

    # 检查模型是否存在（使用绝对路径）
    model_path = os.path.join(SCRIPT_DIR, "models", "best_model")
    if not os.path.exists(model_path):
        print(f"\n错误: 模型文件不存在: {model_path}")
        print("请先运行: python train.py")
        sys.exit(1)

    print(f"\n加载模型: {model_path}")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForNER.from_pretrained(model_path)
    model.to(Config.DEVICE)
    model.eval()
    print(f"✓ 模型加载成功！设备: {Config.DEVICE}")

    # 示例查询
    test_queries = [
        "昨天张山购买了多少商品",
        "本月北京的销售额是多少",
        "2023年第三季度上海地区用户增长率",
        "上周五李四购买了哪些产品",
        "本月杭州地区的订单数量和销售额"
    ]

    print("\n" + "="*70)
    print("开始推理...")
    print("="*70)

    for idx, query in enumerate(test_queries, 1):
        print(f"\n【示例 {idx}】")
        print("-"*70)

        # 提取实体
        result, entities, tokens = extract_entities(query, model, tokenizer)

        # 格式化输出
        format_output(query, result)

    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)

    print("\n💡 提示:")
    print("  - 交互式推理: python inference.py")
    print("  - 批量推理: python inference.py queries.txt results.json")
    print("  - 测试模型: python test.py")
    print("  - 重新训练: python train.py")

except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
