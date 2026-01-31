#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
项目验证脚本 - 检查所有必需的文件和依赖
"""

import os
import sys


def check_file(filepath, required=True):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"✓ {filepath}")
        return True
    else:
        if required:
            print(f"✗ {filepath} (缺失)")
        else:
            print(f"- {filepath} (可选，不存在)")
        return False


def check_import(module_name):
    """检查Python包是否可以导入"""
    try:
        __import__(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError:
        print(f"✗ {module_name} (未安装)")
        return False


def main():
    print("="*60)
    print("数据库字段属性实体识别模型 - 项目验证")
    print("="*60)
    
    # 检查Python版本
    print("\nPython环境:")
    print(f"  Python版本: {sys.version}")
    print(f"  Python路径: {sys.executable}")
    
    # 检查依赖包
    print("\n依赖包:")
    packages = [
        'torch',
        'transformers',
        'sklearn',
        'numpy',
        'pandas',
        'seqeval',
        'tqdm'
    ]
    packages_ok = all([check_import(pkg) for pkg in packages])
    
    # 检查项目文件
    print("\n项目文件:")
    files = [
        'config.py',
        'data_processor.py',
        'model.py',
        'train.py',
        'test.py',
        'inference.py',
        'utils.py',
        'requirements.txt',
        'README.md'
    ]
    files_ok = all([check_file(f) for f in files])
    
    # 检查数据集
    print("\n数据集:")
    dataset_files = [
        'dataset/train.txt',
        'dataset/dev.txt',
        'dataset/test.txt'
    ]
    dataset_ok = all([check_file(f) for f in dataset_files])
    
    # 检查模型目录
    print("\n模型目录:")
    if os.path.exists('models'):
        print("✓ models/")
        if os.path.exists('models/best_model'):
            print("  ✓ models/best_model/ (已训练)")
        else:
            print("  - models/best_model/ (未训练)")
    else:
        print("- models/ (将在训练时创建)")
    
    # 统计数据集
    print("\n数据集统计:")
    if dataset_ok:
        for dataset_file in dataset_files:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                sentences = 0
                for line in lines:
                    if line.strip() == '':
                        sentences += 1
                print(f"  {dataset_file}: {sentences} 个句子, {len(lines)} 行")
    
    # 验证配置
    print("\n配置验证:")
    try:
        from config import Config
        print(f"✓ config.py 加载成功")
        print(f"  模型: {Config.MODEL_NAME}")
        print(f"  标签数: {Config.NUM_LABELS}")
        print(f"  批次大小: {Config.BATCH_SIZE}")
        print(f"  训练轮数: {Config.NUM_EPOCHS}")
        print(f"  设备: {Config.DEVICE}")
        config_ok = True
    except Exception as e:
        print(f"✗ config.py 加载失败: {e}")
        config_ok = False
    
    # 总结
    print("\n" + "="*60)
    print("验证总结:")
    print("="*60)
    
    all_ok = packages_ok and files_ok and dataset_ok and config_ok
    
    if all_ok:
        print("✓ 所有检查通过！项目已准备就绪。")
        print("\n下一步:")
        print("  1. 运行训练: python train.py")
        print("  2. 测试模型: python test.py")
        print("  3. 推理使用: python inference.py")
        print("  4. 或运行快速开始: bash run_demo.sh")
        return 0
    else:
        print("✗ 检查未通过，请解决上述问题。")
        if not packages_ok:
            print("\n安装依赖: pip install -r requirements.txt")
        if not dataset_ok:
            print("\n请检查数据集文件是否存在")
        return 1


if __name__ == "__main__":
    sys.exit(main())
