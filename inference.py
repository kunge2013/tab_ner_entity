import os
import torch
from utils import load_model_and_tokenizer, extract_entities, format_output, parse_date
from config import Config

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    """
    交互式推理脚本
    """
    print("="*60)
    print("数据库字段属性实体识别 - 推理系统")
    print("="*60)
    print("\n功能说明:")
    print("  - FIELD: 字段名 (如: 商品、价格)")
    print("  - TIME: 时间 (如: 昨天、2023-01-01)")
    print("  - METRIC: 指标 (如: 购买数量、销售额)")
    print("  - DIMENSION: 维度 (如: 张山、北京)")
    print("="*60)
    
    # 加载模型
    model_path = os.path.join(Config.OUTPUT_DIR, 'best_model')
    
    if not os.path.exists(model_path):
        print(f"\n错误: 模型文件不存在: {model_path}")
        print("请先运行训练脚本 train.py 训练模型")
        return
    
    print(f"\n加载模型: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
    print("模型加载成功!\n")
    
    # 示例查询
    examples = [
        "昨天张山购买了多少商品",
        "本月北京的销售额是多少",
        "2023年第三季度上海地区用户增长率",
        "上周五李四购买了哪些产品",
        "本月杭州地区的订单数量和销售额"
    ]
    
    print("示例查询:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")
    
    print("\n" + "="*60)
    print("输入您的问题（输入 'quit' 或 'exit' 退出）:")
    print("="*60)
    
    while True:
        try:
            # 获取用户输入
            text = input("\n请输入问题: ").strip()
            
            if not text:
                continue
            
            if text.lower() in ['quit', 'exit', '退出', 'q']:
                print("再见!")
                break
            
            # 提取实体
            result, entities, tokens = extract_entities(text, model, tokenizer)
            
            # 格式化输出
            format_output(text, result)
            
            # 进一步处理时间
            if result.get('TIME'):
                print("\n时间标准化:")
                for time_expr in result['TIME']:
                    parsed = parse_date(text)
                    print(f"  {time_expr} -> {parsed}")
            
        except KeyboardInterrupt:
            print("\n\n程序被中断")
            break
        except Exception as e:
            print(f"\n错误: {str(e)}")
            import traceback
            traceback.print_exc()


def batch_inference(input_file, output_file):
    """
    批量推理
    """
    print("="*60)
    print("批量推理模式")
    print("="*60)
    
    # 加载模型
    model_path = os.path.join(Config.OUTPUT_DIR, 'best_model')
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    print(f"加载模型: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
    print("模型加载成功!")
    
    # 读取输入文件
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    print(f"\n处理 {len(queries)} 个查询...")
    
    results = []
    for idx, query in enumerate(tqdm(queries, desc="Processing")):
        result, entities, tokens = extract_entities(query, model, tokenizer)
        
        # 处理时间标准化
        time_normalized = []
        for time_expr in result.get('TIME', []):
            parsed = parse_date(query)
            time_normalized.append({
                'original': time_expr,
                'normalized': parsed
            })
        
        results.append({
            'query': query,
            'entities': result,
            'time_normalized': time_normalized,
            'detailed_entities': entities
        })
    
    # 保存结果
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 打印前几个结果
    print("\n前5个查询的结果:")
    for i in range(min(5, results)):
        print(f"\n{i+1}. {results[i]['query']}")
        print(f"   字段名: {', '.join(results[i]['entities']['FIELD']) or '无'}")
        print(f"   时间: {', '.join(results[i]['entities']['TIME']) or '无'}")
        print(f"   指标: {', '.join(results[i]['entities']['METRIC']) or '无'}")
        print(f"   维度: {', '.join(results[i]['entities']['DIMENSION']) or '无'}")


if __name__ == "__main__":
    import sys
    from tqdm import tqdm
    
    if len(sys.argv) == 3:
        # 批量推理模式
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        batch_inference(input_file, output_file)
    else:
        # 交互式模式
        main()
