#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速演示脚本 - 展示如何使用训练好的模型进行推理
"""

import os
import sys


def load_demo_model():
    """
    加载演示模型（如果已训练则加载，否则使用预训练模型）
    """
    print("检查模型...")
    
    model_path = "models/best_model"
    
    if not os.path.exists(model_path):
        print(f"未找到训练好的模型: {model_path}")
        print("将使用预训练BERT模型进行演示（精度较低）")
        model_path = "bert-base-chinese"
    else:
        print(f"加载训练好的模型: {model_path}")
    
    return model_path


def demo_inference():
    """
    演示推理功能
    """
    print("\n" + "="*60)
    print("数据库字段属性实体识别 - 演示")
    print("="*60)
    
    try:
        from transformers import BertTokenizer, BertForTokenClassification
        import torch
        
        # 加载模型
        model_path = load_demo_model()
        print(f"\n从 {model_path} 加载模型...")
        
        tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # 检查是否有训练好的模型
        if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            # 加载训练好的模型
            model = BertForTokenClassification.from_pretrained(model_path)
        else:
            print("使用预训练模型演示（需要先训练才能获得准确结果）")
            # 使用预训练模型创建一个简单的分类器
            from transformers import BertConfig
            config = BertConfig.from_pretrained(model_path)
            config.num_labels = 9  # 9个标签
            model = BertForTokenClassification.from_pretrained(model_path, config=config)
        
        model.eval()
        
        # 示例查询
        test_queries = [
            "昨天张山购买了多少商品",
            "本月北京的销售额是多少",
            "2023年第三季度上海地区用户增长率",
            "上周五李四购买了哪些产品"
        ]
        
        # 标签映射
        id2label = {
            0: 'O',
            1: 'B-FIELD',
            2: 'I-FIELD',
            3: 'B-TIME',
            4: 'I-TIME',
            5: 'B-METRIC',
            6: 'I-METRIC',
            7: 'B-DIMENSION',
            8: 'I-DIMENSION'
        }
        
        print("\n开始推理...")
        print("-"*60)
        
        for idx, query in enumerate(test_queries, 1):
            print(f"\n示例 {idx}: {query}")
            
            # Tokenize
            inputs = tokenizer(query, return_tensors="pt", max_length=128, 
                             padding='max_length', truncation=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            # 解码结果
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            pred_labels = [id2label[p.item()] for p in predictions[0]]
            
            # 提取实体
            entities = {
                'FIELD': [],
                'TIME': [],
                'METRIC': [],
                'DIMENSION': []
            }
            
            current_entity = None
            current_text = ""
            
            for token, label in zip(tokens, pred_labels):
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
                
                # 处理##前缀
                if token.startswith('##'):
                    token = token[2:]
                
                if label.startswith('B-'):
                    if current_entity and current_entity != label[2:]:
                        entities[current_entity].append(current_text)
                    current_entity = label[2:]
                    current_text = token
                elif label.startswith('I-') and current_entity == label[2:]:
                    current_text += token
                else:
                    if current_entity:
                        entities[current_entity].append(current_text)
                    current_entity = None
                    current_text = ""
            
            if current_entity:
                entities[current_entity].append(current_text)
            
            # 显示结果
            print("识别结果:")
            entity_names = {
                'FIELD': '字段名',
                'TIME': '时间',
                'METRIC': '指标',
                'DIMENSION': '维度'
            }
            
            for entity_type, entity_list in entities.items():
                name = entity_names[entity_type]
                if entity_list:
                    print(f"  {name}: {', '.join(entity_list)}")
                else:
                    print(f"  {name}: 未识别到")
        
        print("\n" + "="*60)
        print("演示完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请先安装依赖: pip install -r requirements.txt")
        return False
    
    return True


if __name__ == "__main__":
    success = demo_inference()
    sys.exit(0 if success else 1)
