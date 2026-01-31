import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig
from model import BertForNER
from data_processor import NERDataset, create_dataloader
from config import Config
from seqeval.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report
)
import json


def test_model(model_path, test_file):
    """
    测试模型性能
    """
    print(f"Loading model from {model_path}...")
    
    # 加载模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForNER.from_pretrained(model_path)
    model.to(Config.DEVICE)
    model.eval()
    
    # 加载测试数据
    print(f"Loading test data from {test_file}...")
    test_dataloader = create_dataloader(test_file, tokenizer, shuffle=False)
    
    # 收集预测结果
    all_predictions = []
    all_true_labels = []
    test_examples = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            token_type_ids = batch['token_type_ids'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=-1)
            
            # 处理每个样本
            for i in range(len(labels)):
                pred_tags = []
                true_tag_list = []
                
                # 只保留非-100的token
                for j in range(len(labels[i])):
                    if labels[i][j] != -100:
                        pred_tags.append(Config.ID2LABEL[preds[i][j].item()])
                        true_tag_list.append(Config.ID2LABEL[labels[i][j].item()])
                
                all_predictions.append(pred_tags)
                all_true_labels.append(true_tag_list)
    
    # 计算总体指标
    print("\n" + "="*60)
    print("总体评估指标")
    print("="*60)
    
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions)
    recall = recall_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # 详细分类报告
    print("\n" + "="*60)
    print("详细分类报告")
    print("="*60)
    report = classification_report(all_true_labels, all_predictions, digits=4)
    print(report)
    
    # 按实体类型统计
    print("\n" + "="*60)
    print("各实体类型性能")
    print("="*60)
    
    entity_types = ['FIELD', 'TIME', 'METRIC', 'DIMENSION']
    for entity_type in entity_types:
        # 筛选特定类型的标签
        filtered_true = []
        filtered_pred = []
        
        for true_tags, pred_tags in zip(all_true_labels, all_predictions):
            f_true = [tag if entity_type in tag else 'O' for tag in true_tags]
            f_pred = [tag if entity_type in tag else 'O' for tag in pred_tags]
            filtered_true.append(f_true)
            filtered_pred.append(f_pred)
        
        # 计算该类型的指标
        if any(entity_type in tags for tags in all_true_labels):
            f_f1 = f1_score(filtered_true, filtered_pred)
            f_precision = precision_score(filtered_true, filtered_pred)
            f_recall = recall_score(filtered_true, filtered_pred)
            
            entity_name = {
                'FIELD': '字段名',
                'TIME': '时间',
                'METRIC': '指标',
                'DIMENSION': '维度'
            }.get(entity_type, entity_type)
            
            print(f"{entity_name:10s} - Precision: {f_precision:.4f}, Recall: {f_recall:.4f}, F1: {f_f1:.4f}")
    
    # 保存结果到JSON
    results = {
        'overall': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'classification_report': report
    }
    
    results_file = "test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到 {results_file}")
    
    return results


def visualize_predictions(model_path, test_file, num_samples=5):
    """
    可视化部分预测结果
    """
    print(f"\n" + "="*60)
    print("可视化预测结果 (前{num_samples}个样本)")
    print("="*60)
    
    # 加载模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForNER.from_pretrained(model_path)
    model.to(Config.DEVICE)
    model.eval()
    
    # 加载测试数据集
    dataset = NERDataset(test_file, tokenizer)
    
    from utils import extract_entities, format_output
    
    count = 0
    for i in range(min(num_samples, len(dataset))):
        example = dataset.examples[i]
        text = ''.join(example['words'])
        true_labels = example['labels']
        
        # 提取实体
        result, entities, tokens = extract_entities(text, model, tokenizer)
        
        print(f"\n样本 {count + 1}:")
        print(f"文本: {text}")
        
        # 显示真实标签
        print("\n真实标注:")
        word_with_labels = []
        for word, label in zip(example['words'], true_labels):
            word_with_labels.append(f"{word}/{label}")
        print(' '.join(word_with_labels))
        
        # 显示预测结果
        print("\n预测结果:")
        format_output(text, result)
        
        count += 1
        print()


def main():
    """
    主函数
    """
    # 测试最佳模型
    best_model_path = os.path.join(Config.OUTPUT_DIR, 'best_model')
    
    if not os.path.exists(best_model_path):
        print(f"模型文件不存在: {best_model_path}")
        print("请先运行训练脚本 train.py")
        return
    
    # 测试模型
    test_model(best_model_path, Config.TEST_FILE)
    
    # 可视化部分预测结果
    visualize_predictions(best_model_path, Config.TEST_FILE, num_samples=5)


if __name__ == "__main__":
    import os
    main()
