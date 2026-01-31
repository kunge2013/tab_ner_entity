import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertConfig
from model import BertForNER
from data_processor import load_data_splits
from config import Config
# from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
# 由于依赖版本冲突，使用自定义指标计算
def compute_metrics(true_labels, predictions):
    """简单的指标计算"""
    total = sum(len(labels) for labels in true_labels)
    correct = sum(sum(1 for t, p in zip(ts, ps) if t == p)
                   for ts, ps in zip(true_labels, predictions))
    accuracy = correct / total if total > 0 else 0.0
    
    # 计算每种实体类型的指标
    entity_types = ['FIELD', 'TIME', 'METRIC', 'DIMENSION']
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    count = 0
    
    for entity_type in entity_types:
        tp = fp = fn = 0
        for ts, ps in zip(true_labels, predictions):
            # 统计该类型的预测和真实标签
            pred_entities = [1 if entity_type in p else 0 for p in ps]
            true_entities = [1 if entity_type in t else 0 for t in ts]
            
            tp += sum(1 for p, t in zip(pred_entities, true_entities) if p == 1 and t == 1)
            fp += sum(1 for p, t in zip(pred_entities, true_entities) if p == 1 and t == 0)
            fn += sum(1 for p, t in zip(pred_entities, true_entities) if p == 0 and t == 1)
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        count += 1
    
    precision_avg = precision_sum / count if count > 0 else 0.0
    recall_avg = recall_sum / count if count > 0 else 0.0
    f1_avg = f1_sum / count if count > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision_avg,
        'recall': recall_avg,
        'f1': f1_avg
    }

def accuracy_score(true_labels, predictions):
    return compute_metrics(true_labels, predictions)['accuracy']

def precision_score(true_labels, predictions):
    return compute_metrics(true_labels, predictions)['precision']

def recall_score(true_labels, predictions):
    return compute_metrics(true_labels, predictions)['recall']

def f1_score(true_labels, predictions):
    return compute_metrics(true_labels, predictions)['f1']
import random


def set_seed(seed):
    """
    设置随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # 将数据移到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        loss = outputs['loss']
        total_loss += loss.item()
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, tokenizer):
    """
    评估模型
    """
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=-1)
            
            # 收集预测和真实标签
            for i in range(len(labels)):
                pred_tags = []
                true_tag_list = []
                for j in range(len(labels[i])):
                    if labels[i][j] != -100:
                        pred_tags.append(Config.ID2LABEL[preds[i][j].item()])
                        true_tag_list.append(Config.ID2LABEL[labels[i][j].item()])
                
                predictions.append(pred_tags)
                true_labels.append(true_tag_list)
    
    # 计算指标
    acc = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_model(model, tokenizer, output_dir):
    """
    保存模型
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


def main():
    # 设置随机种子
    set_seed(Config.SEED)
    
    # 创建输出目录
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("Loading data...")
    train_dataloader, dev_dataloader, test_dataloader, tokenizer = load_data_splits()
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Dev batches: {len(dev_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")
    
    # 初始化模型
    print("Initializing model...")
    bert_config = BertConfig.from_pretrained(Config.MODEL_NAME)
    model = BertForNER.from_pretrained(
        Config.MODEL_NAME,
        config=bert_config
    )
    model.to(Config.DEVICE)
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    total_steps = len(train_dataloader) * Config.NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=Config.WARMUP_STEPS
    )
    
    # 训练循环
    best_f1 = 0.0
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print(f"{'='*50}")
        
        # 训练
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, Config.DEVICE)
        print(f"Average training loss: {train_loss:.4f}")
        
        # 验证
        print("Evaluating on dev set...")
        dev_metrics = evaluate(model, dev_dataloader, Config.DEVICE, tokenizer)
        print(f"Dev - Accuracy: {dev_metrics['accuracy']:.4f}")
        print(f"Dev - Precision: {dev_metrics['precision']:.4f}")
        print(f"Dev - Recall: {dev_metrics['recall']:.4f}")
        print(f"Dev - F1: {dev_metrics['f1']:.4f}")
        
        # 保存最佳模型
        if dev_metrics['f1'] > best_f1:
            best_f1 = dev_metrics['f1']
            save_model(model, tokenizer, os.path.join(Config.OUTPUT_DIR, 'best_model'))
            print(f"New best model saved with F1: {best_f1:.4f}")
    
    # 最终测试
    print(f"\n{'='*50}")
    print("Final evaluation on test set")
    print(f"{'='*50}")
    test_metrics = evaluate(model, test_dataloader, Config.DEVICE, tokenizer)
    print(f"Test - Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test - Precision: {test_metrics['precision']:.4f}")
    print(f"Test - Recall: {test_metrics['recall']:.4f}")
    print(f"Test - F1: {test_metrics['f1']:.4f}")
    
    # 保存最终模型
    save_model(model, tokenizer, os.path.join(Config.OUTPUT_DIR, 'final_model'))
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
