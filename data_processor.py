import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config import Config
import json


class NERDataset(Dataset):
    """
    NER数据集类
    """
    
    def __init__(self, file_path, tokenizer, max_length=Config.MAX_LENGTH):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        self.examples = self._load_data()
    
    def _load_data(self):
        """
        加载数据文件，格式：
        昨    B-TIME
        天    I-TIME
        张    B-DIMENSION
        山    I-DIMENSION
        购    O
        买    O
        了    O
        多    O
        少    O
        商    B-FIELD
        品    I-FIELD
        
        """
        examples = []
        current_words = []
        current_labels = []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    # 空行表示一个句子结束
                    if current_words and current_labels:
                        examples.append({
                            'words': current_words,
                            'labels': current_labels
                        })
                        current_words = []
                        current_labels = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0]
                        label = parts[-1]  # 标签在最后一列
                        current_words.append(word)
                        current_labels.append(label)
        
            # 处理最后一个句子
            if current_words and current_labels:
                examples.append({
                    'words': current_words,
                    'labels': current_labels
                })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        words = example['words']
        labels = example['labels']
        
        # 将词转换为token - 使用encode方式
        tokens = []
        aligned_labels = []
        token_labels = []
        
        # 添加 [CLS] token
        tokens.append('[CLS]')
        token_labels.append(-100)
        
        # 处理每个词
        for word, label in zip(words, labels):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                # 如果词被拆分成多个子词，第一个用B标签，其余用I标签
                word_tokens = [word]  # 降级处理
            
            for i, token in enumerate(word_tokens):
                tokens.append(token)
                if i == 0:
                    # 第一个token使用标签
                    token_labels.append(Config.LABEL2ID[label])
                else:
                    # 后续token使用相同的标签（BIO体系中需要处理I-标签）
                    if label.startswith('B-'):
                        # 转换为I标签
                        new_label = 'I-' + label[2:]
                    else:
                        new_label = label
                    token_labels.append(Config.LABEL2ID.get(new_label, -100))
        
        # 添加 [SEP] token
        tokens.append('[SEP]')
        token_labels.append(-100)
        
        # 转换为token IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # 填充到max_length
        attention_mask = [1] * len(input_ids)
        
        # 添加填充
        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
            attention_mask.extend([0] * pad_length)
            token_labels.extend([-100] * pad_length)
        
        # 截断
        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        token_labels = token_labels[:self.max_length]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.zeros(len(input_ids), dtype=torch.long),
            'labels': torch.tensor(token_labels, dtype=torch.long)
        }
        
        # 移除batch维度
        return {
            'input_ids': tokenized_inputs['input_ids'].squeeze(0),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(0),
            'token_type_ids': tokenized_inputs.get('token_type_ids', torch.zeros_like(tokenized_inputs['input_ids'])).squeeze(0),
            'labels': tokenized_inputs['labels'].squeeze(0)
        }


def create_dataloader(dataset_path, tokenizer, batch_size=Config.BATCH_SIZE, shuffle=True):
    """
    创建数据加载器
    """
    dataset = NERDataset(dataset_path, tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == 'cuda' else False
    )
    return dataloader


def load_data_splits():
    """
    加载训练、验证和测试数据
    """
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_dataloader = create_dataloader(Config.TRAIN_FILE, tokenizer, shuffle=True)
    dev_dataloader = create_dataloader(Config.DEV_FILE, tokenizer, shuffle=False)
    test_dataloader = create_dataloader(Config.TEST_FILE, tokenizer, shuffle=False)
    
    return train_dataloader, dev_dataloader, test_dataloader, tokenizer
