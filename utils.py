import torch
from transformers import BertTokenizer
from model import BertForNER
from config import Config
import re


def load_model_and_tokenizer(model_path):
    """
    加载训练好的模型和tokenizer
    """
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForNER.from_pretrained(model_path)
    model.to(Config.DEVICE)
    model.eval()
    return model, tokenizer


def decode_predictions(input_ids, predictions, tokenizer):
    """
    解码预测结果
    """
    # 将token IDs转换为tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # 构建结果
    entities = []
    current_entity = None
    current_text = ""
    
    for token, pred_id in zip(tokens, predictions):
        label = Config.ID2LABEL[pred_id]
        
        # 跳过特殊token
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            if current_entity:
                entities.append({
                    'type': current_entity,
                    'text': current_text
                })
                current_entity = None
                current_text = ""
            continue
        
        # 处理token（去除##前缀）
        if token.startswith('##'):
            token = token[2:]
        
        # 解析标签
        if label.startswith('B-'):
            # 开始新的实体
            if current_entity:
                entities.append({
                    'type': current_entity,
                    'text': current_text
                })
            current_entity = label[2:]
            current_text = token
        elif label.startswith('I-') and current_entity:
            # 继续当前实体
            if label[2:] == current_entity:
                current_text += token
            else:
                # 实体类型不匹配，结束当前实体
                entities.append({
                    'type': current_entity,
                    'text': current_text
                })
                current_entity = label[2:]
                current_text = token
        else:
            # O标签
            if current_entity:
                entities.append({
                    'type': current_entity,
                    'text': current_text
                })
                current_entity = None
                current_text = ""
    
    # 处理最后一个实体
    if current_entity:
        entities.append({
            'type': current_entity,
            'text': current_text
        })
    
    return entities, tokens


def extract_entities(text, model, tokenizer):
    """
    从文本中提取实体
    
    Args:
        text: 输入文本
        model: BERT NER模型
        tokenizer: BERT tokenizer
    
    Returns:
        dict: 包含各类实体的字典
    """
    # 对文本进行tokenization
    inputs = tokenizer(
        text,
        max_length=Config.MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(Config.DEVICE)
    attention_mask = inputs['attention_mask'].to(Config.DEVICE)
    token_type_ids = inputs.get('token_type_ids', torch.zeros_like(input_ids)).to(Config.DEVICE)
    
    # 预测
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
    
    logits = outputs['logits']
    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    
    # 解码结果
    entities, tokens = decode_predictions(input_ids[0].cpu().numpy(), predictions, tokenizer)
    
    # 按类型组织实体
    result = {
        'FIELD': [],
        'TIME': [],
        'METRIC': [],
        'DIMENSION': []
    }
    
    for entity in entities:
        entity_type = entity['type']
        if entity_type in result:
            result[entity_type].append(entity['text'])
    
    return result, entities, tokens


def format_output(text, result):
    """
    格式化输出结果
    """
    print("\n" + "="*60)
    print(f"输入文本: {text}")
    print("="*60)
    print("\n识别结果:")
    print("-"*60)
    
    for entity_type, entities in result.items():
        entity_name = {
            'FIELD': '字段名',
            'TIME': '时间',
            'METRIC': '指标',
            'DIMENSION': '维度'
        }.get(entity_type, entity_type)
        
        if entities:
            print(f"\n{entity_name}: {', '.join(entities)}")
        else:
            print(f"\n{entity_name}: 未识别到")
    
    print("\n" + "="*60)


def parse_date(text):
    """
    解析日期文本，返回标准格式
    """
    from datetime import datetime, timedelta
    import jieba
    
    # 简单的相对日期处理
    today = datetime.now()
    
    if '昨天' in text:
        return (today - timedelta(days=1)).strftime('%Y-%m-%d')
    elif '前天' in text:
        return (today - timedelta(days=2)).strftime('%Y-%m-%d')
    elif '今天' in text or '今日' in text:
        return today.strftime('%Y-%m-%d')
    elif '明天' in text or '明日' in text:
        return (today + timedelta(days=1)).strftime('%Y-%m-%d')
    elif '本月' in text:
        return today.strftime('%Y-%m')
    elif '上个月' in text or '上月' in text:
        return (today.replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
    elif '下个月' in text or '下月' in text:
        if today.month == 12:
            return f"{today.year + 1}-01"
        else:
            return today.strftime('%Y-%m')[:5] + f"{today.month + 1:02d}"
    
    # 尝试解析标准日期格式
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',
        r'\d{4}/\d{2}/\d{2}',
        r'\d{4}年\d{2}月\d{2}日',
        r'\d{2}-\d{2}-\d{2}',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                date_str = match.group()
                if '年' in date_str:
                    return datetime.strptime(date_str, '%Y年%m月%d日').strftime('%Y-%m-%d')
                elif '-' in date_str and date_str.count('-') == 2:
                    if len(date_str.split('-')[0]) == 4:
                        return date_str  # 已经是标准格式
                    else:
                        return datetime.strptime(date_str, '%y-%m-%d').strftime('%Y-%m-%d')
                elif '/' in date_str:
                    return datetime.strptime(date_str, '%Y/%m/%d').strftime('%Y-%m-%d')
            except:
                pass
    
    return text  # 无法解析，返回原始文本
