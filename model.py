import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from config import Config

class BertForNER(BertPreTrainedModel):
    """
    基于BERT的命名实体识别模型
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = Config.NUM_LABELS
        
        # 加载预训练的BERT模型
        self.bert = BertModel(config)
        
        # Dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 分类器层
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        # 初始化权重
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # 计算每个token的loss
            loss = loss_fct(
                logits.view(-1, self.num_labels), 
                labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }
