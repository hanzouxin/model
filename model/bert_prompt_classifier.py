import torch
import torch.nn as nn
from transformers import BertModel


class BertPromptClassifier(nn.Module):
    def __init__(self, num_classes, bert_path="/home/yuck/bert-base-uncased", dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            bert_path,
            local_files_only=True
        )
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = outputs.last_hidden_state[:, 0, :]
        cls_feat = self.dropout(cls_feat)
        logits = self.classifier(cls_feat)
        return logits
