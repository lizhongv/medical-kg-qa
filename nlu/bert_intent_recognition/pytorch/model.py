# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from transformers import AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_base():
    local = os.path.join(HERE, "base_model")
    if os.path.exists(os.path.join(local, "config.json")):
        return local
    return "hfl/rbt3"


class TextCnnHead(nn.Module):
    def __init__(self, hidden, num_labels, channels=256, kernels=(3, 4, 5), dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden, channels, k, padding=k // 2) for k in kernels
        ])
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden + channels * len(kernels), 512)
        self.act = nn.ReLU()
        self.out = nn.Linear(512, num_labels)

    def forward(self, cls, tokens):
        # tokens: [B, L, H] -> [B, H, L]
        x = tokens.transpose(1, 2)
        feats = []
        for conv in self.convs:
            c = torch.relu(conv(x))            # [B, C, L]
            c = torch.max(c, dim=2).values     # [B, C]
            feats.append(c)
        cnn = self.dropout(torch.cat(feats, dim=1))
        h = self.act(self.dense(torch.cat([cls, cnn], dim=1)))
        return self.out(h)


class IntentModel(nn.Module):
    def __init__(self, base_name_or_path, num_labels=13):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_name_or_path)
        hidden = self.bert.config.hidden_size
        self.head = TextCnnHead(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state          # [B, L, H]
        cls = seq[:, 0]                      # [B, H]
        tokens = seq[:, 1:-1]                # [B, L-2, H]
        return self.head(cls, tokens)
