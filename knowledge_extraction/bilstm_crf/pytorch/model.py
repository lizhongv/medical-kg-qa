# -*- coding: utf-8 -*-
import torch.nn as nn
from torchcrf import CRF


class BiLstmCrf(nn.Module):
    def __init__(self, vocab_size, num_tags, emb_dim=128, hidden=128, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, chars, mask):
        x = self.embedding(chars)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return self.hidden2tag(x)

    def loss(self, chars, tags, mask):
        emissions = self(chars, mask)
        return -self.crf(emissions, tags, mask=mask, reduction="mean")

    def decode(self, chars, mask):
        emissions = self(chars, mask)
        return self.crf.decode(emissions, mask=mask)
