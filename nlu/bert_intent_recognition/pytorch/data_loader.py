# -*- coding: utf-8 -*-
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_csv(path):
    df = pd.read_csv(path)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, maxlen=60):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True,
                             max_length=self.maxlen, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def make_collate(tokenizer):
    def collate(batch):
        ids = [b["input_ids"] for b in batch]
        ams = [b["attention_mask"] for b in batch]
        labels = torch.stack([b["label"] for b in batch])
        maxlen = max(x.size(0) for x in ids)
        pad_id = tokenizer.pad_token_id or 0
        import torch.nn.functional as F
        ids = torch.stack([F.pad(x, (0, maxlen - x.size(0)), value=pad_id) for x in ids])
        ams = torch.stack([F.pad(x, (0, maxlen - x.size(0)), value=0) for x in ams])
        return ids, ams, labels
    return collate
