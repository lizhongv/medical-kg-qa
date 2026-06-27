# -*- coding: utf-8 -*-
"""
意图识别训练(RBT3 + TextCNN,PyTorch,13 类)——单文件自包含。

依赖: pip install torch transformers pandas
基座: hfl/rbt3。优先读本地 ./base_model/(若放了模型),否则联网从 Hub 自动下载。
数据: ../data/{train,test}.csv(仓库已自带,字段 text,label_class,label)
训练: python train.py                 # 全量
      python train.py --limit 100     # 冒烟
产物: ./checkpoint/best_model.pt, label2id.json
"""
import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CKPT = os.path.join(HERE, "checkpoint")


def resolve_base():
    """有本地 base_model/ 用本地,否则用 Hub 名 hfl/rbt3(联网自动下载)。"""
    local = os.path.join(HERE, "base_model")
    if os.path.exists(os.path.join(local, "config.json")):
        return local
    return "hfl/rbt3"


# ----------------------------- 数据 -----------------------------
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, maxlen=60):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.maxlen = tokenizer, maxlen

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True,
                             max_length=self.maxlen, return_tensors="pt")
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}


def make_collate(tokenizer):
    pad_id = tokenizer.pad_token_id or 0

    def collate(batch):
        ids = [b["input_ids"] for b in batch]
        ams = [b["attention_mask"] for b in batch]
        labels = torch.stack([b["label"] for b in batch])
        maxlen = max(x.size(0) for x in ids)
        ids = torch.stack([F.pad(x, (0, maxlen - x.size(0)), value=pad_id) for x in ids])
        ams = torch.stack([F.pad(x, (0, maxlen - x.size(0)), value=0) for x in ams])
        return ids, ams, labels
    return collate


# ----------------------------- 模型 -----------------------------
class IntentModel(nn.Module):
    """RBT3 编码 → CLS + TextCNN(核3/4/5) → concat → Dense512 → 分类。"""
    def __init__(self, base, num_labels=13, channels=256, kernels=(3, 4, 5)):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base)
        hidden = self.bert.config.hidden_size
        self.convs = nn.ModuleList([nn.Conv1d(hidden, channels, k, padding=k // 2) for k in kernels])
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(hidden + channels * len(kernels), 512)
        self.out = nn.Linear(512, num_labels)

    def forward(self, input_ids, attention_mask):
        seq = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = seq[:, 0]                       # [B, H]
        x = seq[:, 1:-1].transpose(1, 2)      # [B, H, L-2]
        feats = [torch.max(torch.relu(conv(x)), dim=2).values for conv in self.convs]
        cnn = self.dropout(torch.cat(feats, dim=1))
        h = torch.relu(self.dense(torch.cat([cls, cnn], dim=1)))
        return self.out(h)


# ----------------------------- 训练 -----------------------------
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for ids, ams, labels in loader:
            ids, ams, labels = ids.to(device), ams.to(device), labels.to(device)
            pred = model(ids, ams).argmax(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = resolve_base()
    print("device:", device, "| base:", base)
    tokenizer = AutoTokenizer.from_pretrained(base)

    train_df = pd.read_csv(os.path.join(DATA, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA, "test.csv"))
    if args.limit:
        train_df = train_df.head(args.limit)
        test_df = test_df.head(max(args.limit // 5, 1))

    os.makedirs(CKPT, exist_ok=True)
    id2name = {int(r.label): str(r.label_class) for r in train_df.itertuples()}
    with open(os.path.join(CKPT, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(id2name, f, ensure_ascii=False)

    collate = make_collate(tokenizer)
    tr = DataLoader(IntentDataset(train_df["text"].astype(str).tolist(),
                                  train_df["label"].astype(int).tolist(), tokenizer),
                    batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    te = DataLoader(IntentDataset(test_df["text"].astype(str).tolist(),
                                  test_df["label"].astype(int).tolist(), tokenizer),
                    batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = IntentModel(base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for ids, ams, labels in tr:
            ids, ams, labels = ids.to(device), ams.to(device), labels.to(device)
            opt.zero_grad()
            loss = crit(model(ids, ams), labels)
            loss.backward()
            opt.step()
            tot += loss.item()
        acc = evaluate(model, te, device)
        print(f"epoch {ep+1} loss {tot/len(tr):.4f} test_acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(CKPT, "best_model.pt"))
            print("  saved best_model.pt")


if __name__ == "__main__":
    main()
