# -*- coding: utf-8 -*-
"""
医疗 NER 训练(BiLSTM-CRF,PyTorch,字符级)——单文件自包含。

依赖: pip install torch pytorch-crf
数据: cMedQANER(ChineseBLUE),首次运行自动下载到 ./data/
训练: python train.py                 # 全量
      python train.py --limit 200     # 冒烟(只用前200句)
产物: ./checkpoint/best_ner.pt, vocab.json, tag2id.json
"""
import os
import json
import argparse
import urllib.request
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CKPT = os.path.join(HERE, "checkpoint")
BASE_URL = "https://raw.githubusercontent.com/chenxshuo/ChineseBLUE/master/data/cMedQANER"


# ----------------------------- 数据 -----------------------------
def ensure_data():
    """首次运行时下载 cMedQANER 到 ./data/。"""
    os.makedirs(DATA, exist_ok=True)
    for name in ["train.txt", "dev.txt", "test.txt"]:
        dst = os.path.join(DATA, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        print(f"下载 {name} ...")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", dst)


def read_bio(path):
    """读取字符级 BIO 文件,按空行分句,返回 [(chars, tags), ...]。"""
    sentences, chars, tags = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if chars:
                    sentences.append((chars, tags))
                    chars, tags = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            chars.append(parts[0])
            tags.append(parts[-1])
    if chars:
        sentences.append((chars, tags))
    return sentences


def build_vocab(sentences):
    c = Counter()
    for chars, _ in sentences:
        c.update(chars)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for ch, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
        vocab[ch] = len(vocab)
    return vocab


def build_tag2id(sentences):
    tags = set()
    for _, t in sentences:
        tags.update(t)
    tag2id = {"<PAD>": 0}
    for t in sorted(tags):
        tag2id[t] = len(tag2id)
    return tag2id


class NerDataset(Dataset):
    def __init__(self, sentences, vocab, tag2id):
        self.sentences, self.vocab, self.tag2id = sentences, vocab, tag2id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        chars, tags = self.sentences[idx]
        cid = [self.vocab.get(c, 1) for c in chars]
        tid = [self.tag2id[t] for t in tags]
        return (torch.tensor(cid, dtype=torch.long),
                torch.tensor(tid, dtype=torch.long), len(cid))


def collate_fn(batch):
    lengths = [b[2] for b in batch]
    maxlen, n = max(lengths), len(batch)
    chars = torch.zeros(n, maxlen, dtype=torch.long)
    tags = torch.zeros(n, maxlen, dtype=torch.long)
    mask = torch.zeros(n, maxlen, dtype=torch.bool)
    for i, (cid, tid, ln) in enumerate(batch):
        chars[i, :ln], tags[i, :ln], mask[i, :ln] = cid, tid, True
    return chars, tags, mask


# ----------------------------- 模型 -----------------------------
class BiLstmCrf(nn.Module):
    def __init__(self, vocab_size, num_tags, emb_dim=128, hidden=128, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden // 2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def _emissions(self, chars):
        x = self.dropout(self.lstm(self.embedding(chars))[0])
        return self.hidden2tag(x)

    def loss(self, chars, tags, mask):
        return -self.crf(self._emissions(chars), tags, mask=mask, reduction="mean")

    def decode(self, chars, mask):
        return self.crf.decode(self._emissions(chars), mask=mask)


# ----------------------------- 训练 -----------------------------
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for chars, tags, mask in loader:
            chars, tags, mask = chars.to(device), tags.to(device), mask.to(device)
            for i, p in enumerate(model.decode(chars, mask)):
                gold = tags[i][mask[i]].tolist()
                for a, b in zip(p, gold):
                    total += 1
                    correct += int(a == b)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="只用前N句做冒烟测试,0=全量")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    ensure_data()

    train_sents = read_bio(os.path.join(DATA, "train.txt"))
    dev_sents = read_bio(os.path.join(DATA, "dev.txt"))
    if args.limit:
        train_sents = train_sents[:args.limit]
        dev_sents = dev_sents[:max(args.limit // 5, 1)]

    vocab, tag2id = build_vocab(train_sents), build_tag2id(train_sents)
    os.makedirs(CKPT, exist_ok=True)
    with open(os.path.join(CKPT, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(CKPT, "tag2id.json"), "w", encoding="utf-8") as f:
        json.dump(tag2id, f, ensure_ascii=False)

    train_loader = DataLoader(NerDataset(train_sents, vocab, tag2id),
                              batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(NerDataset(dev_sents, vocab, tag2id),
                            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = BiLstmCrf(len(vocab), len(tag2id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for chars, tags, mask in train_loader:
            chars, tags, mask = chars.to(device), tags.to(device), mask.to(device)
            opt.zero_grad()
            loss = model.loss(chars, tags, mask)
            loss.backward()
            opt.step()
            tot += loss.item()
        acc = evaluate(model, dev_loader, device)
        print(f"epoch {ep+1} loss {tot/len(train_loader):.4f} dev_token_acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(CKPT, "best_ner.pt"))
            print("  saved best_ner.pt")


if __name__ == "__main__":
    main()
