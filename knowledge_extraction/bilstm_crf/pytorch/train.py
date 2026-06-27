# -*- coding: utf-8 -*-
import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from data_loader import read_bio, build_vocab, build_tag2id, NerDataset, collate_fn
from model import BiLstmCrf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CKPT = os.path.join(HERE, "checkpoint")


def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for chars, tags, mask, _ in loader:
            chars, tags, mask = chars.to(device), tags.to(device), mask.to(device)
            paths = model.decode(chars, mask)
            for i, p in enumerate(paths):
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

    train_sents = read_bio(os.path.join(DATA, "train.txt"))
    dev_sents = read_bio(os.path.join(DATA, "dev.txt"))
    if args.limit:
        train_sents = train_sents[:args.limit]
        dev_sents = dev_sents[:max(args.limit // 5, 1)]

    vocab = build_vocab(train_sents)
    tag2id = build_tag2id(train_sents)
    os.makedirs(CKPT, exist_ok=True)
    with open(os.path.join(CKPT, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(CKPT, "tag2id.json"), "w", encoding="utf-8") as f:
        json.dump(tag2id, f, ensure_ascii=False)

    train_ds = NerDataset(train_sents, vocab, tag2id)
    dev_ds = NerDataset(dev_sents, vocab, tag2id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = BiLstmCrf(len(vocab), len(tag2id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for chars, tags, mask, _ in train_loader:
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
