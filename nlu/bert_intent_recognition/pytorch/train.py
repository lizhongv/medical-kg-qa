# -*- coding: utf-8 -*-
import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import IntentModel, resolve_base
from data_loader import load_csv, IntentDataset, make_collate

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CKPT = os.path.join(HERE, "checkpoint")


def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for ids, ams, labels in loader:
            ids, ams, labels = ids.to(device), ams.to(device), labels.to(device)
            logits = model(ids, ams)
            pred = logits.argmax(1)
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
    print("device:", device)
    base = resolve_base()
    print("base:", base)
    tokenizer = AutoTokenizer.from_pretrained(base)

    tr_texts, tr_labels = load_csv(os.path.join(DATA, "train.csv"))
    te_texts, te_labels = load_csv(os.path.join(DATA, "test.csv"))
    if args.limit:
        tr_texts, tr_labels = tr_texts[:args.limit], tr_labels[:args.limit]
        te_texts, te_labels = te_texts[:max(args.limit // 5, 1)], te_labels[:max(args.limit // 5, 1)]

    os.makedirs(CKPT, exist_ok=True)
    # label2id 来自数据(label 已是 0..12);保存 id->name 供推理
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA, "train.csv"))
    id2name = {int(r.label): str(r.label_class) for r in df.itertuples()}
    json.dump(id2name, open(os.path.join(CKPT, "label2id.json"), "w", encoding="utf-8"), ensure_ascii=False)

    collate = make_collate(tokenizer)
    tr = DataLoader(IntentDataset(tr_texts, tr_labels, tokenizer), batch_size=args.batch_size,
                    shuffle=True, collate_fn=collate)
    te = DataLoader(IntentDataset(te_texts, te_labels, tokenizer), batch_size=args.batch_size,
                    shuffle=False, collate_fn=collate)

    model = IntentModel(base, num_labels=13).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = torch.nn.CrossEntropyLoss()

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
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), os.path.join(CKPT, "best_model.pt"))
            print("  saved best_model.pt")


if __name__ == "__main__":
    main()
