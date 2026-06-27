# -*- coding: utf-8 -*-
import os
import json
import torch
from model import BiLstmCrf

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoint")


def load():
    vocab = json.load(open(os.path.join(CKPT, "vocab.json"), encoding="utf-8"))
    tag2id = json.load(open(os.path.join(CKPT, "tag2id.json"), encoding="utf-8"))
    id2tag = {v: k for k, v in tag2id.items()}
    model = BiLstmCrf(len(vocab), len(tag2id))
    model.load_state_dict(torch.load(os.path.join(CKPT, "best_ner.pt"), map_location="cpu"))
    model.eval()
    return model, vocab, id2tag


def predict(text):
    model, vocab, id2tag = load()
    chars = list(text)
    cid = torch.tensor([[vocab.get(c, 1) for c in chars]], dtype=torch.long)
    mask = torch.ones_like(cid, dtype=torch.bool)
    path = model.decode(cid, mask)[0]
    return list(zip(chars, [id2tag[i] for i in path]))


if __name__ == "__main__":
    print(predict("淋球菌性尿道炎的症状"))
