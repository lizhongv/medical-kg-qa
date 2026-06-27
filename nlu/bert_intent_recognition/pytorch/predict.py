# -*- coding: utf-8 -*-
import os
import json
import torch
from transformers import AutoTokenizer
from model import IntentModel, resolve_base

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoint")


def main():
    base = resolve_base()
    tokenizer = AutoTokenizer.from_pretrained(base)
    id2name = json.load(open(os.path.join(CKPT, "label2id.json"), encoding="utf-8"))
    model = IntentModel(base, num_labels=13)
    model.load_state_dict(torch.load(os.path.join(CKPT, "best_model.pt"), map_location="cpu"))
    model.eval()
    text = "淋球菌性尿道炎的症状"
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=60)
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        prob = torch.softmax(logits, 1)[0]
        idx = int(prob.argmax())
    print({"name": id2name[str(idx)], "confidence": float(prob[idx])})


if __name__ == "__main__":
    main()
