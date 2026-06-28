# -*- coding: utf-8 -*-
import os
import json


class IntentModel:
    def __init__(self, ckpt_dir=None, base=None):
        self.model = self.tokenizer = self.id2name = None
        self._torch = None
        self._ok = False
        weights = os.path.join(ckpt_dir, "best_model.pt") if ckpt_dir else None
        if weights and os.path.exists(weights):
            try:
                import torch
                from transformers import AutoTokenizer
                # 直接内联最小推理:加载 state_dict 到等价结构
                self._torch = torch
                with open(os.path.join(ckpt_dir, "label2id.json"), encoding="utf-8") as f:
                    self.id2name = json.load(f)
                # 基座必须与训练时一致(层数/结构),否则 load_state_dict 维度不匹配。
                # 优先用显式 base,否则读训练侧写的 meta.json,再否则回退 hfl/rbt3。
                resolved_base = base or _read_meta_base(ckpt_dir)
                self.tokenizer = AutoTokenizer.from_pretrained(resolved_base)
                self.model = _load_intent_model(resolved_base, weights, len(self.id2name), torch)
                self.model.eval()
                self._ok = True
            except Exception:
                self._ok = False

    @property
    def available(self) -> bool:
        return self._ok

    def predict(self, text):
        if not self._ok:
            return {"name": None, "confidence": 0.0}
        torch = self._torch
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=60)
        with torch.no_grad():
            logits = self.model(enc["input_ids"], enc["attention_mask"])
            prob = torch.softmax(logits, 1)[0]
            idx = int(prob.argmax())
        return {"name": self.id2name[str(idx)], "confidence": float(prob[idx])}


def _read_meta_base(ckpt_dir):
    """从训练侧写的 meta.json 读取基座名;缺失则回退 hfl/rbt3。"""
    meta = os.path.join(ckpt_dir, "meta.json")
    if os.path.exists(meta):
        try:
            with open(meta, encoding="utf-8") as f:
                return json.load(f).get("base") or "hfl/rbt3"
        except Exception:
            pass
    return "hfl/rbt3"


def _load_intent_model(base, weights, num_labels, torch):
    import torch.nn as nn
    from transformers import AutoModel

    class IntentNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(base)
            h = self.bert.config.hidden_size
            self.convs = nn.ModuleList([nn.Conv1d(h, 256, k, padding=k // 2) for k in (3, 4, 5)])
            self.dropout = nn.Dropout(0.2)
            self.dense = nn.Linear(h + 256 * 3, 512)
            self.out = nn.Linear(512, num_labels)

        def forward(self, ids, am):
            seq = self.bert(input_ids=ids, attention_mask=am).last_hidden_state
            cls = seq[:, 0]
            x = seq[:, 1:-1].transpose(1, 2)
            feats = [torch.max(torch.relu(c(x)), dim=2).values for c in self.convs]
            cnn = self.dropout(torch.cat(feats, dim=1))
            return self.out(torch.relu(self.dense(torch.cat([cls, cnn], dim=1))))

    net = IntentNet()
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return net
