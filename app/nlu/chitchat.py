# -*- coding: utf-8 -*-
import os
import pickle

_KEYWORDS = {
    "greet": ["你好", "您好", "hi", "hello", "哈喽", "在吗"],
    "goodbye": ["再见", "拜拜", "bye", "结束"],
    "isbot": ["你是谁", "你是机器人", "你叫什么"],
    "deny": ["不是", "不对", "错了"],
}


class Chitchat:
    def __init__(self, model_dir=None):
        self.vec = self.clf = self.id2label = None
        if model_dir and os.path.exists(os.path.join(model_dir, "vec.pkl")):
            try:
                with open(os.path.join(model_dir, "vec.pkl"), "rb") as f:
                    self.vec = pickle.load(f)
                with open(os.path.join(model_dir, "LR.pkl"), "rb") as f:
                    self.clf = pickle.load(f)
                with open(os.path.join(model_dir, "id2label.pkl"), "rb") as f:
                    self.id2label = pickle.load(f)
            except Exception:
                self.vec = self.clf = self.id2label = None

    @property
    def available(self) -> bool:
        return self.clf is not None

    def classify(self, text):
        if self.available:
            x = self.vec.transform([" ".join(list(text.lower()))])
            import numpy as np
            label = self.id2label.get(int(np.argmax(self.clf.predict_proba(x), axis=1)[0]))
            return label if label in _KEYWORDS else None
        for label, kws in _KEYWORDS.items():
            if any(k in text.lower() for k in kws):
                return label
        return None
