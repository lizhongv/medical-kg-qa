# -*- coding: utf-8 -*-
"""
闲聊意图分类训练(sklearn:字符级 TF-IDF + LR + GBDT 集成)——单文件自包含。

依赖: pip install scikit-learn numpy
数据: ./data/intent_recog_data.txt(仓库已自带,每行「文本,标签」)
训练: python train_modern.py
产物: ./model/{vec,LR,gbdt,id2label}.pkl(覆盖旧版,兼容当前 sklearn)
"""
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def load_data(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.rsplit(",", 1)
            if len(parts) != 2:
                continue
            texts.append(" ".join(list(parts[0].lower())))
            labels.append(parts[1])
    return texts, labels


def train_and_save(data_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    texts, labels = load_data(data_path)
    label_set = sorted(set(labels))
    label2id = {l: i for i, l in enumerate(label_set)}
    id2label = {i: l for l, i in label2id.items()}
    y = np.array([label2id[l] for l in labels])

    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(texts)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X, y)
    gbdt = GradientBoostingClassifier(random_state=42)
    gbdt.fit(X.toarray(), y)

    proba = (lr.predict_proba(X) + gbdt.predict_proba(X.toarray())) / 2
    acc = float((proba.argmax(1) == y).mean())

    with open(os.path.join(out_dir, "vec.pkl"), "wb") as f:
        pickle.dump(vec, f)
    with open(os.path.join(out_dir, "LR.pkl"), "wb") as f:
        pickle.dump(lr, f)
    with open(os.path.join(out_dir, "gbdt.pkl"), "wb") as f:
        pickle.dump(gbdt, f)
    with open(os.path.join(out_dir, "id2label.pkl"), "wb") as f:
        pickle.dump(id2label, f)
    return {"acc": acc}


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    res = train_and_save(
        os.path.join(HERE, "data", "intent_recog_data.txt"),
        os.path.join(HERE, "model"),
    )
    print("train acc:", res["acc"])
