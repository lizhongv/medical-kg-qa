# PyTorch 训练包迁移 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为意图识别、NER、闲聊三个模型提供可在 GPU 环境直接训练的 PyTorch 训练包(数据 + 脚本 + 基座模型)。

**Architecture:** 在每个模型目录下新建 `pytorch/` 子目录,完全不改现有 TF 代码。三个训练包相互独立,各自带数据加载、模型定义、训练脚本、依赖与说明。本机仅做 CPU 冒烟验证(小批数据跑通管线与前向),完整收敛由用户在 GPU 完成。

**Tech Stack:** Python 3.10+,PyTorch 2.x,HuggingFace `transformers`,`pytorch-crf`,`scikit-learn`,`pandas`,`pytest`。

## Global Constraints

- 不修改既有 TF/Keras 代码;新代码只放在 `*/pytorch/` 子目录或新增独立脚本。
- 训练脚本一律设备无关:`device = 'cuda' if torch.cuda.is_available() else 'cpu'`。
- 不使用 `bert4keras` / `tensorflow` / `keras` 任何依赖。
- 意图基座模型固定 `hfl/rbt3`(3 层,hidden 768,vocab 21128)。
- NER 数据源固定 `chenxshuo/ChineseBLUE` 的 `data/cMedQANER/{train,dev,test}.txt`(字符级 BIO,标签形如 `B_disease`/`I_disease`/`O`,空行分句,11 类实体)。
- 所有下载走 GitHub raw(代理已验证可用);HuggingFace 大文件 CDN 不可用,基座模型走"本地优先 + Hub 回退"。
- 字符级处理:意图用 `hfl/rbt3` tokenizer;NER 按字符切分(`list(text)`)。

---

### Task 1: NER 数据下载与 BIO 加载器

**Files:**
- Create: `knowledge_extraction/bilstm_crf/pytorch/data/.gitkeep`
- Create: `knowledge_extraction/bilstm_crf/pytorch/download_data.py`
- Create: `knowledge_extraction/bilstm_crf/pytorch/data_loader.py`
- Test: `knowledge_extraction/bilstm_crf/pytorch/tests/test_data_loader.py`

**Interfaces:**
- Produces:
  - `download_data.py` 下载三份文件到 `pytorch/data/{train,dev,test}.txt`
  - `read_bio(path) -> list[tuple[list[str], list[str]]]`:返回 `[(chars, tags), ...]`,按空行分句
  - `build_vocab(sentences, min_freq=1) -> dict[str,int]`:含 `'<PAD>':0, '<UNK>':1`
  - `build_tag2id(sentences) -> dict[str,int]`:含 `'<PAD>':0`,其余标签按排序分配
  - `class NerDataset(torch.utils.data.Dataset)`:`__getitem__ -> (char_ids:LongTensor, tag_ids:LongTensor, length:int)`
  - `collate_fn(batch) -> (padded_chars, padded_tags, mask:BoolTensor, lengths)`,pad 值 0,mask 标记非 pad 位置

- [ ] **Step 1: 写下载脚本**

`download_data.py`:
```python
# -*- coding: utf-8 -*-
"""下载 cMedQANER(ChineseBLUE)到 ./data/。GitHub raw 通道。"""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/chenxshuo/ChineseBLUE/master/data/cMedQANER"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in ["train.txt", "dev.txt", "test.txt"]:
        dst = os.path.join(OUT, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"已存在,跳过: {name}")
            continue
        url = f"{BASE}/{name}"
        print(f"下载 {url}")
        urllib.request.urlretrieve(url, dst)
        print(f"  -> {dst} ({os.path.getsize(dst)} bytes)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行下载脚本**

Run: `cd knowledge_extraction/bilstm_crf/pytorch && python download_data.py`
Expected: 生成 `data/train.txt`(约 177503 行)、`data/dev.txt`、`data/test.txt`。

- [ ] **Step 3: 写失败测试**

`tests/test_data_loader.py`:
```python
import os
import torch
from knowledge_extraction.bilstm_crf.pytorch.data_loader import (
    read_bio, build_vocab, build_tag2id, NerDataset, collate_fn,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "..", "data", "train.txt")


def test_read_bio_basic():
    sents = read_bio(TRAIN)
    assert len(sents) > 1000
    chars, tags = sents[0]
    assert len(chars) == len(tags)
    assert all(t == "O" or t.startswith("B_") or t.startswith("I_") for t in tags)


def test_build_maps():
    sents = read_bio(TRAIN)
    vocab = build_vocab(sents)
    tag2id = build_tag2id(sents)
    assert vocab["<PAD>"] == 0 and vocab["<UNK>"] == 1
    assert tag2id["<PAD>"] == 0
    assert "B_disease" in tag2id


def test_dataset_and_collate():
    sents = read_bio(TRAIN)[:8]
    vocab = build_vocab(sents)
    tag2id = build_tag2id(sents)
    ds = NerDataset(sents, vocab, tag2id)
    batch = [ds[i] for i in range(4)]
    chars, tags, mask, lengths = collate_fn(batch)
    assert chars.shape == tags.shape == mask.shape
    assert chars.shape[0] == 4
    assert mask.dtype == torch.bool
```

- [ ] **Step 4: 运行测试确认失败**

Run: `pytest knowledge_extraction/bilstm_crf/pytorch/tests/test_data_loader.py -v`
Expected: FAIL(`ModuleNotFoundError: data_loader`)

- [ ] **Step 5: 实现 data_loader.py**

```python
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset


def read_bio(path):
    sentences = []
    chars, tags = [], []
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


def build_vocab(sentences, min_freq=1):
    from collections import Counter
    c = Counter()
    for chars, _ in sentences:
        c.update(chars)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for ch, freq in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq >= min_freq:
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
        self.sentences = sentences
        self.vocab = vocab
        self.tag2id = tag2id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        chars, tags = self.sentences[idx]
        cid = [self.vocab.get(c, 1) for c in chars]
        tid = [self.tag2id[t] for t in tags]
        return (torch.tensor(cid, dtype=torch.long),
                torch.tensor(tid, dtype=torch.long),
                len(cid))


def collate_fn(batch):
    lengths = [b[2] for b in batch]
    maxlen = max(lengths)
    n = len(batch)
    chars = torch.zeros(n, maxlen, dtype=torch.long)
    tags = torch.zeros(n, maxlen, dtype=torch.long)
    mask = torch.zeros(n, maxlen, dtype=torch.bool)
    for i, (cid, tid, ln) in enumerate(batch):
        chars[i, :ln] = cid
        tags[i, :ln] = tid
        mask[i, :ln] = True
    return chars, tags, mask, torch.tensor(lengths, dtype=torch.long)
```

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest knowledge_extraction/bilstm_crf/pytorch/tests/test_data_loader.py -v`
Expected: PASS(3 passed)

- [ ] **Step 7: 提交**

```bash
git add knowledge_extraction/bilstm_crf/pytorch/
git commit -m "feat(ner): cMedQANER downloader and BIO data loader"
```

---

### Task 2: NER 模型(BiLSTM-CRF)

**Files:**
- Create: `knowledge_extraction/bilstm_crf/pytorch/model.py`
- Test: `knowledge_extraction/bilstm_crf/pytorch/tests/test_model.py`

**Interfaces:**
- Consumes: `collate_fn` 输出 `(chars, tags, mask, lengths)`
- Produces:
  - `class BiLstmCrf(nn.Module)`,`__init__(vocab_size, num_tags, emb_dim=128, hidden=128, dropout=0.5)`
  - `forward(chars, mask) -> emissions:FloatTensor[B,L,num_tags]`
  - `loss(chars, tags, mask) -> scalar`(CRF 负对数似然)
  - `decode(chars, mask) -> list[list[int]]`(Viterbi 最优路径)

- [ ] **Step 1: 写失败测试**

`tests/test_model.py`:
```python
import torch
from knowledge_extraction.bilstm_crf.pytorch.model import BiLstmCrf


def test_forward_shapes():
    m = BiLstmCrf(vocab_size=50, num_tags=7)
    chars = torch.randint(0, 50, (4, 10))
    mask = torch.ones(4, 10, dtype=torch.bool)
    em = m(chars, mask)
    assert em.shape == (4, 10, 7)


def test_loss_and_decode():
    m = BiLstmCrf(vocab_size=50, num_tags=7)
    chars = torch.randint(0, 50, (4, 10))
    tags = torch.randint(0, 7, (4, 10))
    mask = torch.ones(4, 10, dtype=torch.bool)
    loss = m.loss(chars, tags, mask)
    assert loss.dim() == 0 and loss.item() > 0
    paths = m.decode(chars, mask)
    assert len(paths) == 4 and len(paths[0]) == 10
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest knowledge_extraction/bilstm_crf/pytorch/tests/test_model.py -v`
Expected: FAIL(`ModuleNotFoundError: model`)

- [ ] **Step 3: 实现 model.py**

```python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchcrf import CRF


class BiLstmCrf(nn.Module):
    def __init__(self, vocab_size, num_tags, emb_dim=128, hidden=128, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, chars, mask):
        x = self.embedding(chars)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return self.hidden2tag(x)

    def loss(self, chars, tags, mask):
        emissions = self(chars, mask)
        return -self.crf(emissions, tags, mask=mask, reduction="mean")

    def decode(self, chars, mask):
        emissions = self(chars, mask)
        return self.crf.decode(emissions, mask=mask)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest knowledge_extraction/bilstm_crf/pytorch/tests/test_model.py -v`
Expected: PASS(2 passed)。若报 `No module named torchcrf`,先 `pip install pytorch-crf`。

- [ ] **Step 5: 提交**

```bash
git add knowledge_extraction/bilstm_crf/pytorch/model.py knowledge_extraction/bilstm_crf/pytorch/tests/test_model.py
git commit -m "feat(ner): BiLSTM-CRF model with pytorch-crf"
```

---

### Task 3: NER 训练脚本与依赖/说明

**Files:**
- Create: `knowledge_extraction/bilstm_crf/pytorch/train.py`
- Create: `knowledge_extraction/bilstm_crf/pytorch/predict.py`
- Create: `knowledge_extraction/bilstm_crf/pytorch/requirements.txt`
- Create: `knowledge_extraction/bilstm_crf/pytorch/README.md`

**Interfaces:**
- Consumes: Task 1 的 loader、Task 2 的 `BiLstmCrf`
- Produces: 训练后生成 `checkpoint/best_ner.pt`、`checkpoint/vocab.json`、`checkpoint/tag2id.json`

- [ ] **Step 1: 写 requirements.txt**

```
torch>=2.0
pytorch-crf==0.7.2
tqdm
```

- [ ] **Step 2: 写 train.py**

```python
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
    json.dump(vocab, open(os.path.join(CKPT, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(tag2id, open(os.path.join(CKPT, "tag2id.json"), "w", encoding="utf-8"), ensure_ascii=False)

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
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), os.path.join(CKPT, "best_ner.pt"))
            print("  saved best_ner.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: CPU 冒烟训练(小批)**

Run: `cd knowledge_extraction/bilstm_crf/pytorch && python train.py --limit 200 --epochs 1 --batch_size 16`
Expected: 打印 device、1 个 epoch 的 loss 与 dev_token_acc,生成 `checkpoint/{best_ner.pt,vocab.json,tag2id.json}`,无报错。

- [ ] **Step 4: 写 predict.py**

```python
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
```

- [ ] **Step 5: 运行 predict 自检**

Run: `cd knowledge_extraction/bilstm_crf/pytorch && python predict.py`
Expected: 打印每个字符与其预测标签的列表(冒烟模型,标签未必准,但无报错)。

- [ ] **Step 6: 写 README.md**

```markdown
# NER(BiLSTM-CRF)PyTorch 训练包

## 安装
pip install -r requirements.txt

## 数据
python download_data.py   # 下载 cMedQANER 到 ./data/

## 训练
python train.py --epochs 20 --batch_size 64        # 全量
python train.py --limit 200 --epochs 1             # 冒烟

## 产物
checkpoint/best_ner.pt, vocab.json, tag2id.json

## 说明
本模型用随机初始化字向量从零训练,无需基座预训练模型。
实体类型:body/crowd/department/disease/drug/feature/physiology/symptom/test/time/treatment。
```

- [ ] **Step 7: 提交**

```bash
git add knowledge_extraction/bilstm_crf/pytorch/
git commit -m "feat(ner): training/predict scripts, requirements, README"
```

---

### Task 4: 意图基座模型获取

**Files:**
- Create: `nlu/bert_intent_recognition/pytorch/download_base.py`
- Create: `nlu/bert_intent_recognition/pytorch/base_model/.gitkeep`

**Interfaces:**
- Produces: 尽量把 `hfl/rbt3` 下到 `pytorch/base_model/`;失败时打印指引,训练脚本回退 Hub。

- [ ] **Step 1: 写 download_base.py**

```python
# -*- coding: utf-8 -*-
"""尝试把 hfl/rbt3 下到 ./base_model/。若网络受限,训练脚本会回退到 Hub 名称。"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "base_model")


def main():
    os.makedirs(OUT, exist_ok=True)
    try:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("hfl/rbt3")
        mdl = AutoModel.from_pretrained("hfl/rbt3")
        tok.save_pretrained(OUT)
        mdl.save_pretrained(OUT)
        print("基座模型已保存到", OUT)
    except Exception as e:
        print("下载失败:", e)
        print("可在 GPU 机器联网后重跑本脚本,或让 train.py 直接用 'hfl/rbt3' 自动下载。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 尝试运行(允许失败)**

Run: `cd nlu/bert_intent_recognition/pytorch && python download_base.py`
Expected: 成功则 `base_model/` 出现 `config.json/pytorch_model.bin/vocab.txt`;失败则打印回退指引(本机代理受限属预期)。

- [ ] **Step 3: 提交**

```bash
git add nlu/bert_intent_recognition/pytorch/download_base.py nlu/bert_intent_recognition/pytorch/base_model/.gitkeep
git commit -m "feat(intent): base model (hfl/rbt3) downloader with Hub fallback"
```

---

### Task 5: 意图模型(RBT3 + TextCNN)与数据加载

**Files:**
- Create: `nlu/bert_intent_recognition/pytorch/model.py`
- Create: `nlu/bert_intent_recognition/pytorch/data_loader.py`
- Test: `nlu/bert_intent_recognition/pytorch/tests/test_model.py`

**Interfaces:**
- Produces:
  - `resolve_base() -> str`:存在本地 `base_model/` 则返回其路径,否则返回 `'hfl/rbt3'`
  - `class IntentModel(nn.Module)`,`__init__(base_name_or_path, num_labels=13)`,`forward(input_ids, attention_mask) -> logits[B,num_labels]`
  - `load_csv(path) -> (texts:list[str], labels:list[int])`
  - `class IntentDataset(Dataset)`:用 tokenizer 编码,`__getitem__ -> dict(input_ids, attention_mask, label)`
  - `make_collate(tokenizer) -> collate_fn`

- [ ] **Step 1: 写失败测试**(用本地构造的微型 BERT,避免依赖下载)

`tests/test_model.py`:
```python
import torch
from transformers import BertConfig, BertModel
from nlu.bert_intent_recognition.pytorch.model import TextCnnHead


def test_textcnn_head_shape():
    head = TextCnnHead(hidden=32, num_labels=13)
    cls = torch.randn(4, 32)
    tokens = torch.randn(4, 18, 32)  # [B, L-2, H]
    logits = head(cls, tokens)
    assert logits.shape == (4, 13)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest nlu/bert_intent_recognition/pytorch/tests/test_model.py -v`
Expected: FAIL(`ModuleNotFoundError: model`)

- [ ] **Step 3: 实现 model.py**

```python
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from transformers import AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_base():
    local = os.path.join(HERE, "base_model")
    if os.path.exists(os.path.join(local, "config.json")):
        return local
    return "hfl/rbt3"


class TextCnnHead(nn.Module):
    def __init__(self, hidden, num_labels, channels=256, kernels=(3, 4, 5), dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden, channels, k, padding=k // 2) for k in kernels
        ])
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden + channels * len(kernels), 512)
        self.act = nn.ReLU()
        self.out = nn.Linear(512, num_labels)

    def forward(self, cls, tokens):
        # tokens: [B, L, H] -> [B, H, L]
        x = tokens.transpose(1, 2)
        feats = []
        for conv in self.convs:
            c = torch.relu(conv(x))            # [B, C, L]
            c = torch.max(c, dim=2).values     # [B, C]
            feats.append(c)
        cnn = self.dropout(torch.cat(feats, dim=1))
        h = self.act(self.dense(torch.cat([cls, cnn], dim=1)))
        return self.out(h)


class IntentModel(nn.Module):
    def __init__(self, base_name_or_path, num_labels=13):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_name_or_path)
        hidden = self.bert.config.hidden_size
        self.head = TextCnnHead(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state          # [B, L, H]
        cls = seq[:, 0]                      # [B, H]
        tokens = seq[:, 1:-1]                # [B, L-2, H]
        return self.head(cls, tokens)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest nlu/bert_intent_recognition/pytorch/tests/test_model.py -v`
Expected: PASS(1 passed)

- [ ] **Step 5: 实现 data_loader.py**

```python
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
```

- [ ] **Step 6: 提交**

```bash
git add nlu/bert_intent_recognition/pytorch/model.py nlu/bert_intent_recognition/pytorch/data_loader.py nlu/bert_intent_recognition/pytorch/tests/
git commit -m "feat(intent): RBT3+TextCNN model and CSV data loader"
```

---

### Task 6: 意图训练脚本与依赖/说明

**Files:**
- Create: `nlu/bert_intent_recognition/pytorch/train.py`
- Create: `nlu/bert_intent_recognition/pytorch/predict.py`
- Create: `nlu/bert_intent_recognition/pytorch/requirements.txt`
- Create: `nlu/bert_intent_recognition/pytorch/README.md`

**Interfaces:**
- Consumes: Task 5 的 `IntentModel/resolve_base/load_csv/IntentDataset/make_collate`,Task 4 的 `base_model/`
- Produces: `checkpoint/best_model.pt`、`checkpoint/label2id.json`

- [ ] **Step 1: 写 requirements.txt**

```
torch>=2.0
transformers>=4.30
pandas
scikit-learn
tqdm
```

- [ ] **Step 2: 写 train.py**

```python
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
```

- [ ] **Step 3: 写 predict.py**

```python
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
```

- [ ] **Step 4: 写 README.md**

```markdown
# 意图识别(RBT3 + TextCNN)PyTorch 训练包

## 安装
pip install -r requirements.txt

## 基座模型
python download_base.py   # 尝试下载 hfl/rbt3 到 ./base_model/
# 若失败,train.py 会自动用 Hub 名 'hfl/rbt3'(需联网)

## 训练
python train.py --epochs 5 --batch_size 32         # 全量
python train.py --limit 100 --epochs 1             # 冒烟

## 产物
checkpoint/best_model.pt, checkpoint/label2id.json

## 数据
读取 ../data/{train,test}.csv(仓库已自带,字段 text,label_class,label)
```

- [ ] **Step 5: 冒烟说明(本机不强制运行)**

本机代理无法下载 `hfl/rbt3`,故 `train.py` 全量/冒烟在本机可能无法联网获取基座。
若 `download_base.py` 成功则可运行:`python train.py --limit 100 --epochs 1`(预期打印 device/base/1 epoch 指标并保存产物)。
否则在 GPU 机器上运行。README 与本步骤已注明此前置条件。

- [ ] **Step 6: 提交**

```bash
git add nlu/bert_intent_recognition/pytorch/
git commit -m "feat(intent): training/predict scripts, requirements, README"
```

---

### Task 7: 闲聊分类现代 sklearn 重训脚本

**Files:**
- Create: `nlu/sklearn_Classification/train_modern.py`
- Test: `nlu/sklearn_Classification/tests/test_train_modern.py`

**Interfaces:**
- Consumes: `nlu/sklearn_Classification/data/intent_recog_data.txt`(`文本<制表符>标签`)
- Produces: `model_file/{vec.pkl, LR.pkl, gbdt.pkl, id2label.pkl}`;`train_and_save(data_path, out_dir) -> dict(acc)`

- [ ] **Step 1: 写失败测试**

`tests/test_train_modern.py`:
```python
import os
import pickle
from nlu.sklearn_Classification.train_modern import train_and_save

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "intent_recog_data.txt")
OUT = os.path.join(HERE, "..", "model_file")


def test_train_and_save(tmp_path):
    out = str(tmp_path)
    res = train_and_save(DATA, out)
    assert res["acc"] >= 0.5
    for fn in ["vec.pkl", "LR.pkl", "gbdt.pkl", "id2label.pkl"]:
        assert os.path.exists(os.path.join(out, fn))
    id2label = pickle.load(open(os.path.join(out, "id2label.pkl"), "rb"))
    assert "greet" in id2label.values()
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest nlu/sklearn_Classification/tests/test_train_modern.py -v`
Expected: FAIL(`ModuleNotFoundError: train_modern`)

- [ ] **Step 3: 实现 train_modern.py**

```python
# -*- coding: utf-8 -*-
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

    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    gbdt = GradientBoostingClassifier()
    gbdt.fit(X.toarray(), y)

    proba = (lr.predict_proba(X) + gbdt.predict_proba(X.toarray())) / 2
    acc = float((proba.argmax(1) == y).mean())

    pickle.dump(vec, open(os.path.join(out_dir, "vec.pkl"), "wb"))
    pickle.dump(lr, open(os.path.join(out_dir, "LR.pkl"), "wb"))
    pickle.dump(gbdt, open(os.path.join(out_dir, "gbdt.pkl"), "wb"))
    pickle.dump(id2label, open(os.path.join(out_dir, "id2label.pkl"), "wb"))
    return {"acc": acc}


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    res = train_and_save(
        os.path.join(HERE, "data", "intent_recog_data.txt"),
        os.path.join(HERE, "model_file"),
    )
    print("train acc:", res["acc"])
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest nlu/sklearn_Classification/tests/test_train_modern.py -v`
Expected: PASS(1 passed)

- [ ] **Step 5: 实跑生成产物**

Run: `cd nlu/sklearn_Classification && python train_modern.py`
Expected: 打印 `train acc: <值>`,`model_file/` 下生成四个 pkl。

- [ ] **Step 6: 提交**

```bash
git add nlu/sklearn_Classification/train_modern.py nlu/sklearn_Classification/tests/
git commit -m "feat(chitchat): modern sklearn retraining script"
```

---

### Task 8: 顶层训练总说明 TRAINING.md

**Files:**
- Create: `TRAINING.md`

- [ ] **Step 1: 写 TRAINING.md**

```markdown
# 训练总说明(PyTorch 迁移)

三个模型的训练包,均可在 GPU(或 CPU 回退)运行。本文件只覆盖训练侧;
推理侧改造(app.py/modules.py 加载新权重)为后续单独步骤。

## 1. 意图识别(RBT3 + TextCNN)
cd nlu/bert_intent_recognition/pytorch
pip install -r requirements.txt
python download_base.py        # 下载 hfl/rbt3 基座(失败则训练时自动从 Hub 拉)
python train.py --epochs 5 --batch_size 32
# 产物: checkpoint/best_model.pt, label2id.json

## 2. NER(BiLSTM-CRF)
cd knowledge_extraction/bilstm_crf/pytorch
pip install -r requirements.txt
python download_data.py        # 下载 cMedQANER
python train.py --epochs 20 --batch_size 64
# 产物: checkpoint/best_ner.pt, vocab.json, tag2id.json

## 3. 闲聊分类(sklearn)
cd nlu/sklearn_Classification
python train_modern.py
# 产物: model_file/{vec,LR,gbdt,id2label}.pkl

## 冒烟测试(快速验证管线)
意图: python train.py --limit 100 --epochs 1
NER:  python train.py --limit 200 --epochs 1

## 训练完成后
把三组产物保留在各自 checkpoint/model_file 目录,等待推理侧改造对接。
```

- [ ] **Step 2: 提交**

```bash
git add TRAINING.md
git commit -m "docs: top-level training guide for PyTorch migration"
```

---

## 自检结果

- **Spec 覆盖**:模型1(意图)→ Task 4/5/6;模型2(NER)→ Task 1/2/3;模型3(闲聊)→ Task 7;目录布局/TRAINING.md → Task 8;基座下载风险 → Task 4 + Task 6 Step 5。全部覆盖。
- **占位符**:无 TODO/TBD;每个代码步骤均给出完整代码。
- **类型一致性**:`collate_fn`→`(chars,tags,mask,lengths)` 在 Task1/2/3 一致;`IntentModel/resolve_base/load_csv/make_collate` 在 Task5/6 一致;`train_and_save(data_path,out_dir)->{acc}` 在 Task7 一致。
- **已知前置**:意图训练需 `hfl/rbt3`(本机代理可能无法下载,GPU 机联网即可);NER/闲聊本机 CPU 可冒烟。
