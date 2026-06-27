# LLM 级联对话架构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把医疗 KBQA 重构为「小模型快路 + 云端 LLM 慢路 + Neo4j 接地」的级联架构,对外 REST API,含生产件(Redis/护栏/缓存),缺外部依赖时优雅降级。

**Architecture:** 新代码放在 `app/` 包内,按职责分模块(api/orchestrator/nlu/kg/llm/memory/safety/cache)。快路只用本地小模型推理,LLM 只在慢路经统一 OpenAI 兼容客户端调用。所有外部依赖(Neo4j/Redis/LLM/embedding)缺失时降级而非崩溃,因此绝大多数任务可在本机用 mock/stub 单测。

**Tech Stack:** Python 3.10+,FastAPI,uvicorn,openai SDK,neo4j 驱动,redis-py,scikit-learn,pyahocorasick,pytest,httpx。

## Global Constraints

- 新代码全部置于 `app/` 包;不引入 tensorflow/keras/bert4keras。
- 快路 NLU 不得调用 LLM 或任何网络;只做本地推理。
- LLM 访问只经 `app/llm/client.py`;别处不直接发 HTTP 调 LLM。
- 外部连接(Neo4j/Redis/LLM)集中在 `app/settings.py`,密钥走环境变量,不硬编码。
- LLM 慢路回答只能基于 `kg` 返回的事实,不得自由编造医疗结论。
- 每模块可独立单测;缺外部依赖时降级(无 Redis→内存;无 LLM key→慢路退提示;无 PyTorch 权重→词典/规则兜底;无 embedding→精确匹配)。
- 不破坏离线资产:`build_kg_utils.py`、`config.semantic_slot`、三个 `*/pytorch/` 训练包。
- 从仓库根运行 pytest(包路径导入 `app.xxx`)。
- 每个 commit 只暂存本任务文件,不用 `git add -A`(工作树有未跟踪大文件与改动)。

## 文件结构

```
app/
  __init__.py
  settings.py                 配置(env + 默认值)
  llm/
    __init__.py
    client.py                 OpenAI 兼容客户端
    understand.py             慢路理解(意图+实体)
    generate.py               慢路回答生成(流式)
  kg/
    __init__.py
    neo4j_client.py           Neo4j 查询封装
    templates.py              意图→参数化 Cypher(复用 config.semantic_slot)
    text2cypher.py            慢路 LLM 生成 Cypher
  nlu/
    __init__.py
    chitchat.py               sklearn 闲聊分类
    intent.py                 PyTorch 意图(降级:unavailable)
    slot.py                   词典AC + PyTorch NER(降级:仅词典)
    normalize.py              实体归一化→标准ID
    pipeline.py               analyze(text)->NluResult 聚合
  orchestrator/
    __init__.py
    router.py                 路由决策
    controller.py             主流程编排
  memory/
    __init__.py
    store.py                  Redis + 内存降级
  safety/
    __init__.py
    guardrails.py             免责/拒诊断/PII/事实校验
  cache/
    __init__.py
    semantic_cache.py         语义缓存(降级精确)
  api/
    __init__.py
    main.py                   FastAPI /chat /health
  tests/
    __init__.py
    test_*.py
requirements-service.txt
app/README.md
```

---

## 阶段一:骨架

### Task 1: 配置 settings.py

**Files:**
- Create: `app/__init__.py`(空)、`app/settings.py`、`app/tests/__init__.py`(空)、`app/tests/test_settings.py`

**Interfaces:**
- Produces:
  - `class Settings`,字段:`accept_threshold:float=0.8`,`deny_threshold:float=0.4`,`neo4j_uri:str`,`neo4j_user:str`,`neo4j_password:str`,`redis_url:str|None`,`llm_base_url:str|None`,`llm_api_key:str|None`,`llm_model:str`
  - `load_settings(env: dict | None = None) -> Settings`:从 env 读取,缺失用默认值

- [ ] **Step 1: 写失败测试**

`app/tests/test_settings.py`:
```python
from app.settings import load_settings


def test_defaults_when_env_empty():
    s = load_settings({})
    assert s.accept_threshold == 0.8
    assert s.deny_threshold == 0.4
    assert s.neo4j_uri == "bolt://127.0.0.1:7687"
    assert s.llm_api_key is None


def test_reads_from_env():
    s = load_settings({
        "KBQA_ACCEPT_THRESHOLD": "0.7",
        "KBQA_LLM_API_KEY": "sk-x",
        "KBQA_LLM_MODEL": "gpt-4o-mini",
    })
    assert s.accept_threshold == 0.7
    assert s.llm_api_key == "sk-x"
    assert s.llm_model == "gpt-4o-mini"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_settings.py -v`
Expected: FAIL(`ModuleNotFoundError: app.settings`)

- [ ] **Step 3: 实现 settings.py**

```python
# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass


@dataclass
class Settings:
    accept_threshold: float = 0.8
    deny_threshold: float = 0.4
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "123456"
    redis_url: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"


def load_settings(env: dict | None = None) -> Settings:
    e = os.environ if env is None else env

    def f(key, default):
        v = e.get(key)
        return float(v) if v is not None else default

    return Settings(
        accept_threshold=f("KBQA_ACCEPT_THRESHOLD", 0.8),
        deny_threshold=f("KBQA_DENY_THRESHOLD", 0.4),
        neo4j_uri=e.get("KBQA_NEO4J_URI", "bolt://127.0.0.1:7687"),
        neo4j_user=e.get("KBQA_NEO4J_USER", "neo4j"),
        neo4j_password=e.get("KBQA_NEO4J_PASSWORD", "123456"),
        redis_url=e.get("KBQA_REDIS_URL"),
        llm_base_url=e.get("KBQA_LLM_BASE_URL"),
        llm_api_key=e.get("KBQA_LLM_API_KEY"),
        llm_model=e.get("KBQA_LLM_MODEL", "gpt-4o-mini"),
    )
```
同时创建空文件 `app/__init__.py`、`app/tests/__init__.py`。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_settings.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add app/__init__.py app/settings.py app/tests/__init__.py app/tests/test_settings.py
git commit -m "feat(app): settings loader with env + defaults"
```

---

### Task 2: LLM 客户端 llm/client.py

**Files:**
- Create: `app/llm/__init__.py`(空)、`app/llm/client.py`、`app/tests/test_llm_client.py`

**Interfaces:**
- Consumes: `app.settings.Settings`
- Produces:
  - `class LLMClient`,`__init__(settings)`;`available -> bool`(有 key 才 True)
  - `chat(messages: list[dict], tools: list | None = None, stream: bool = False) -> dict`:返回 `{"content": str, "tool_calls": list}`;无 key 时抛 `LLMUnavailable`
  - `class LLMUnavailable(Exception)`
  - 内部用 `openai.OpenAI(base_url, api_key)`;注入点 `self._client` 便于测试替换

- [ ] **Step 1: 写失败测试**(用假 client,不联网)

`app/tests/test_llm_client.py`:
```python
import pytest
from app.settings import Settings
from app.llm.client import LLMClient, LLMUnavailable


class FakeCompletions:
    def create(self, **kw):
        class M:
            content = "hello"
            tool_calls = None
        class C:
            message = M()
        class R:
            choices = [C()]
        return R()


class FakeClient:
    def __init__(self):
        self.chat = type("X", (), {"completions": FakeCompletions()})()


def test_unavailable_without_key():
    c = LLMClient(Settings(llm_api_key=None))
    assert c.available is False
    with pytest.raises(LLMUnavailable):
        c.chat([{"role": "user", "content": "hi"}])


def test_chat_returns_content():
    c = LLMClient(Settings(llm_api_key="sk-x"))
    c._client = FakeClient()
    out = c.chat([{"role": "user", "content": "hi"}])
    assert out["content"] == "hello"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_llm_client.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 client.py**

```python
# -*- coding: utf-8 -*-
class LLMUnavailable(Exception):
    pass


class LLMClient:
    def __init__(self, settings):
        self.settings = settings
        self._client = None
        if settings.llm_api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=settings.llm_base_url,
                    api_key=settings.llm_api_key,
                )
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def chat(self, messages, tools=None, stream=False):
        if not self.available:
            raise LLMUnavailable("no LLM client configured")
        resp = self._client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            tools=tools,
            stream=stream,
        )
        if stream:
            return resp  # 调用方自行迭代
        msg = resp.choices[0].message
        return {"content": msg.content or "", "tool_calls": getattr(msg, "tool_calls", None)}
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_llm_client.py -v`
Expected: PASS(2 passed)。若缺 `openai`,测试仍过(用 key 时构造失败则 `_client=None`,`test_chat_returns_content` 用注入的 FakeClient 不依赖 openai 包)。

- [ ] **Step 5: 提交**

```bash
git add app/llm/__init__.py app/llm/client.py app/tests/test_llm_client.py
git commit -m "feat(llm): OpenAI-compatible client with graceful unavailability"
```

---

### Task 3: Neo4j 客户端 kg/neo4j_client.py

**Files:**
- Create: `app/kg/__init__.py`(空)、`app/kg/neo4j_client.py`、`app/tests/test_neo4j_client.py`

**Interfaces:**
- Consumes: `app.settings.Settings`
- Produces:
  - `class KGClient`,`__init__(settings)`;`available -> bool`
  - `query(cypher: str, params: dict | None = None) -> list[dict]`:不可用时返回 `[]`
  - 内部驱动注入点 `self._driver`

- [ ] **Step 1: 写失败测试**(假 driver)

`app/tests/test_neo4j_client.py`:
```python
from app.settings import Settings
from app.kg.neo4j_client import KGClient


class FakeResult(list):
    def data(self):
        return list(self)


class FakeSession:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def run(self, cypher, **params):
        return FakeResult([{"name": "高血压"}])


class FakeDriver:
    def session(self): return FakeSession()


def test_unavailable_returns_empty():
    c = KGClient(Settings())
    c._driver = None
    assert c.available is False
    assert c.query("MATCH (n) RETURN n") == []


def test_query_returns_rows():
    c = KGClient(Settings())
    c._driver = FakeDriver()
    rows = c.query("MATCH (p) RETURN p.name", {})
    assert rows == [{"name": "高血压"}]
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_neo4j_client.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 neo4j_client.py**

```python
# -*- coding: utf-8 -*-
class KGClient:
    def __init__(self, settings):
        self.settings = settings
        self._driver = None
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
        except Exception:
            self._driver = None

    @property
    def available(self) -> bool:
        return self._driver is not None

    def query(self, cypher, params=None):
        if not self.available:
            return []
        params = params or {}
        try:
            with self._driver.session() as session:
                return list(session.run(cypher, **params).data())
        except Exception:
            return []
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_neo4j_client.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add app/kg/__init__.py app/kg/neo4j_client.py app/tests/test_neo4j_client.py
git commit -m "feat(kg): Neo4j client wrapper with graceful unavailability"
```

---

### Task 4: REST 骨架 api/main.py

**Files:**
- Create: `app/api/__init__.py`(空)、`app/api/main.py`、`app/tests/test_api.py`、`requirements-service.txt`

**Interfaces:**
- Produces:
  - FastAPI 应用 `app`(模块级)
  - `GET /health -> {"status": "ok", "deps": {...}}`
  - `POST /chat`,body `{"text": str, "session_id": str}` → `{"answer": str, "path": str}`(此阶段 `answer` 固定占位,`path="stub"`)

- [ ] **Step 1: 写 requirements-service.txt**

```
fastapi
uvicorn
openai
neo4j
redis
scikit-learn
pyahocorasick
pandas
torch
transformers
httpx
pytest
```

- [ ] **Step 2: 写失败测试**

`app/tests/test_api.py`:
```python
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat_stub():
    r = client.post("/chat", json={"text": "高血压怎么治", "session_id": "u1"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and "path" in body
```

- [ ] **Step 3: 运行确认失败**

Run: `python -m pytest app/tests/test_api.py -v`
Expected: FAIL(`ModuleNotFoundError: app.api.main`)

- [ ] **Step 4: 实现 main.py**

```python
# -*- coding: utf-8 -*-
from fastapi import FastAPI
from pydantic import BaseModel
from app.settings import load_settings
from app.kg.neo4j_client import KGClient
from app.llm.client import LLMClient

app = FastAPI(title="KBQA")
_settings = load_settings()
_kg = KGClient(_settings)
_llm = LLMClient(_settings)


class ChatIn(BaseModel):
    text: str
    session_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok", "deps": {"neo4j": _kg.available, "llm": _llm.available}}


@app.post("/chat")
def chat(req: ChatIn):
    # 骨架阶段:占位回复,后续任务接入 controller
    return {"answer": "(骨架)收到:" + req.text, "path": "stub"}
```

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest app/tests/test_api.py -v`
Expected: PASS(2 passed)。若缺 `fastapi`,先 `pip install fastapi httpx`。

- [ ] **Step 6: 提交**

```bash
git add app/api/__init__.py app/api/main.py app/tests/test_api.py requirements-service.txt
git commit -m "feat(api): FastAPI skeleton with /health and /chat stub"
```

---

## 阶段二:快路

### Task 5: 闲聊分类 nlu/chitchat.py

**Files:**
- Create: `app/nlu/__init__.py`(空)、`app/nlu/chitchat.py`、`app/tests/test_chitchat.py`

**Interfaces:**
- Produces:
  - `class Chitchat`,`__init__(model_dir: str | None = None)`;`available -> bool`
  - `classify(text: str) -> str | None`:返回 `greet|goodbye|deny|isbot` 或 None(非闲聊/不可用)
  - 降级:模型缺失时用关键词规则(`你好/hi→greet`,`再见/bye→goodbye`,`是的/对→deny? `→实际用 `accept` 词);仅在明确命中时返回

- [ ] **Step 1: 写失败测试**

`app/tests/test_chitchat.py`:
```python
from app.nlu.chitchat import Chitchat


def test_keyword_fallback_greet():
    c = Chitchat(model_dir=None)   # 无模型 → 关键词降级
    assert c.classify("你好") == "greet"
    assert c.classify("再见") == "goodbye"


def test_non_chitchat_returns_none():
    c = Chitchat(model_dir=None)
    assert c.classify("高血压的症状有哪些") is None
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_chitchat.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 chitchat.py**

```python
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
                self.vec = pickle.load(open(os.path.join(model_dir, "vec.pkl"), "rb"))
                self.clf = pickle.load(open(os.path.join(model_dir, "LR.pkl"), "rb"))
                self.id2label = pickle.load(open(os.path.join(model_dir, "id2label.pkl"), "rb"))
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
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_chitchat.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add app/nlu/__init__.py app/nlu/chitchat.py app/tests/test_chitchat.py
git commit -m "feat(nlu): chitchat classifier with keyword fallback"
```

---

### Task 6: 词典+模型 NER nlu/slot.py

**Files:**
- Create: `app/nlu/slot.py`、`app/tests/test_slot.py`

**Interfaces:**
- Produces:
  - `class SlotFiller`,`__init__(diseases_path: str)`;构建 AC 自动机
  - `extract(text: str) -> list[str]`:返回命中的疾病名(最长匹配去重)
  - 仅词典路即可满足快路;PyTorch NER 为可选增强(本任务先只做词典路,接口预留)

- [ ] **Step 1: 写失败测试**(用临时小词典)

`app/tests/test_slot.py`:
```python
import json
from app.nlu.slot import SlotFiller


def test_extract_from_dict(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps(["高血压", "高血压病", "感冒"], ensure_ascii=False), encoding="utf-8")
    sf = SlotFiller(str(p))
    got = sf.extract("高血压病怎么治")
    assert "高血压病" in got        # 命中最长
    assert sf.extract("我感冒了") == ["感冒"]


def test_no_hit(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps(["感冒"], ensure_ascii=False), encoding="utf-8")
    assert SlotFiller(str(p)).extract("你好呀") == []
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_slot.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 slot.py**

```python
# -*- coding: utf-8 -*-
import json
import ahocorasick


class SlotFiller:
    def __init__(self, diseases_path):
        words = json.load(open(diseases_path, encoding="utf-8"))
        self.actree = ahocorasick.Automaton()
        for i, w in enumerate(words):
            if w:
                self.actree.add_word(w, (i, w))
        self.actree.make_automaton()

    def extract(self, text):
        hits = [v[1] for _, v in self.actree.iter(text)]
        # 去掉被更长匹配包含的短词
        final = [w for w in hits if not any(w != o and w in o for o in hits)]
        # 去重保序
        seen, out = set(), []
        for w in final:
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_slot.py -v`
Expected: PASS(2 passed)。若缺 `pyahocorasick`,先安装。

- [ ] **Step 5: 提交**

```bash
git add app/nlu/slot.py app/tests/test_slot.py
git commit -m "feat(nlu): dictionary slot filler via Aho-Corasick"
```

---

### Task 7: 意图模型 nlu/intent.py(带降级)

**Files:**
- Create: `app/nlu/intent.py`、`app/tests/test_intent.py`

**Interfaces:**
- Produces:
  - `class IntentModel`,`__init__(ckpt_dir: str | None = None, base: str | None = None)`;`available -> bool`
  - `predict(text: str) -> dict`:`{"name": str, "confidence": float}`;不可用时返回 `{"name": None, "confidence": 0.0}`
  - 不可用 = 无权重或无 transformers/torch;此时由慢路 LLM 兜底意图

- [ ] **Step 1: 写失败测试**(无权重→降级)

`app/tests/test_intent.py`:
```python
from app.nlu.intent import IntentModel


def test_unavailable_returns_none():
    m = IntentModel(ckpt_dir=None)
    assert m.available is False
    out = m.predict("高血压怎么治")
    assert out["name"] is None and out["confidence"] == 0.0
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_intent.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 intent.py**

```python
# -*- coding: utf-8 -*-
import os
import json


class IntentModel:
    def __init__(self, ckpt_dir=None, base=None):
        self.model = self.tokenizer = self.id2name = None
        self._ok = False
        weights = os.path.join(ckpt_dir, "best_model.pt") if ckpt_dir else None
        if weights and os.path.exists(weights):
            try:
                import torch
                from transformers import AutoTokenizer
                # 复用训练侧定义(运行时按需导入,避免训练包耦合)
                from importlib import import_module
                train_mod = import_module(
                    "nlu.bert_intent_recognition.pytorch.train"
                ) if False else None
                # 直接内联最小推理:加载 state_dict 到等价结构
                self._torch = torch
                self.id2name = json.load(open(os.path.join(ckpt_dir, "label2id.json"), encoding="utf-8"))
                self.tokenizer = AutoTokenizer.from_pretrained(base or "hfl/rbt3")
                self.model = _load_intent_model(base or "hfl/rbt3", weights, len(self.id2name), torch)
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
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    return net
```
> 注:推理结构与训练侧 `nlu/bert_intent_recognition/pytorch/train.py` 的 `IntentModel` 完全一致,以保证 `load_state_dict` 对齐。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_intent.py -v`
Expected: PASS(1 passed)(无权重走降级,不触发 torch/transformers 导入)

- [ ] **Step 5: 提交**

```bash
git add app/nlu/intent.py app/tests/test_intent.py
git commit -m "feat(nlu): intent model loader with graceful degradation"
```

---

### Task 8: NLU 聚合 nlu/pipeline.py

**Files:**
- Create: `app/nlu/pipeline.py`、`app/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Chitchat`、`IntentModel`、`SlotFiller`
- Produces:
  - `class NluPipeline`,`__init__(chitchat, intent, slot)`
  - `analyze(text: str) -> dict`:`NluResult` = `{"kind","intent","confidence","slots","matched"}`
    - 闲聊命中 → `kind="chitchat", intent=<闲聊类>`
    - 否则 `kind="diagnosis"`,填 `intent/confidence`(意图模型)+ `slots={"Disease": <词典首个命中或None>}`,`matched = slots["Disease"] is not None`

- [ ] **Step 1: 写失败测试**(用假组件)

`app/tests/test_pipeline.py`:
```python
from app.nlu.pipeline import NluPipeline


class FakeChit:
    def classify(self, t): return "greet" if t == "你好" else None
class FakeIntent:
    def predict(self, t): return {"name": "治疗方法", "confidence": 0.92}
class FakeSlot:
    def extract(self, t): return ["高血压"] if "高血压" in t else []


def test_chitchat_branch():
    p = NluPipeline(FakeChit(), FakeIntent(), FakeSlot())
    r = p.analyze("你好")
    assert r["kind"] == "chitchat" and r["intent"] == "greet"


def test_diagnosis_branch():
    p = NluPipeline(FakeChit(), FakeIntent(), FakeSlot())
    r = p.analyze("高血压怎么治")
    assert r["kind"] == "diagnosis"
    assert r["intent"] == "治疗方法" and r["confidence"] == 0.92
    assert r["slots"]["Disease"] == "高血压" and r["matched"] is True
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_pipeline.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 pipeline.py**

```python
# -*- coding: utf-8 -*-
class NluPipeline:
    def __init__(self, chitchat, intent, slot):
        self.chitchat = chitchat
        self.intent = intent
        self.slot = slot

    def analyze(self, text):
        chat = self.chitchat.classify(text)
        if chat:
            return {"kind": "chitchat", "intent": chat, "confidence": 1.0,
                    "slots": {"Disease": None}, "matched": False}
        intent = self.intent.predict(text)
        diseases = self.slot.extract(text)
        disease = diseases[0] if diseases else None
        return {"kind": "diagnosis", "intent": intent["name"],
                "confidence": intent["confidence"],
                "slots": {"Disease": disease}, "matched": disease is not None}
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_pipeline.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add app/nlu/pipeline.py app/tests/test_pipeline.py
git commit -m "feat(nlu): pipeline aggregating chitchat/intent/slot into NluResult"
```

---

### Task 9: Cypher 模板 kg/templates.py

**Files:**
- Create: `app/kg/templates.py`、`app/tests/test_templates.py`

**Interfaces:**
- Consumes: 仓库根 `config.semantic_slot`
- Produces:
  - `render(intent: str, slots: dict) -> list[str]`:把该意图的 `cql_template`(可能是 str 或 list)用 `slots` 渲染成 Cypher 列表;未知意图返回 `[]`
  - `reply_prefix(intent: str, slots: dict) -> str`:渲染 `reply_template`

- [ ] **Step 1: 写失败测试**

`app/tests/test_templates.py`:
```python
from app.kg.templates import render, reply_prefix


def test_render_single():
    cqls = render("定义", {"Disease": "高血压"})
    assert len(cqls) == 1
    assert "高血压" in cqls[0] and "RETURN p.desc" in cqls[0]


def test_render_list_intent():
    cqls = render("治疗方法", {"Disease": "感冒"})
    assert len(cqls) == 3 and all("感冒" in c for c in cqls)


def test_unknown_intent():
    assert render("不存在的意图", {"Disease": "x"}) == []


def test_reply_prefix():
    assert "高血压" in reply_prefix("定义", {"Disease": "高血压"})
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_templates.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 templates.py**

```python
# -*- coding: utf-8 -*-
import sys
import os

# 引入仓库根 config.semantic_slot
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import semantic_slot


def render(intent, slots):
    info = semantic_slot.get(intent)
    if not info or "cql_template" not in info:
        return []
    tpl = info["cql_template"]
    tpls = tpl if isinstance(tpl, list) else [tpl]
    return [t.format(**slots) for t in tpls]


def reply_prefix(intent, slots):
    info = semantic_slot.get(intent) or {}
    tpl = info.get("reply_template", "")
    try:
        return tpl.format(**slots)
    except Exception:
        return tpl
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_templates.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: 提交**

```bash
git add app/kg/templates.py app/tests/test_templates.py
git commit -m "feat(kg): Cypher template renderer reusing config.semantic_slot"
```

---

### Task 10: 路由 orchestrator/router.py

**Files:**
- Create: `app/orchestrator/__init__.py`(空)、`app/orchestrator/router.py`、`app/tests/test_router.py`

**Interfaces:**
- Consumes: `NluResult`、`Settings`
- Produces:
  - `route(nlu: dict, settings) -> str`:返回 `"chitchat"|"fast"|"slow"|"deny"`
    - `kind=="chitchat"` → `"chitchat"`
    - 诊断:`confidence>=accept 且 matched` → `"fast"`
    - `confidence<deny` 或 `intent is None` → `"slow"`(交 LLM 兜底,不直接拒)
    - 其余(中等置信 或 高置信但未命中实体)→ `"slow"`

- [ ] **Step 1: 写失败测试**

`app/tests/test_router.py`:
```python
from app.settings import Settings
from app.orchestrator.router import route

S = Settings(accept_threshold=0.8, deny_threshold=0.4)


def test_chitchat():
    assert route({"kind": "chitchat", "intent": "greet"}, S) == "chitchat"


def test_fast_high_conf_matched():
    nlu = {"kind": "diagnosis", "intent": "定义", "confidence": 0.9,
           "slots": {"Disease": "高血压"}, "matched": True}
    assert route(nlu, S) == "fast"


def test_slow_low_conf():
    nlu = {"kind": "diagnosis", "intent": None, "confidence": 0.1,
           "slots": {"Disease": None}, "matched": False}
    assert route(nlu, S) == "slow"


def test_slow_high_conf_no_entity():
    nlu = {"kind": "diagnosis", "intent": "定义", "confidence": 0.95,
           "slots": {"Disease": None}, "matched": False}
    assert route(nlu, S) == "slow"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_router.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 router.py**

```python
# -*- coding: utf-8 -*-
def route(nlu, settings):
    if nlu.get("kind") == "chitchat":
        return "chitchat"
    conf = nlu.get("confidence", 0.0)
    if nlu.get("intent") and conf >= settings.accept_threshold and nlu.get("matched"):
        return "fast"
    return "slow"
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_router.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: 提交**

```bash
git add app/orchestrator/__init__.py app/orchestrator/router.py app/tests/test_router.py
git commit -m "feat(orchestrator): confidence-based router"
```

---

### Task 11: 编排器快路 orchestrator/controller.py

**Files:**
- Create: `app/orchestrator/controller.py`、`app/tests/test_controller_fast.py`
- Modify: `app/api/main.py`(接入 controller)

**Interfaces:**
- Consumes: `NluPipeline`、`route`、`kg.templates`、`KGClient`
- Produces:
  - `class Controller`,`__init__(nlu, kg, settings, gossip=None)`
  - `handle(text, session_id) -> dict`:`{"answer": str, "path": str}`;本任务实现 `chitchat` 与 `fast` 两路(`slow` 暂返回占位 `path="slow"`)

- [ ] **Step 1: 写失败测试**(全 mock)

`app/tests/test_controller_fast.py`:
```python
from app.settings import Settings
from app.orchestrator.controller import Controller


class FakeNlu:
    def analyze(self, t):
        if t == "你好":
            return {"kind": "chitchat", "intent": "greet", "confidence": 1.0,
                    "slots": {"Disease": None}, "matched": False}
        return {"kind": "diagnosis", "intent": "定义", "confidence": 0.95,
                "slots": {"Disease": "高血压"}, "matched": True}


class FakeKG:
    available = True
    def query(self, cypher, params=None):
        return [{"p.desc": "一种慢性病"}]


def test_chitchat_path():
    c = Controller(FakeNlu(), FakeKG(), Settings())
    out = c.handle("你好", "u1")
    assert out["path"] == "chitchat" and out["answer"]


def test_fast_path():
    c = Controller(FakeNlu(), FakeKG(), Settings())
    out = c.handle("高血压是什么", "u1")
    assert out["path"] == "fast"
    assert "高血压" in out["answer"] and "慢性病" in out["answer"]
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_controller_fast.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 controller.py**

```python
# -*- coding: utf-8 -*-
import random
from app.orchestrator.router import route
from app.kg import templates

_GOSSIP = {
    "greet": ["你好,我是智能医疗助手小智,有什么可以帮您?"],
    "goodbye": ["再见,祝您健康。"],
    "isbot": ["我是医疗诊断助手小智。"],
    "deny": ["好的,那您可以换个问法试试。"],
}


def _flatten(rows):
    vals = []
    for r in rows:
        for v in r.values():
            vals.extend(v if isinstance(v, list) else [v])
    return [str(v) for v in vals if v is not None and str(v).strip()]


class Controller:
    def __init__(self, nlu, kg, settings, gossip=None):
        self.nlu = nlu
        self.kg = kg
        self.settings = settings
        self.gossip = gossip or _GOSSIP

    def handle(self, text, session_id):
        nlu = self.nlu.analyze(text)
        path = route(nlu, self.settings)
        if path == "chitchat":
            return {"answer": random.choice(self.gossip.get(nlu["intent"], ["在的"])), "path": "chitchat"}
        if path == "fast":
            return self._fast(nlu)
        return {"answer": "(慢路待接入)", "path": "slow"}

    def _fast(self, nlu):
        intent, slots = nlu["intent"], nlu["slots"]
        facts = []
        for cql in templates.render(intent, slots):
            facts += _flatten(self.kg.query(cql))
        if not facts:
            return {"answer": "唔~我装满知识的大脑此刻很贫瘠", "path": "fast"}
        return {"answer": templates.reply_prefix(intent, slots) + "、".join(facts), "path": "fast"}
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_controller_fast.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 接入 api/main.py**

把 `chat` 端点改为:
```python
from app.nlu.chitchat import Chitchat
from app.nlu.intent import IntentModel
from app.nlu.slot import SlotFiller
from app.nlu.pipeline import NluPipeline
from app.orchestrator.controller import Controller
import os

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_diseases = os.path.join(_root, "knowledge_extraction", "bilstm_crf", "checkpoint", "diseases.json")
_chit = Chitchat(os.path.join(_root, "nlu", "sklearn_Classification", "model_file"))
_intent = IntentModel(os.path.join(_root, "nlu", "bert_intent_recognition", "pytorch", "checkpoint"))
_slot = SlotFiller(_diseases) if os.path.exists(_diseases) else None
_nlu = NluPipeline(_chit, _intent, _slot) if _slot else None
_controller = Controller(_nlu, _kg, _settings) if _nlu else None


@app.post("/chat")
def chat(req: ChatIn):
    if _controller is None:
        return {"answer": "(NLU 资源缺失)", "path": "stub"}
    return _controller.handle(req.text, req.session_id)
```

- [ ] **Step 6: 跑 API 快路冒烟(mock 不可用时返回占位,不报错)**

Run: `python -m pytest app/tests/test_api.py -v`
Expected: PASS(/chat 不再 500;无 diseases.json 时返回占位)

- [ ] **Step 7: 提交**

```bash
git add app/orchestrator/controller.py app/tests/test_controller_fast.py app/api/main.py
git commit -m "feat(orchestrator): controller fast+chitchat paths, wired into API"
```

---

## 阶段三:会话记忆

### Task 12: 会话状态 memory/store.py

**Files:**
- Create: `app/memory/__init__.py`(空)、`app/memory/store.py`、`app/tests/test_memory.py`

**Interfaces:**
- Consumes: `Settings`
- Produces:
  - `class MemoryStore`,`__init__(settings)`;无 `redis_url` 或连接失败 → 内存字典
  - `get(session_id) -> dict`:默认 `{"slots": {}, "history": [], "last_intent": None}`
  - `set(session_id, state) -> None`

- [ ] **Step 1: 写失败测试**(内存降级路径)

`app/tests/test_memory.py`:
```python
from app.settings import Settings
from app.memory.store import MemoryStore


def test_inmemory_roundtrip():
    m = MemoryStore(Settings(redis_url=None))
    assert m.get("u1") == {"slots": {}, "history": [], "last_intent": None}
    m.set("u1", {"slots": {"Disease": "高血压"}, "history": ["hi"], "last_intent": "定义"})
    assert m.get("u1")["slots"]["Disease"] == "高血压"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_memory.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 store.py**

```python
# -*- coding: utf-8 -*-
import json

_DEFAULT = {"slots": {}, "history": [], "last_intent": None}


class MemoryStore:
    def __init__(self, settings):
        self._redis = None
        self._mem = {}
        if settings.redis_url:
            try:
                import redis
                self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    def get(self, session_id):
        if self._redis:
            raw = self._redis.get(f"kbqa:{session_id}")
            return json.loads(raw) if raw else dict(_DEFAULT)
        return self._mem.get(session_id, dict(_DEFAULT))

    def set(self, session_id, state):
        if self._redis:
            self._redis.set(f"kbqa:{session_id}", json.dumps(state, ensure_ascii=False))
        else:
            self._mem[session_id] = state
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_memory.py -v`
Expected: PASS(1 passed)

- [ ] **Step 5: 提交**

```bash
git add app/memory/__init__.py app/memory/store.py app/tests/test_memory.py
git commit -m "feat(memory): session store with Redis and in-memory fallback"
```

---

### Task 13: 槽位继承(controller 接入 memory)

**Files:**
- Modify: `app/orchestrator/controller.py`
- Test: `app/tests/test_controller_memory.py`

**Interfaces:**
- Consumes: `MemoryStore`
- Produces: `Controller.__init__(nlu, kg, settings, memory=None, gossip=None)`;诊断分支:槽位为空时从上轮 `slots` 继承,处理后回写 `slots/last_intent`

- [ ] **Step 1: 写失败测试**

`app/tests/test_controller_memory.py`:
```python
from app.settings import Settings
from app.orchestrator.controller import Controller
from app.memory.store import MemoryStore


class NluNoSlot:
    def __init__(self): self.first = True
    def analyze(self, t):
        # 第一轮带实体,第二轮不带(考察继承)
        d = "高血压" if self.first else None
        self.first = False
        return {"kind": "diagnosis", "intent": "定义", "confidence": 0.95,
                "slots": {"Disease": d}, "matched": d is not None}


class FakeKG:
    available = True
    def query(self, c, params=None): return [{"x": "慢性病"}]


def test_slot_inheritance():
    mem = MemoryStore(Settings(redis_url=None))
    c = Controller(NluNoSlot(), FakeKG(), Settings(), memory=mem)
    c.handle("高血压是什么", "u1")          # 第一轮:存入 Disease=高血压
    out = c.handle("那病因呢", "u1")         # 第二轮:无实体 → 继承
    assert "慢性病" in out["answer"]         # 仍能查到(继承成功 → 走到 fast)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_controller_memory.py -v`
Expected: FAIL(继承未实现,第二轮无实体走 slow 占位)

- [ ] **Step 3: 修改 controller.py**

`__init__` 增加 `memory=None`,存 `self.memory`。`handle` 改为:
```python
    def handle(self, text, session_id):
        state = self.memory.get(session_id) if self.memory else {"slots": {}, "last_intent": None}
        nlu = self.nlu.analyze(text)
        if nlu["kind"] == "diagnosis":
            # 槽位继承
            if not nlu["slots"].get("Disease") and state.get("slots", {}).get("Disease"):
                nlu["slots"]["Disease"] = state["slots"]["Disease"]
                nlu["matched"] = True
        path = route(nlu, self.settings)
        if path == "chitchat":
            ans = {"answer": __import__("random").choice(self.gossip.get(nlu["intent"], ["在的"])), "path": "chitchat"}
        elif path == "fast":
            ans = self._fast(nlu)
        else:
            ans = {"answer": "(慢路待接入)", "path": "slow"}
        if self.memory:
            new_state = {"slots": nlu["slots"], "last_intent": nlu.get("intent"),
                         "history": (state.get("history", []) + [text])[-10:]}
            self.memory.set(session_id, new_state)
        return ans
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_controller_memory.py app/tests/test_controller_fast.py -v`
Expected: PASS(全部)

- [ ] **Step 5: 接入 api/main.py**

在 `main.py` 构造 `MemoryStore` 并传入 Controller:
```python
from app.memory.store import MemoryStore
_memory = MemoryStore(_settings)
# Controller(... , memory=_memory)
```

- [ ] **Step 6: 提交**

```bash
git add app/orchestrator/controller.py app/tests/test_controller_memory.py app/api/main.py
git commit -m "feat(orchestrator): slot inheritance via memory store"
```

---

## 阶段四:慢路(LLM)

### Task 14: 慢路理解 llm/understand.py

**Files:**
- Create: `app/llm/understand.py`、`app/tests/test_understand.py`

**Interfaces:**
- Consumes: `LLMClient`
- Produces:
  - `understand(text: str, llm, intents: list[str]) -> dict`:`{"intent": str|None, "disease": str|None}`;LLM 不可用 → 全 None
  - 用 LLM 输出 JSON(让模型回 `{"intent","disease"}`,解析容错)

- [ ] **Step 1: 写失败测试**(mock LLM)

`app/tests/test_understand.py`:
```python
from app.llm.understand import understand


class FakeLLM:
    available = True
    def chat(self, messages, tools=None, stream=False):
        return {"content": '{"intent": "病因", "disease": "高血压"}', "tool_calls": None}


class DeadLLM:
    available = False
    def chat(self, *a, **k): raise RuntimeError


def test_parse_llm_json():
    out = understand("高血压为啥得", FakeLLM(), ["病因", "定义"])
    assert out["intent"] == "病因" and out["disease"] == "高血压"


def test_unavailable():
    out = understand("x", DeadLLM(), ["病因"])
    assert out == {"intent": None, "disease": None}
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_understand.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 understand.py**

```python
# -*- coding: utf-8 -*-
import json
import re


def understand(text, llm, intents):
    if not getattr(llm, "available", False):
        return {"intent": None, "disease": None}
    sys = ("你是医疗问答的语义解析器。从用户问句中抽取意图和疾病实体。"
           "意图只能从这个列表里选:" + "、".join(intents) + "。"
           '只输出 JSON:{"intent": <意图或null>, "disease": <疾病名或null>}')
    try:
        resp = llm.chat([{"role": "system", "content": sys},
                         {"role": "user", "content": text}])
        m = re.search(r"\{.*\}", resp["content"], re.S)
        data = json.loads(m.group(0)) if m else {}
        intent = data.get("intent")
        return {"intent": intent if intent in intents else None,
                "disease": data.get("disease")}
    except Exception:
        return {"intent": None, "disease": None}
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_understand.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add app/llm/understand.py app/tests/test_understand.py
git commit -m "feat(llm): slow-path intent/entity understanding"
```

---

### Task 15: text-to-Cypher kg/text2cypher.py

**Files:**
- Create: `app/kg/text2cypher.py`、`app/tests/test_text2cypher.py`

**Interfaces:**
- Consumes: `LLMClient`
- Produces:
  - `SCHEMA: str`(图谱 schema 描述常量:节点疾病/症状/药品…,关系 has_symptom 等)
  - `text_to_cypher(question: str, llm) -> str | None`:LLM 生成只读 Cypher;不可用或疑似写操作 → None
  - 安全:拒绝包含 `CREATE/DELETE/SET/MERGE/REMOVE` 的语句

- [ ] **Step 1: 写失败测试**

`app/tests/test_text2cypher.py`:
```python
from app.kg.text2cypher import text_to_cypher


class FakeLLM:
    available = True
    def __init__(self, c): self.c = c
    def chat(self, messages, tools=None, stream=False):
        return {"content": self.c, "tool_calls": None}


def test_returns_read_cypher():
    cql = text_to_cypher("高血压有什么症状",
                         FakeLLM("MATCH (p:疾病)-[:has_symptom]->(q) WHERE p.name='高血压' RETURN q.name"))
    assert cql.startswith("MATCH")


def test_rejects_write():
    assert text_to_cypher("删库", FakeLLM("MATCH (n) DELETE n")) is None
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_text2cypher.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 text2cypher.py**

```python
# -*- coding: utf-8 -*-
import re

SCHEMA = """节点(label): 疾病, 症状, 药品, 食物, 检查, 科室, 菜谱, 药企
关系: (疾病)-[:has_symptom]->(症状), (疾病)-[:acompany_with]->(疾病),
      (疾病)-[:recommand_drug]->(药品), (疾病)-[:need_check]->(检查),
      (疾病)-[:cure_department]->(科室), (疾病)-[:not_eat]->(食物)
疾病属性: name, desc, cause, prevent, easy_get, cure_way, cure_lasttime, cured_prob"""

_WRITE = re.compile(r"\b(CREATE|DELETE|SET|MERGE|REMOVE|DROP)\b", re.I)


def text_to_cypher(question, llm):
    if not getattr(llm, "available", False):
        return None
    sys = ("你是 Neo4j 查询生成器。根据图谱 schema 把用户问题转成**只读** Cypher。"
           "只输出一条 Cypher,不要解释。\nschema:\n" + SCHEMA)
    try:
        resp = llm.chat([{"role": "system", "content": sys},
                         {"role": "user", "content": question}])
        cql = resp["content"].strip().strip("`").replace("cypher\n", "").strip()
        if not cql or _WRITE.search(cql):
            return None
        return cql
    except Exception:
        return None
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_text2cypher.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add app/kg/text2cypher.py app/tests/test_text2cypher.py
git commit -m "feat(kg): LLM text-to-Cypher with write-operation guard"
```

---

### Task 16: 回答生成 llm/generate.py

**Files:**
- Create: `app/llm/generate.py`、`app/tests/test_generate.py`

**Interfaces:**
- Consumes: `LLMClient`
- Produces:
  - `generate(question: str, facts: list[str], llm) -> str`:基于事实生成回答;无事实 → 固定"未找到";LLM 不可用 → 回退把事实拼接返回

- [ ] **Step 1: 写失败测试**

`app/tests/test_generate.py`:
```python
from app.llm.generate import generate


class FakeLLM:
    available = True
    def chat(self, messages, tools=None, stream=False):
        return {"content": "根据资料,高血压是一种慢性病。", "tool_calls": None}


class DeadLLM:
    available = False


def test_with_llm():
    out = generate("高血压是什么", ["慢性病", "需长期管理"], FakeLLM())
    assert "高血压" in out


def test_no_facts():
    assert "未找到" in generate("x", [], FakeLLM())


def test_fallback_without_llm():
    out = generate("高血压是什么", ["慢性病"], DeadLLM())
    assert "慢性病" in out
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_generate.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 generate.py**

```python
# -*- coding: utf-8 -*-
def generate(question, facts, llm):
    if not facts:
        return "很抱歉,知识库中未找到相关信息。"
    if not getattr(llm, "available", False):
        return "、".join(facts)
    sys = ("你是医疗助手。**只能**基于给定事实回答,不得编造。"
           "用简洁中文回答用户问题。")
    ctx = "事实:\n- " + "\n- ".join(facts) + f"\n\n问题:{question}"
    try:
        resp = llm.chat([{"role": "system", "content": sys},
                         {"role": "user", "content": ctx}])
        return resp["content"].strip() or "、".join(facts)
    except Exception:
        return "、".join(facts)
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_generate.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add app/llm/generate.py app/tests/test_generate.py
git commit -m "feat(llm): fact-grounded answer generation with fallback"
```

---

### Task 17: 编排器慢路(controller 接入慢路)

**Files:**
- Modify: `app/orchestrator/controller.py`
- Test: `app/tests/test_controller_slow.py`

**Interfaces:**
- Consumes: `understand`、`text_to_cypher`、`generate`、`LLMClient`、意图列表(来自 `config.semantic_slot` 的键)
- Produces: `Controller.__init__(..., llm=None)`;`slow` 分支:`understand → render模板或text2cypher → kg.query → generate`

- [ ] **Step 1: 写失败测试**

`app/tests/test_controller_slow.py`:
```python
from app.settings import Settings
from app.orchestrator.controller import Controller


class NluSlow:
    def analyze(self, t):
        return {"kind": "diagnosis", "intent": None, "confidence": 0.1,
                "slots": {"Disease": None}, "matched": False}


class FakeKG:
    available = True
    def query(self, c, params=None): return [{"x": "头晕"}]


class FakeLLM:
    available = True
    def chat(self, messages, tools=None, stream=False):
        # understand 阶段返回 JSON;generate 阶段返回散文;用 system 内容区分
        sys = messages[0]["content"]
        if "语义解析" in sys:
            return {"content": '{"intent":"临床表现(病症表现)","disease":"高血压"}', "tool_calls": None}
        if "查询生成器" in sys:
            return {"content": "MATCH (p:疾病)-[:has_symptom]->(q) WHERE p.name='高血压' RETURN q.name", "tool_calls": None}
        return {"content": "高血压常见症状包括头晕。", "tool_calls": None}


def test_slow_path_end_to_end():
    c = Controller(NluSlow(), FakeKG(), Settings(), llm=FakeLLM())
    out = c.handle("高血压有啥不舒服", "u1")
    assert out["path"] == "slow"
    assert "头晕" in out["answer"] or "高血压" in out["answer"]
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_controller_slow.py -v`
Expected: FAIL(慢路仍是占位)

- [ ] **Step 3: 修改 controller.py 慢路分支**

顶部加导入与意图列表:
```python
from app.llm import understand as _understand_mod
from app.llm.generate import generate as _generate
from app.kg.text2cypher import text_to_cypher as _t2c
from app.kg import templates as _templates
```
`__init__` 增加 `llm=None`,存 `self.llm`;意图列表:
```python
from config import semantic_slot as _semantic_slot
self._intents = [k for k in _semantic_slot.keys() if k != "unrecognized"]
```
把 `else: ans = {...slow占位}` 换成 `ans = self._slow(text)`,并新增:
```python
    def _slow(self, text):
        u = _understand_mod.understand(text, self.llm, self._intents) if self.llm else {"intent": None, "disease": None}
        facts = []
        # 有意图+疾病 → 优先模板;否则 text-to-Cypher
        if u["intent"] and u["disease"]:
            for cql in _templates.render(u["intent"], {"Disease": u["disease"]}):
                facts += _flatten(self.kg.query(cql))
        if not facts:
            cql = _t2c(text, self.llm) if self.llm else None
            if cql:
                facts += _flatten(self.kg.query(cql))
        return {"answer": _generate(text, facts, self.llm), "path": "slow"}
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_controller_slow.py app/tests/test_controller_fast.py app/tests/test_controller_memory.py -v`
Expected: PASS(全部)

- [ ] **Step 5: 接入 api/main.py**

构造 `LLMClient` 已存在(`_llm`);把它传入 Controller:`Controller(..., memory=_memory, llm=_llm)`。

- [ ] **Step 6: 提交**

```bash
git add app/orchestrator/controller.py app/tests/test_controller_slow.py app/api/main.py
git commit -m "feat(orchestrator): slow path (understand+t2c+generate)"
```

---

## 阶段五:安全护栏

### Task 18: 护栏 safety/guardrails.py

**Files:**
- Create: `app/safety/__init__.py`(空)、`app/safety/guardrails.py`、`app/tests/test_guardrails.py`

**Interfaces:**
- Produces:
  - `apply(answer: str, user_text: str) -> str`:
    - 命中"确诊/开药/剂量/我得了什么病"类诉求 → 在答案前置就医提示
    - 一律追加免责声明尾注
    - 简单 PII 脱敏(11 位手机号、18 位身份证 → 掩码)

- [ ] **Step 1: 写失败测试**

`app/tests/test_guardrails.py`:
```python
from app.safety.guardrails import apply


def test_disclaimer_appended():
    out = apply("高血压可以挂心内科。", "高血压挂什么科")
    assert "仅供参考" in out


def test_diagnosis_request_prefixed():
    out = apply("可能是感冒。", "我这症状是得了什么病")
    assert "及时就医" in out or "面诊" in out


def test_pii_masked():
    out = apply("联系13812345678", "x")
    assert "13812345678" not in out
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_guardrails.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 guardrails.py**

```python
# -*- coding: utf-8 -*-
import re

_DISCLAIMER = "\n\n(以上内容仅供参考,不能替代专业医生的诊断与治疗,如有不适请及时就医。)"
_DIAG_PAT = re.compile(r"(我.*(得了|患了|是什么病))|(确诊)|(开.*药)|(吃.*剂量)|(我的症状)")
_PHONE = re.compile(r"\b1[3-9]\d{9}\b")
_IDCARD = re.compile(r"\b\d{17}[\dXx]\b")


def apply(answer, user_text):
    out = answer
    if _DIAG_PAT.search(user_text or ""):
        out = "请注意:我无法替代医生做出诊断,建议您及时就医面诊。\n" + out
    out = _PHONE.sub("***********", out)
    out = _IDCARD.sub("******************", out)
    return out + _DISCLAIMER
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_guardrails.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add app/safety/__init__.py app/safety/guardrails.py app/tests/test_guardrails.py
git commit -m "feat(safety): guardrails (disclaimer/diagnosis-warning/PII)"
```

---

### Task 19: controller 出口接入护栏

**Files:**
- Modify: `app/orchestrator/controller.py`
- Test: `app/tests/test_controller_guardrails.py`

**Interfaces:**
- Produces: `handle` 返回前对**诊断类**答案统一 `guardrails.apply(answer, text)`(闲聊不加免责)

- [ ] **Step 1: 写失败测试**

`app/tests/test_controller_guardrails.py`:
```python
from app.settings import Settings
from app.orchestrator.controller import Controller


class FakeNlu:
    def analyze(self, t):
        return {"kind": "diagnosis", "intent": "所属科室", "confidence": 0.95,
                "slots": {"Disease": "高血压"}, "matched": True}


class FakeKG:
    available = True
    def query(self, c, params=None): return [{"x": "心内科"}]


def test_guardrail_on_diagnosis():
    c = Controller(FakeNlu(), FakeKG(), Settings())
    out = c.handle("高血压挂什么科", "u1")
    assert "仅供参考" in out["answer"]
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_controller_guardrails.py -v`
Expected: FAIL(尚未加免责)

- [ ] **Step 3: 修改 controller.py**

顶部 `from app.safety import guardrails as _guard`。`handle` 在返回前:
```python
        if nlu["kind"] == "diagnosis":
            ans["answer"] = _guard.apply(ans["answer"], text)
```
(放在 memory.set 之前或之后均可;闲聊分支不过护栏)

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_controller_guardrails.py app/tests/test_controller_fast.py app/tests/test_controller_slow.py app/tests/test_controller_memory.py -v`
Expected: PASS(全部;注意 memory 测试断言"慢性病"仍成立——免责是追加,不影响 substring)

- [ ] **Step 5: 提交**

```bash
git add app/orchestrator/controller.py app/tests/test_controller_guardrails.py
git commit -m "feat(orchestrator): apply guardrails to diagnosis answers"
```

---

## 阶段六:缓存、归一化、收尾

### Task 20: 语义缓存 cache/semantic_cache.py

**Files:**
- Create: `app/cache/__init__.py`(空)、`app/cache/semantic_cache.py`、`app/tests/test_cache.py`

**Interfaces:**
- Produces:
  - `class SemanticCache`,`__init__(threshold=0.95)`(无 embedding 依赖,降级为精确匹配)
  - `lookup(question: str) -> str | None`
  - `save(question: str, answer: str) -> None`

- [ ] **Step 1: 写失败测试**

`app/tests/test_cache.py`:
```python
from app.cache.semantic_cache import SemanticCache


def test_exact_hit():
    c = SemanticCache()
    assert c.lookup("高血压怎么治") is None
    c.save("高血压怎么治", "答案A")
    assert c.lookup("高血压怎么治") == "答案A"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_cache.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 semantic_cache.py**

```python
# -*- coding: utf-8 -*-
class SemanticCache:
    """精确匹配缓存;预留 embedding 相似(无 embedding 依赖时退化为精确)。"""
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self._store = {}

    def lookup(self, question):
        return self._store.get(question.strip())

    def save(self, question, answer):
        self._store[question.strip()] = answer
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_cache.py -v`
Expected: PASS(1 passed)

- [ ] **Step 5: 接入 controller(可选先于缓存层)**

`Controller.__init__` 增加 `cache=None`;`handle` 开头:
```python
        if self.cache:
            hit = self.cache.lookup(text)
            if hit is not None:
                return {"answer": hit, "path": "cache"}
```
返回前(诊断类、护栏之后)`self.cache.save(text, ans["answer"])`。

- [ ] **Step 6: 提交**

```bash
git add app/cache/__init__.py app/cache/semantic_cache.py app/tests/test_cache.py app/orchestrator/controller.py
git commit -m "feat(cache): exact-match semantic cache wired into controller"
```

---

### Task 21: 实体归一化 nlu/normalize.py

**Files:**
- Create: `app/nlu/normalize.py`、`app/tests/test_normalize.py`

**Interfaces:**
- Consumes: 疾病词典(`diseases.json`)
- Produces:
  - `class Normalizer`,`__init__(diseases_path)`
  - `normalize(mention: str) -> str | None`:精确命中返回标准名;否则 None(预留 embedding 近邻)

- [ ] **Step 1: 写失败测试**

`app/tests/test_normalize.py`:
```python
import json
from app.nlu.normalize import Normalizer


def test_exact(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps(["高血压", "感冒"], ensure_ascii=False), encoding="utf-8")
    n = Normalizer(str(p))
    assert n.normalize("高血压") == "高血压"
    assert n.normalize("不存在的病") is None
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest app/tests/test_normalize.py -v`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现 normalize.py**

```python
# -*- coding: utf-8 -*-
import json


class Normalizer:
    def __init__(self, diseases_path):
        self.names = set(json.load(open(diseases_path, encoding="utf-8")))

    def normalize(self, mention):
        if mention in self.names:
            return mention
        return None
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest app/tests/test_normalize.py -v`
Expected: PASS(1 passed)

- [ ] **Step 5: 提交**

```bash
git add app/nlu/normalize.py app/tests/test_normalize.py
git commit -m "feat(nlu): entity normalizer (exact match, embedding-ready)"
```

---

### Task 22: 收尾(全量测试 + README + 清理 itchat)

**Files:**
- Create: `app/README.md`
- Delete: `itchat_app.py`(已被 REST API 取代)
- Test: 全量 `python -m pytest app/tests/ -v`

- [ ] **Step 1: 全量测试**

Run: `python -m pytest app/tests/ -v`
Expected: 全部 PASS(逐文件运行;若两同名 test 冲突,加 `-p no:cacheprovider` 或分目录运行)

- [ ] **Step 2: 写 app/README.md**

```markdown
# KBQA 服务(LLM 级联架构)

## 启动
pip install -r requirements-service.txt
# 配置(可选,缺失则降级):
export KBQA_LLM_API_KEY=sk-xxx
export KBQA_LLM_BASE_URL=https://api.openai.com/v1
export KBQA_LLM_MODEL=gpt-4o-mini
export KBQA_NEO4J_PASSWORD=123456
export KBQA_REDIS_URL=redis://127.0.0.1:6379/0
uvicorn app.api.main:app --host 0.0.0.0 --port 8000

## 接口
GET  /health   各依赖可用状态
POST /chat     {"text": "...", "session_id": "u1"} -> {"answer","path"}

## 架构
快路(小模型,本地推理)守低延迟;慢路(LLM)接长尾 + 生成;Neo4j 事实接地;
缺依赖时优雅降级(无 LLM/Redis/Neo4j/权重均不崩溃)。
详见 docs/superpowers/specs/2026-06-28-llm-cascade-architecture-design.md
```

- [ ] **Step 3: 删除 itchat_app.py**

```bash
git rm itchat_app.py
```

- [ ] **Step 4: 提交**

```bash
git add app/README.md
git commit -m "docs+chore: app README; remove itchat_app (replaced by REST API)"
```

---

## 自检结果

- **Spec 覆盖**:settings(T1)、llm/client(T2)、neo4j(T3)、api(T4,T11,T13,T17)、nlu(T5-T8,T21)、templates(T9)、router(T10)、controller(T11,T13,T17,T19,T20)、memory(T12)、慢路 understand/t2c/generate(T14-T16)、safety(T18-T19)、cache(T20)、降级策略(各 Task 均含)、删 itchat(T22)。全覆盖。
- **占位符**:无 TODO/TBD;每个代码步骤均有完整代码。
- **类型一致**:`NluResult` 五键在 T8/T10/T11/T13/T17 一致;`route` 返回值集合一致;`Controller.__init__` 形参在 T11→T13→T17→T19→T20 渐进扩展(每次显式列出);`llm.chat` 返回 `{"content","tool_calls"}` 在 T2/T14/T15/T16 一致。
- **已知前置**:意图模型需训练权重(无则降级走慢路);LLM 需 key(无则慢路退化);Neo4j/Redis 需启动(无则降级)。均不阻塞单测。
