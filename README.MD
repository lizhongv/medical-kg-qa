# 基于知识图谱的医疗诊断问答系统(现代化版)

一个中文医疗领域的 **KBQA(知识库问答)** 系统。用户提问 → 识别意图与疾病实体 → 从 Neo4j 知识图谱查询事实 → 生成接地(grounded)的回答。

本版本已从原始 **TensorFlow1.x / bert4keras / itchat** 老栈,全面现代化为:

- **训练侧**:全部 PyTorch / scikit-learn(摆脱装不上的 TF1.x)。
- **在线侧**:**「小模型快路 + 云端 LLM 慢路 + Neo4j 事实接地」的级联架构**,对外 REST API。
- **LLM**:OpenAI 兼容接口(已验证 DeepSeek `deepseek-chat`)。

> 原始项目说明见 [`README_original.MD`](./README_original.MD)。

---

## 一、整体架构

```
                          用户问句 (REST: POST /chat)
                                   │
                                   ▼
                   ┌──────────────────────────────────┐
                   │  对话编排器 Controller            │
                   │  (语义缓存 → NLU → 路由 → 取事实    │
                   │   → 生成 → 安全护栏 → 落上下文)     │
                   └───────────────┬──────────────────┘
                                   ▼
          ┌────────────────────────────────────────────────┐
          │  快速 NLU(本地小模型,~10–50ms,不调 LLM)       │
          │  闲聊分类(sklearn) · 意图(BERT,test_acc 0.89) │
          │  疾病槽位(AC自动机词典) · 实体归一化            │
          └───────────────┬────────────────────────────────┘
                          ▼  置信度路由
            ┌─────────────┴──────────────┐
        高置信&命中(快路)               低置信/未命中/复杂(慢路)
            ▼                                   ▼
   意图→参数化 Cypher                  LLM 理解意图+实体(understand)
   查 Neo4j → 模板回复                 → LLM text-to-Cypher / 模板
                                       → 查 Neo4j → LLM 基于事实生成
            └─────────────┬──────────────┘
                          ▼
              医疗安全护栏(免责声明 / 拒诊断处方 / PII 脱敏)
                          ▼
                     返回用户(可流式)

  事实来源: Neo4j 知识图谱(8808 疾病 + 症状/药品/检查/科室/食物关系)
  多轮记忆: Redis(可降级为进程内内存),支持槽位继承
```

**核心思想**:意图模型只在置信度高时走快路(快、省、确定);模糊问题落到 LLM 慢路兜底;**所有答案以 Neo4j 为唯一事实来源,防止 LLM 幻觉**。缺任一外部依赖(Neo4j/Redis/LLM/权重)时**优雅降级**而非崩溃。

---

## 二、技术栈

| 层面 | 技术 |
|---|---|
| 在线服务 | FastAPI + uvicorn,REST API |
| 意图识别 | BERT(`hfl/chinese-roberta-wwm-ext`)+ TextCNN(PyTorch) |
| 命名实体/槽位 | AC 自动机疾病词典(pyahocorasick)+ BiLSTM-CRF(PyTorch,备选) |
| 闲聊分类 | TF-IDF + LogisticRegression + GBDT(scikit-learn) |
| 大模型 | OpenAI 兼容客户端(DeepSeek / 任意兼容端点) |
| 知识图谱 | Neo4j(官方 `neo4j` 驱动) |
| 多轮记忆 | Redis(可降级内存) |

---

## 三、目录结构

```
.
├── app/                      在线服务(REST API)
│   ├── api/main.py             FastAPI:POST /chat、GET /health
│   ├── orchestrator/           controller.py(编排) + router.py(置信度路由)
│   ├── nlu/                    快速 NLU:chitchat/intent/slot/normalize/pipeline
│   ├── kg/                     neo4j_client / templates(意图→Cypher) / text2cypher
│   ├── llm/                    client / understand / generate
│   ├── memory/store.py         会话状态(Redis + 内存降级,槽位继承)
│   ├── safety/guardrails.py    医疗安全护栏
│   ├── cache/semantic_cache.py 语义缓存
│   └── settings.py             统一配置(读环境变量)
│
├── config.py                 13类意图 → Cypher模板 + 回复模板(app 复用)
│
├── training/                 ⭐ 模型训练(每个模型:脚本+data+checkpoint 同目录)
│   ├── intent/                意图(BERT+TextCNN)
│   │   ├── train.py            训练脚本
│   │   ├── gen_data.py         合成数据生成(模板 + 可选 LLM 改写)
│   │   ├── data/               train.csv/test.csv + 清洗后 train_cleanLLM.csv/test_cleanLLM.csv
│   │   ├── checkpoint/         best_model.pt / label2id.json / meta.json
│   │   ├── requirements.txt / README.md   (基座默认 Hub 下载;离线可手动建 base_model/)
│   ├── ner/                   NER(BiLSTM-CRF)
│   │   ├── train.py
│   │   ├── data/               cMedQANER(首次训练自动下载)
│   │   ├── checkpoint/         best_ner.pt / vocab.json / tag2id.json
│   │   └── requirements.txt / README.md
│   └── chitchat/              闲聊(sklearn)
│       ├── train.py
│       ├── data/               intent_recog_data.txt
│       ├── model/              vec/LR/gbdt/id2label.pkl
│       └── README.md
│
├── kg/                       知识图谱
│   ├── import_kg.py            导入 Neo4j(官方驱动,免 py2neo)
│   ├── medical.json            源数据(8808 疾病,~47MB,不入库)
│   └── diseases.json           疾病词典(app 槽位用)
│
├── run_service.bat           启动服务(含 LLM/Neo4j 配置;gitignore,含密钥勿提交)
├── test_chat.py              命令行测试 /chat
├── requirements.txt          在线服务依赖
├── README.MD / README_original.MD
└── docs/                     设计 spec + 实现计划
```

---

## 四、快速开始(部署运行)

### 前置
- Python 3.10+,Java 17(Neo4j 需要)
- `pip install -r requirements.txt`

### 第 1 步:启动 Neo4j
下载 Neo4j Community,设密码并启动:
```cmd
<neo4j>\bin\neo4j-admin dbms set-initial-password 12345678
<neo4j>\bin\neo4j console
```
浏览器 http://localhost:7474 能登录即成功。

### 第 2 步:导入知识图谱
```bash
cd kg
# PowerShell 设密码: $env:KBQA_NEO4J_PASSWORD="12345678"
python import_kg.py        # 8808 疾病 + 症状/药品/科室/检查/食物关系
```

### 第 3 步:配置并启动服务
编辑 `run_service.bat` 填入 LLM key / Neo4j 密码,然后:
```cmd
run_service.bat            :: 起 uvicorn,端口 8000
```
关键环境变量(`app/settings.py` 读取):
| 变量 | 说明 |
|---|---|
| `KBQA_LLM_API_KEY` | LLM 密钥(如 DeepSeek) |
| `KBQA_LLM_BASE_URL` | 如 `https://api.deepseek.com` |
| `KBQA_LLM_MODEL` | 如 `deepseek-chat` |
| `KBQA_NEO4J_URI/USER/PASSWORD` | Neo4j 连接 |
| `KBQA_REDIS_URL` | 可选,不设则用内存 |

### 第 4 步:测试
```bash
python test_chat.py                    # 跑一组示例问题
python test_chat.py "高血压怎么治"     # 单问
# 或浏览器交互页: http://localhost:8000/docs
```
预期:`你好`→闲聊;医疗问句→`fast`(意图命中)或`slow`(LLM兜底),答案来自 Neo4j + 免责声明。

---

## 五、模型训练(三个,互相独立)

> 训练在有 GPU、可联网拉基座的机器上进行。产物落各自 `checkpoint/` / `model/`,在线服务自动加载。

### ① 意图识别(BERT + TextCNN)
```bash
cd training/intent
pip install -r requirements.txt
python train.py --train_file data/train_cleanLLM.csv \
                --test_file  data/test_cleanLLM.csv --class_weight
# 产物: checkpoint/best_model.pt, label2id.json, meta.json
```
- 默认基座 `hfl/chinese-roberta-wwm-ext`(12层);`--base hfl/rbt3` 可换 3 层小模型。
- `meta.json` 记录基座,推理侧据此对齐结构加载。
- **数据质量是关键**:原始标注约 40% 有误,经 **LLM 逐条清洗**后 test_acc 从 0.68 → **0.89**。`gen_data.py` 可生成模板合成数据辅助。

### ② NER(BiLSTM-CRF,字符级)
```bash
cd training/ner
pip install -r requirements.txt        # torch + pytorch-crf
python train.py                        # 首次自动下载 cMedQANER(公开基准,免清洗)
# 产物: checkpoint/best_ner.pt, vocab.json, tag2id.json
```
> 在线服务的疾病槽位实际走 **AC自动机词典**(`kg/diseases.json`),稳定可靠;BiLSTM-CRF 为备选增强。

### ③ 闲聊分类(sklearn)
```bash
cd training/chitchat
python train.py                        # 几秒;产物 model/*.pkl
```

---

## 六、API

| 端点 | 说明 |
|---|---|
| `GET /health` | 依赖状态:`{"status":"ok","deps":{"neo4j":bool,"llm":bool}}` |
| `POST /chat` | 入参 `{"text": "...", "session_id": "..."}`;出参 `{"answer": "...", "path": "chitchat\|fast\|slow\|cache"}` |

---

## 七、13 类意图

定义、病因、预防、临床表现(病症表现)、相关病症、治疗方法、所属科室、传染性、治愈率、禁忌、化验/体检方案、治疗时间、其他。
每类映射到 `config.py:semantic_slot` 的 Cypher 模板与回复模板。

---

## 八、开发文档
设计与实现过程见 `docs/superpowers/specs/` 与 `docs/superpowers/plans/`。
