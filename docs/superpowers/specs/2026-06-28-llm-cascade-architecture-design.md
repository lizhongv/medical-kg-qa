# 设计文档:LLM 级联对话架构(医疗 KBQA 现代化)

- 日期:2026-06-28
- 目标:把原 itchat 流水线重构为「小模型快路 + 云端 LLM 慢路 + Neo4j 事实接地」的级联架构,对外提供 REST API,含生产件(Redis 多轮、医疗安全护栏、语义缓存)。

## 1. 背景与决策(已与用户确认)

- LLM 一律走 **OpenAI 兼容 API**(云端,先验证;后续可换本地),`base_url/key/model` 可配。
- **快路只用本地小模型**(sklearn 闲聊 + PyTorch 意图/NER + 词典 + embedding 归一化),不碰 LLM;LLM 仅在慢路(理解兜底 / text-to-Cypher / 回答生成)。
- **Neo4j 知识图谱是唯一事实来源**,LLM 回答必须基于检索到的事实(grounding)。
- 范围:**全量含生产件**;对外 **REST API**。
- 缺外部依赖时各模块**优雅降级**(无 Redis→内存;无 LLM key→慢路退模板;无 PyTorch 权重→词典+规则兜底)。

## 2. 全局约束(每个任务都隐含遵守)

- 快路 NLU 不得调用 LLM 或任何网络服务;只做本地权重前向推理。
- LLM 访问只能经由 `llm/client.py` 的统一 OpenAI 兼容客户端;不在别处直接发 HTTP。
- 所有外部连接(Neo4j/Redis/LLM)集中在 `settings.py` 配置,密钥走环境变量,代码不硬编码。
- LLM 慢路回答只能基于 `kg` 返回的事实;不得让 LLM 自由编造医疗结论。
- 每个模块可独立单测;缺外部依赖时降级而非崩溃。
- 不破坏离线侧既有资产(`build_kg_utils.py`、三个 `pytorch/` 训练包、`config.semantic_slot`)。
- Python 3.10+;不引入 tensorflow/keras/bert4keras。

## 3. 模块分解与职责

```
api/
  main.py            FastAPI;POST /chat(支持 SSE 流式);GET /health
orchestrator/
  controller.py      主流程编排:缓存→NLU→路由→检索→生成→护栏→落上下文
  router.py          路由决策:置信度阈值 + 是否命中标准实体 → "fast"|"slow"
nlu/
  chitchat.py        sklearn 闲聊分类(加载 model_file/*.pkl)
  intent.py          PyTorch 意图模型(加载 best_model.pt);无权重→降级标记
  slot.py            NER:PyTorch token 分类 + AC 自动机疾病词典(双召回)
  normalize.py       实体归一化:embedding 检索 mention→KG 标准疾病ID
kg/
  neo4j_client.py    Neo4j 连接 + Cypher 执行(改造自 modules.neo4j_searcher)
  templates.py       意图→参数化 Cypher(快路;复用 config.semantic_slot)
  text2cypher.py     慢路:LLM 生成 Cypher / GraphRAG 检索
llm/
  client.py          OpenAI 兼容客户端:chat(messages, tools, stream),超时/重试
  understand.py      慢路理解:意图+实体抽取(function calling / 结构化输出)
  generate.py        基于事实的回答生成(流式)
memory/
  store.py           会话状态:Redis(降级内存字典);槽位/历史/上轮意图
safety/
  guardrails.py      免责声明 + 拒诊断/处方 + PII 过滤 + 慢路事实校验
cache/
  semantic_cache.py  语义缓存:embedding 相似命中(降级精确匹配)
settings.py          统一配置:阈值/端点/密钥/连接;敏感项环境变量
```

## 4. 关键接口契约

```python
# nlu —— 统一输出
NluResult = {
  "kind": "chitchat" | "diagnosis",
  "intent": str,
  "confidence": float,
  "slots": {"Disease": str | None},
  "matched": bool,            # 槽位是否归一化到 KG 标准实体
}

# router
route(nlu: NluResult, thresholds) -> "fast" | "slow" | "chitchat" | "deny"

# llm/client —— OpenAI 兼容
chat(messages: list, tools: list | None = None, stream: bool = False) -> dict | Iterator

# kg
query(cypher: str, params: dict) -> list[dict]
render_template(intent: str, slots: dict) -> (cypher: str, params: dict)   # 快路
text_to_cypher(question: str, schema: str) -> (cypher: str, params: dict)  # 慢路

# memory
get(session_id) -> {"slots": dict, "history": list, "last_intent": str | None}
set(session_id, state) -> None

# safety
apply(answer: str, facts: list, context) -> str          # 加免责/拒答/PII/事实校验

# cache
lookup(question) -> str | None
save(question, answer) -> None
```

## 5. 数据流

**快路**(`router → "fast"`,高置信 & 命中标准疾病):
```
controller: cache.lookup 未命中 → nlu(chitchat/intent/slot/normalize)
          → router="fast" → kg.render_template → kg.query(Neo4j)
          → 模板回复 → safety.apply → cache.save + memory.set → 返回
```
**慢路**(`router → "slow"`,低置信/未命中/复杂):
```
controller → llm.understand(重抽意图+实体) → kg.text_to_cypher → kg.query
          → llm.generate(基于事实流式) → safety.apply → memory.set → SSE 返回
```
**闲聊/拒答**:`router="chitchat"`→话术;`"deny"`→拒答话术。
全程读写 `memory`(槽位继承);出口统一过 `safety`。

## 6. 外部依赖与降级策略

| 依赖 | 缺失时降级 |
|---|---|
| Neo4j | 返回"知识库不可用"友好提示;`/health` 标红 |
| PyTorch 意图/NER 权重 | `nlu` 标记 model_unavailable → 槽位走词典、意图走慢路 LLM 兜底 |
| sklearn 闲聊 pkl | 本机可生成(`train_modern.py`),缺失则闲聊判定退化为规则关键词 |
| LLM API key | 慢路退化为"模板兜底 + 无法回答"提示;快路不受影响 |
| Redis | 退化为进程内内存字典(单实例开发用) |
| embedding 模型(归一化/缓存) | 退化为精确字符串匹配 |

## 7. 复用 / 删除

- 复用:Neo4j 图谱、`config.semantic_slot`(→templates)、`modules.neo4j_searcher`(→neo4j_client)、三个已迁移小模型、词典 NER、`build_kg_utils.py`。
- 删除:`itchat_app.py`(换 REST API);`modules.py` 拆分进新模块后移除其在线职责。
- 新增:orchestrator / llm / memory / safety / cache / api / normalize。

## 8. 分阶段构建顺序

1. **骨架**:`settings.py` + `llm/client.py` + `kg/neo4j_client.py` + `api/main.py`(`/chat` 空跑 + `/health`)。
2. **快路**:`nlu`(chitchat/intent/slot)+ `kg/templates.py` + `orchestrator/router.py` + `controller`(快路打通)。
3. **会话**:`memory/store.py`(Redis+内存降级)+ 槽位继承多轮。
4. **慢路**:`llm/understand.py` + `kg/text2cypher.py` + `llm/generate.py`(流式)。
5. **安全**:`safety/guardrails.py`(免责/拒诊断/PII/事实校验)接入出口。
6. **缓存 + 归一化 + 收尾**:`cache/semantic_cache.py`、`nlu/normalize.py`、日志/监控。

## 9. 技术栈

FastAPI · uvicorn · PyTorch/transformers · scikit-learn · `openai` SDK(兼容客户端)· neo4j 官方驱动(或 py2neo 升级)· redis-py · pyahocorasick · pytest · httpx(测试)。

## 10. 测试策略

- 每模块 pytest 单测,外部依赖用 mock/stub(LLM 客户端 mock、Neo4j 用内存假数据或 testcontainer 可选)。
- `router`、`templates`、`guardrails`、`memory` 为纯逻辑,重点覆盖。
- `api` 用 FastAPI TestClient 跑 `/chat` 快路(NLU/KG 注入 mock)。
- 端到端冒烟在用户提供 Neo4j/权重/LLM key 后进行(文档说明)。

## 11. 验收标准

1. `uvicorn` 起服务,`/health` 返回各依赖状态;`/chat` 快路在 mock 依赖下返回结构化回答。
2. 路由按置信度/命中正确分流(单测覆盖三类边界)。
3. 慢路在 mock LLM 下走通 understand→text2cypher→generate 链路。
4. 安全护栏对"诊断/处方/PII"类输入正确拦截或加免责。
5. 缺任一外部依赖时服务不崩溃,按第 6 节降级。
6. 不残留 itchat / TF 依赖;离线资产不受损。
