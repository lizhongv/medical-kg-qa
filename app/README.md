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
