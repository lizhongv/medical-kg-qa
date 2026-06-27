# -*- coding: utf-8 -*-
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


def test_slow_path_llm_none():
    from app.settings import Settings
    from app.orchestrator.controller import Controller
    class NluSlow2:
        def analyze(self, t):
            return {"kind": "diagnosis", "intent": None, "confidence": 0.1,
                    "slots": {"Disease": None}, "matched": False}
    class KG2:
        available = True
        def query(self, c, params=None): return []
    c = Controller(NluSlow2(), KG2(), Settings(), llm=None)
    out = c.handle("随便问问", "u_none")
    assert out["path"] == "slow" and isinstance(out["answer"], str)
