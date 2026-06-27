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
