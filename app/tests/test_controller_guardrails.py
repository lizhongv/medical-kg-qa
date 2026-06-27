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
