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
