# -*- coding: utf-8 -*-
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


class NluDiagChatDiag:
    def __init__(self): self.calls = 0
    def analyze(self, t):
        self.calls += 1
        if self.calls == 1:
            return {"kind": "diagnosis", "intent": "定义", "confidence": 0.95,
                    "slots": {"Disease": "高血压"}, "matched": True}
        if self.calls == 2:
            return {"kind": "chitchat", "intent": "greet", "confidence": 1.0,
                    "slots": {"Disease": None}, "matched": False}
        return {"kind": "diagnosis", "intent": "病因", "confidence": 0.95,
                "slots": {"Disease": None}, "matched": False}


def test_slot_survives_chitchat_interleave():
    mem = MemoryStore(Settings(redis_url=None))
    c = Controller(NluDiagChatDiag(), FakeKG(), Settings(), memory=mem)
    c.handle("高血压是什么", "u9")   # store 高血压
    c.handle("谢谢", "u9")            # chitchat — must NOT erase
    out = c.handle("那病因呢", "u9")  # inherit 高血压 → fast → KG hit
    assert "慢性病" in out["answer"]
