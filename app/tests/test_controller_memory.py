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
