# -*- coding: utf-8 -*-
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import semantic_slot as _semantic_slot
from app.orchestrator.router import route
from app.kg import templates
from app.llm import understand as _understand_mod
from app.llm.generate import generate as _generate
from app.kg.text2cypher import text_to_cypher as _t2c
from app.kg import templates as _templates
from app.safety import guardrails as _guard

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
    def __init__(self, nlu, kg, settings, gossip=None, memory=None, llm=None):
        self.nlu = nlu
        self.kg = kg
        self.settings = settings
        self.gossip = gossip or _GOSSIP
        self.memory = memory
        self.llm = llm
        self._intents = [k for k in _semantic_slot.keys() if k != "unrecognized"]

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
            ans = {"answer": random.choice(self.gossip.get(nlu["intent"], ["在的"])), "path": "chitchat"}
        elif path == "fast":
            ans = self._fast(nlu)
        else:
            ans = self._slow(text)
        if nlu["kind"] == "diagnosis":
            ans["answer"] = _guard.apply(ans["answer"], text)
        if self.memory:
            merged_slots = {**state.get("slots", {}),
                            **{k: v for k, v in nlu["slots"].items() if v is not None}}
            new_state = {"slots": merged_slots,
                         "last_intent": nlu.get("intent") or state.get("last_intent"),
                         "history": (state.get("history", []) + [text])[-10:]}
            self.memory.set(session_id, new_state)
        return ans

    def _fast(self, nlu):
        intent, slots = nlu["intent"], nlu["slots"]
        facts = []
        for cql in templates.render(intent, slots):
            facts += _flatten(self.kg.query(cql))
        if not facts:
            return {"answer": "唔~我装满知识的大脑此刻很贫瘠", "path": "fast"}
        return {"answer": templates.reply_prefix(intent, slots) + "、".join(facts), "path": "fast"}

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
