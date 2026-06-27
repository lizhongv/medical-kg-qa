# -*- coding: utf-8 -*-
import random
from app.orchestrator.router import route
from app.kg import templates

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
    def __init__(self, nlu, kg, settings, gossip=None):
        self.nlu = nlu
        self.kg = kg
        self.settings = settings
        self.gossip = gossip or _GOSSIP

    def handle(self, text, session_id):
        nlu = self.nlu.analyze(text)
        path = route(nlu, self.settings)
        if path == "chitchat":
            return {"answer": random.choice(self.gossip.get(nlu["intent"], ["在的"])), "path": "chitchat"}
        if path == "fast":
            return self._fast(nlu)
        return {"answer": "(慢路待接入)", "path": "slow"}

    def _fast(self, nlu):
        intent, slots = nlu["intent"], nlu["slots"]
        facts = []
        for cql in templates.render(intent, slots):
            facts += _flatten(self.kg.query(cql))
        if not facts:
            return {"answer": "唔~我装满知识的大脑此刻很贫瘠", "path": "fast"}
        return {"answer": templates.reply_prefix(intent, slots) + "、".join(facts), "path": "fast"}
