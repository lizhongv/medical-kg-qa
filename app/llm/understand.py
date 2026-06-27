# -*- coding: utf-8 -*-
import json
import re


def understand(text, llm, intents):
    if not getattr(llm, "available", False):
        return {"intent": None, "disease": None}
    system_prompt = ("你是医疗问答的语义解析器。从用户问句中抽取意图和疾病实体。"
                     "意图只能从这个列表里选:" + "、".join(intents) + "。"
                     '只输出 JSON:{"intent": <意图或null>, "disease": <疾病名或null>}')
    try:
        resp = llm.chat([{"role": "system", "content": system_prompt},
                         {"role": "user", "content": text}])
        m = re.search(r"\{.*\}", resp["content"], re.S)
        data = json.loads(m.group(0)) if m else {}
        intent = data.get("intent")
        return {"intent": intent if intent in intents else None,
                "disease": data.get("disease")}
    except Exception:
        return {"intent": None, "disease": None}
