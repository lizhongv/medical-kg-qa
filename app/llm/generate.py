# -*- coding: utf-8 -*-
def generate(question, facts, llm):
    if not facts:
        return "很抱歉,知识库中未找到相关信息。"
    if not getattr(llm, "available", False):
        return "、".join(facts)
    system_prompt = ("你是医疗助手。**只能**基于给定事实回答,不得编造。"
                     "用简洁中文回答用户问题。")
    ctx = "事实:\n- " + "\n- ".join(facts) + f"\n\n问题:{question}"
    try:
        resp = llm.chat([{"role": "system", "content": system_prompt},
                         {"role": "user", "content": ctx}])
        return resp["content"].strip() or "、".join(facts)
    except Exception:
        return "、".join(facts)
