# -*- coding: utf-8 -*-
import re

SCHEMA = """节点(label): 疾病, 症状, 药品, 食物, 检查, 科室, 菜谱, 药企
关系: (疾病)-[:has_symptom]->(症状), (疾病)-[:acompany_with]->(疾病),
      (疾病)-[:recommand_drug]->(药品), (疾病)-[:need_check]->(检查),
      (疾病)-[:cure_department]->(科室), (疾病)-[:not_eat]->(食物)
疾病属性: name, desc, cause, prevent, easy_get, cure_way, cure_lasttime, cured_prob"""

_WRITE = re.compile(r"\b(CREATE|DELETE|SET|MERGE|REMOVE|DROP)\b", re.I)


def text_to_cypher(question, llm):
    if not getattr(llm, "available", False):
        return None
    system_prompt = ("你是 Neo4j 查询生成器。根据图谱 schema 把用户问题转成**只读** Cypher。"
                     "只输出一条 Cypher,不要解释。\nschema:\n" + SCHEMA)
    try:
        resp = llm.chat([{"role": "system", "content": system_prompt},
                         {"role": "user", "content": question}])
        cql = resp["content"].strip().strip("`").replace("cypher\n", "").strip()
        if not cql or _WRITE.search(cql):
            return None
        return cql
    except Exception:
        return None
