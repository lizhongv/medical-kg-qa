# -*- coding: utf-8 -*-
"""
把 medical.json 导入 Neo4j(用官方 neo4j 驱动,无需 py2neo)。
节点:疾病/症状/药品/检查/科室/食物;关系匹配 config.semantic_slot 的 Cypher。

前置:Neo4j 已启动;pip install neo4j
运行:python import_kg.py
连接:bolt://127.0.0.1:7687,用户 neo4j,密码取环境变量 KBQA_NEO4J_PASSWORD(默认 123456)
数据:./medical.json
"""
import os
import json

from neo4j import GraphDatabase

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "medical.json")
URI = os.environ.get("KBQA_NEO4J_URI", "bolt://127.0.0.1:7687")
USER = os.environ.get("KBQA_NEO4J_USER", "neo4j")
PWD = os.environ.get("KBQA_NEO4J_PASSWORD", "123456")

# 疾病属性字段(medical.json -> 节点属性)
PROPS = ["desc", "cause", "prevent", "cure_way", "cure_lasttime", "cured_prob"]
# 关系:(字段, 关系名, 尾节点label)
RELS = [
    ("symptom", "has_symptom", "症状"),
    ("acompany", "acompany_with", "疾病"),
    ("recommand_drug", "recommand_drug", "药品"),
    ("check", "need_check", "检查"),
    ("cure_department", "cure_department", "科室"),
    ("not_eat", "not_eat", "食物"),
]


def load():
    rows = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    driver = GraphDatabase.driver(URI, auth=(USER, PWD))
    rows = load()
    print(f"读入 {len(rows)} 条疾病,开始导入 {URI} ...")
    with driver.session() as s:
        # 索引(加速 MERGE)
        for label in ["疾病", "症状", "药品", "检查", "科室", "食物"]:
            s.run(f"CREATE INDEX IF NOT EXISTS FOR (n:`{label}`) ON (n.name)")

        for i, rec in enumerate(rows, 1):
            name = rec.get("name")
            if not name:
                continue
            props = {p: ("" if rec.get(p) is None else str(rec.get(p))) for p in PROPS}
            props["easy_get"] = str(rec.get("easy_get") or rec.get("get_way") or "")
            s.run(
                "MERGE (d:`疾病` {name:$name}) SET d += $props",
                name=name, props=props,
            )
            for field, rel, tail in RELS:
                vals = rec.get(field) or []
                if isinstance(vals, str):
                    vals = [vals]
                vals = [str(v) for v in vals if v]
                if not vals:
                    continue
                s.run(
                    f"MATCH (d:`疾病` {{name:$name}}) "
                    f"UNWIND $vals AS v "
                    f"MERGE (t:`{tail}` {{name:v}}) "
                    f"MERGE (d)-[:`{rel}`]->(t)",
                    name=name, vals=vals,
                )
            if i % 500 == 0:
                print(f"  {i}/{len(rows)}")
    driver.close()
    print("导入完成。")


if __name__ == "__main__":
    main()
