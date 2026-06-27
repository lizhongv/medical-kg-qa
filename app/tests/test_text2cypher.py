from app.kg.text2cypher import text_to_cypher


class FakeLLM:
    available = True
    def __init__(self, c): self.c = c
    def chat(self, messages, tools=None, stream=False):
        return {"content": self.c, "tool_calls": None}


def test_returns_read_cypher():
    cql = text_to_cypher("高血压有什么症状",
                         FakeLLM("MATCH (p:疾病)-[:has_symptom]->(q) WHERE p.name='高血压' RETURN q.name"))
    assert cql.startswith("MATCH")


def test_rejects_write():
    assert text_to_cypher("删库", FakeLLM("MATCH (n) DELETE n")) is None
