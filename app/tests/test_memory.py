from app.settings import Settings
from app.memory.store import MemoryStore


def test_inmemory_roundtrip():
    m = MemoryStore(Settings(redis_url=None))
    assert m.get("u1") == {"slots": {}, "history": [], "last_intent": None}
    m.set("u1", {"slots": {"Disease": "高血压"}, "history": ["hi"], "last_intent": "定义"})
    assert m.get("u1")["slots"]["Disease"] == "高血压"


def test_sessions_are_independent():
    m = MemoryStore(Settings(redis_url=None))
    s1 = m.get("u1")
    s1["slots"]["Disease"] = "高血压"
    m.set("u1", s1)
    # a fresh session must NOT see u1's slots
    assert m.get("u2") == {"slots": {}, "history": [], "last_intent": None}
