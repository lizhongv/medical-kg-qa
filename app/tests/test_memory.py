from app.settings import Settings
from app.memory.store import MemoryStore


def test_inmemory_roundtrip():
    m = MemoryStore(Settings(redis_url=None))
    assert m.get("u1") == {"slots": {}, "history": [], "last_intent": None}
    m.set("u1", {"slots": {"Disease": "高血压"}, "history": ["hi"], "last_intent": "定义"})
    assert m.get("u1")["slots"]["Disease"] == "高血压"
