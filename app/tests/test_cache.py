from app.cache.semantic_cache import SemanticCache


def test_exact_hit():
    c = SemanticCache()
    assert c.lookup("高血压怎么治") is None
    c.save("高血压怎么治", "答案A")
    assert c.lookup("高血压怎么治") == "答案A"


def test_controller_cache_hit():
    from app.settings import Settings
    from app.orchestrator.controller import Controller
    from app.cache.semantic_cache import SemanticCache
    class NluDiag:
        def analyze(self, t):
            return {"kind": "diagnosis", "intent": "定义", "confidence": 0.95,
                    "slots": {"Disease": "高血压"}, "matched": True}
    class KG3:
        available = True
        def query(self, c, params=None): return [{"x": "慢性病"}]
    cache = SemanticCache()
    c = Controller(NluDiag(), KG3(), Settings(), cache=cache)
    first = c.handle("高血压是什么", "u_c")
    assert first["path"] == "fast"
    second = c.handle("高血压是什么", "u_c")
    assert second["path"] == "cache" and second["answer"] == first["answer"]
