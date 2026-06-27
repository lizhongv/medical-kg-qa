from app.cache.semantic_cache import SemanticCache


def test_exact_hit():
    c = SemanticCache()
    assert c.lookup("高血压怎么治") is None
    c.save("高血压怎么治", "答案A")
    assert c.lookup("高血压怎么治") == "答案A"
