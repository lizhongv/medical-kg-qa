from app.nlu.chitchat import Chitchat


def test_keyword_fallback_greet():
    c = Chitchat(model_dir=None)   # 无模型 → 关键词降级
    assert c.classify("你好") == "greet"
    assert c.classify("再见") == "goodbye"


def test_non_chitchat_returns_none():
    c = Chitchat(model_dir=None)
    assert c.classify("高血压的症状有哪些") is None
