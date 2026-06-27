from app.llm.generate import generate


class FakeLLM:
    available = True
    def chat(self, messages, tools=None, stream=False):
        return {"content": "根据资料,高血压是一种慢性病。", "tool_calls": None}


class DeadLLM:
    available = False


def test_with_llm():
    out = generate("高血压是什么", ["慢性病", "需长期管理"], FakeLLM())
    assert "高血压" in out


def test_no_facts():
    assert "未找到" in generate("x", [], FakeLLM())


def test_fallback_without_llm():
    out = generate("高血压是什么", ["慢性病"], DeadLLM())
    assert "慢性病" in out
