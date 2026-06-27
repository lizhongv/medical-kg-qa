import pytest
from app.settings import Settings
from app.llm.client import LLMClient, LLMUnavailable


class FakeCompletions:
    def create(self, **kw):
        class M:
            content = "hello"
            tool_calls = None
        class C:
            message = M()
        class R:
            choices = [C()]
        return R()


class FakeClient:
    def __init__(self):
        self.chat = type("X", (), {"completions": FakeCompletions()})()


def test_unavailable_without_key():
    c = LLMClient(Settings(llm_api_key=None))
    assert c.available is False
    with pytest.raises(LLMUnavailable):
        c.chat([{"role": "user", "content": "hi"}])


def test_chat_returns_content():
    c = LLMClient(Settings(llm_api_key="sk-x"))
    c._client = FakeClient()
    out = c.chat([{"role": "user", "content": "hi"}])
    assert out["content"] == "hello"
