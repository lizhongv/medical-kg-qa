from app.llm.understand import understand


class FakeLLM:
    available = True
    def chat(self, messages, tools=None, stream=False):
        return {"content": '{"intent": "病因", "disease": "高血压"}', "tool_calls": None}


class DeadLLM:
    available = False
    def chat(self, *a, **k): raise RuntimeError


def test_parse_llm_json():
    out = understand("高血压为啥得", FakeLLM(), ["病因", "定义"])
    assert out["intent"] == "病因" and out["disease"] == "高血压"


def test_unavailable():
    out = understand("x", DeadLLM(), ["病因"])
    assert out == {"intent": None, "disease": None}
