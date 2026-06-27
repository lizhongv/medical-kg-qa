from app.safety.guardrails import apply


def test_disclaimer_appended():
    out = apply("高血压可以挂心内科。", "高血压挂什么科")
    assert "仅供参考" in out


def test_diagnosis_request_prefixed():
    out = apply("可能是感冒。", "我这症状是得了什么病")
    assert "及时就医" in out or "面诊" in out


def test_pii_masked():
    out = apply("联系13812345678", "x")
    assert "13812345678" not in out
