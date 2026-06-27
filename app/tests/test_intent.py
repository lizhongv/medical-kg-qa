from app.nlu.intent import IntentModel


def test_unavailable_returns_none():
    m = IntentModel(ckpt_dir=None)
    assert m.available is False
    out = m.predict("高血压怎么治")
    assert out["name"] is None and out["confidence"] == 0.0
