from app.settings import Settings
from app.orchestrator.router import route

S = Settings(accept_threshold=0.8, deny_threshold=0.4)


def test_chitchat():
    assert route({"kind": "chitchat", "intent": "greet"}, S) == "chitchat"


def test_fast_high_conf_matched():
    nlu = {"kind": "diagnosis", "intent": "定义", "confidence": 0.9,
           "slots": {"Disease": "高血压"}, "matched": True}
    assert route(nlu, S) == "fast"


def test_slow_low_conf():
    nlu = {"kind": "diagnosis", "intent": None, "confidence": 0.1,
           "slots": {"Disease": None}, "matched": False}
    assert route(nlu, S) == "slow"


def test_slow_high_conf_no_entity():
    nlu = {"kind": "diagnosis", "intent": "定义", "confidence": 0.95,
           "slots": {"Disease": None}, "matched": False}
    assert route(nlu, S) == "slow"
