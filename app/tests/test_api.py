from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat_stub():
    r = client.post("/chat", json={"text": "高血压怎么治", "session_id": "u1"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and "path" in body
