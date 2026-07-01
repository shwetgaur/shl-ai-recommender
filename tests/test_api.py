from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_chat_schema():
    with TestClient(app) as client:
        r = client.post("/chat", json={"messages": [{"role": "user", "content": "I need an assessment."}]})
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) >= {"reply", "recommendations", "end_of_conversation"}
        assert isinstance(body["reply"], str)
        assert isinstance(body["recommendations"], list)
        assert isinstance(body["end_of_conversation"], bool)


def test_chat_empty_messages():
    with TestClient(app) as client:
        r = client.post("/chat", json={"messages": []})
        assert r.status_code == 200
        assert r.json()["recommendations"] == []


def test_chat_missing_messages_field():
    with TestClient(app) as client:
        r = client.post("/chat", json={})
        assert r.status_code == 200


def test_chat_recommendation_item_shape():
    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Java backend engineer, Spring and SQL, confirmed."}]},
        )
        body = r.json()
        for item in body["recommendations"]:
            assert set(item.keys()) >= {"name", "url", "test_type"}
            assert item["url"].startswith("http")
