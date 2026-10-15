from fastapi.testclient import TestClient

from comfy_kaggle_app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_mock(monkeypatch):
    monkeypatch.setenv("COMFY_MOCK", "true")
    with TestClient(app) as client:
        response = client.post("/generate", json={"prompt": "a fox", "seed": 42})
    assert response.status_code == 200
    assert response.json() == {"status": "mock", "image_base64": None}
