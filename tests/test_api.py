from starlette.testclient import TestClient

from api import app


def test_index_get():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "Artifact Classifier" in r.text


def test_predict_no_file():
    client = TestClient(app)
    r = client.post("/", files={})
    assert r.status_code == 200
    # If no file was sent, page should still render
    assert "Artifact Classifier" in r.text


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
