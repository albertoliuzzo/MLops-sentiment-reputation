from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_predict_positive():
    r = client.post("/predict", json={"text": "I love this product, it works great!"})
    assert r.status_code == 200

    data = r.json()
    assert "Sentiment" in data
    assert "Probabilità" in data

    assert data["Sentiment"] in {"positive", "neutral", "negative"}
    assert 0.0 <= data["Probabilità"] <= 1.0

def test_predict_validation_error_empty_text():
    # Qui verifichiamo che la validazione Pydantic funzioni
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422
