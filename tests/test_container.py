import requests

def test_container_health():
    res = requests.get("http://localhost:8000/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["model_version"] == "v0.1.0"

def test_container_predict():
    payload = {
        "item_id": "TEST_01",
        "question": "Как открыть счет депо физическому лицу?",
    }
    res = requests.post("http://localhost:8000/predict", json=payload)
    data = res.json()
    
    assert data["manual_review_required"] is False
    assert data["probability"] >= 0.82
    assert len(data["sources"]) == 1
    assert data["sources"][0]["page_number"] == 37

def test_container_validation_error():
    payload = {
            "item_id": "TEST_01",
            "question": "Как?",
    }
    res = requests.post("http://localhost:8000/predict", json=payload)
    assert res.status_code == 422