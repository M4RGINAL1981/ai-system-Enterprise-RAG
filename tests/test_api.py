from fastapi.testclient import TestClient
from app.main import app 

def test_predict_success():
    with TestClient(app) as client:
        payload = {"item_id": "TEST_01", "question": "Как открыть счет депо физическому лицу?"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        
        data = res.json()
        assert "request_id" in data
        assert data["source"] in ["qdrant_dense", "fallback_tfidf", "fallback_heuristic"]
        assert data["manual_review_required"] is False

def test_validate_error():
    with TestClient(app) as client:
        payload = {"item_id": "TEST_02", "question": "Как?"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422 

'''
def test_fallback_activation():
    original_model = app.state.model 
    app.state.model = None 
    payload = {"item_id": "TEST_03", "question": ""}
'''