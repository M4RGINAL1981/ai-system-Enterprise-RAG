from fastapi.testclient import TestClient
from app.main import app 

def test_fallback_when_qdrant_down():
    with TestClient(app) as client:
        retriever = app.state.rag_service.retriever
        orig_url = retriever.qdrant_url
        try:
            # Имитируем падение Qdrant (ломаем адрес и сбрасываем клиент)
            retriever.qdrant_url = "http://localhost:9999"
            retriever._qdrant_client = None 

            res = client.post("/predict", json={"item_id": "FALLBACK_01", "question": "Какие документы нужны для открытия счета депо?"})
            assert res.status_code == 200
            data = res.json()
            assert data["source"] == "fallback_tfidf"
            assert data["manual_review_required"] is False 
        finally:
            retriever.qdrant_url = orig_url 
            retriever._qdrant_client = None 
            
            