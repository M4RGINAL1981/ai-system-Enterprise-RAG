from app.api.schemas import PredictionRequest, PredictionResponse
from app.services.prediction import RAGPredictionService
import time, logging

from app.core.logger import json_logger
logger = json_logger("rag_api")

# Docker
from contextlib import asynccontextmanager 
from fastapi import FastAPI, HTTPException

MODEL_VERSION = "v0.1.0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_service = RAGPredictionService()
    yield

app = FastAPI(title="AI system Enterprise RAG", version=MODEL_VERSION, lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    start = time.perf_counter()
    try:
        response = app.state.rag_service.predict(req)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        rec = logger.makeRecord("rag_api", logging.INFO, "", 0, "inference", (), None)
        rec.extra_data = {
            "trace_id": response.request_id,
            "item_id": req.item_id,
            "probability": response.probability,
            "source": response.source,
            "manual_review": response.manual_review_required,
            "latency_ms": latency_ms
        }
        logger.handle(rec)
        return response
    except Exception as exc:
        logger.error(f"Ошибка при обработке запроса {req.item_id}:{exc}")
        raise HTTPException(status_code=500, detail="Ошибка инференса") from exc

