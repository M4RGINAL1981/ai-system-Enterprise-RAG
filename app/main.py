from app.api.schemas import PredictionRequest, PredictionResponse
from app.services.prediction import RAGPredictionService

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
    try:
        return app.state.rag_service.predict(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Ошибка инференса") from exc

