import uuid 
from app.api.schemas import PredictionRequest, PredictionResponse, DocumentSource
from app.ml.preprocessing import clean_and_sanitize_text
from app.ml.inference import DocumentRetriever

class RAGPredictionService:
    def __init__(self):
        self.retriever = DocumentRetriever()
        self.model_version = "tfedf-baseline-v1"

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        clean_question = clean_and_sanitize_text(request.question)

        chunk, score = self.retriever.retrieve(clean_question)

        CONFIDENCE_THRESHOLD = 0.3

        if chunk is not None and score >= CONFIDENCE_THRESHOLD:
            prediction_text = f"Согласно документу '{chunk.document_name}': {chunk.text}"
            manual_review = False 
            sources = [
                DocumentSource(
                    document_name = chunk.document_name, 
                    page_number = chunk.page_number,
                    chunk_id = chunk.chunk_id 
                )
            ]
        else: 
            prediction_text = "Релевантная информация в базе регламентов не найдена. Запрос перенаправлен оператору."
            manual_review = True
            sources = []

        return PredictionResponse(
            request_id=str(uuid.uuid4()),
            item_id=request.item_id,
            prediction=prediction_text,
            probability=score,
            model_version=self.model_version,
            manual_review_required=manual_review,
            sources=sources
        )