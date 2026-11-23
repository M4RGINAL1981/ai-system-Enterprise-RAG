import uuid
from app.api.schemas import DocumentSource, PredictionRequest, PredictionResponse
from app.ml.inference import DocumentRetriever
from app.ml.preprocessing import clean_and_sanitize_text


class RAGPredictionService:
    def __init__(self, retriever: DocumentRetriever | None = None):
        self.retriever = retriever or DocumentRetriever()
        self.base_version = "hybrid-rag-v2"

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        clean_question = clean_and_sanitize_text(request.question)
        chunk, score = self.retriever.retrieve(clean_question)

        # Порог зависит от того, какой движок сработал (Dense E5 vs Sparse TF-IDF)
        engine = chunk.source_engine if chunk else "none"
        threshold = 0.82 if engine == "qdrant_dense" else 0.25

        if chunk is not None and score >= threshold:
            prediction_text = (
                f"Согласно регламенту '{chunk.document_name}' (стр. {chunk.page_number}): {chunk.text}"
            )
            manual_review = False
            sources = [
                DocumentSource(
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                )
            ]
        else:
            prediction_text = (
                "Релевантная информация в базе регламентов не найдена. Запрос перенаправлен оператору."
            )
            manual_review = True
            sources = []

        return PredictionResponse(
            request_id=str(uuid.uuid4()),
            item_id=request.item_id,
            prediction=prediction_text,
            probability=min(max(score, 0.0), 1.0),
            model_version=f"{self.base_version}:{engine}",
            manual_review_required=manual_review,
            sources=sources,
            source=engine
        )
