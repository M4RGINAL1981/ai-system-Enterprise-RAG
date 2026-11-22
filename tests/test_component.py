import pytest 
from pydantic import ValidationError

from app.api.schemas import PredictionRequest                    # Cхемы
from app.ml.inference import DocumentChunk, DocumentRetriever
from app.services.prediction import RAGPredictionService


class MockRetriever(DocumentRetriever):
    """Тестовый ретривер для быстрых изолированных юнит-тестов без поднятия Qdrant."""

    def __init__(self):
        pass

    def retrieve(self, query: str):
        if "счет депо" in query.lower():
            return (
                DocumentChunk(
                    chunk_id="chunk_0150",
                    document_name="terms-of-depository-activity-180806.pdf",
                    page_number=37,
                    text="Открытие счета депо владельца для физических лиц осуществляется на основании паспорта.",
                    source_engine="qdrant_dense",
                ),
                0.86,
            )
        if "список владельцев" in query.lower():
            return (
                DocumentChunk(
                    chunk_id="chunk_0095",
                    document_name="terms-of-depository-activity-180806.pdf",
                    page_number=26,
                    text="Список всех владельцев ценных бумаг представляется реестродержателю в течение пяти рабочих дней.",
                    source_engine="qdrant_dense",
                ),
                0.82,
            )
        return None, 0.20


@pytest.fixture
def service():
    return RAGPredictionService(retriever=MockRetriever())


def test_predict_success(service):
    request = PredictionRequest(
        item_id="TEST_01",
        question="Как открыть счет депо физическому лицу?",
    )
    response = service.predict(request)

    assert response.manual_review_required is False
    assert response.probability >= 0.82
    assert len(response.sources) == 1
    assert response.sources[0].page_number == 37


def test_predict_different_query(service):
    request = PredictionRequest(
        item_id="TEST_02",
        question="В какой срок предоставляется список владельцев ценных бумаг?",
    )
    response = service.predict(request)

    assert response.manual_review_required is False
    assert response.sources[0].chunk_id == "chunk_0095"


def test_predict_low_confidence_fallback(service):
    request = PredictionRequest(
        item_id="TEST_03",
        question="Какая сегодня погода и как приготовить шарлотку?",
    )
    response = service.predict(request)

    assert response.manual_review_required is True
    assert len(response.sources) == 0


def test_predict_invalid_input():
    with pytest.raises(ValidationError):
        PredictionRequest(
            item_id="TEST_04",
            question="Как?",
        )
