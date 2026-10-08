import pytest
from pydantic import ValidationError
from app.api.schemas import PredictionRequest
from app.services.prediction import RAGPredictionService

@pytest.fixture
def service():
    return RAGPredictionService()

# Тест 1: Успешный поиск релевантного ответа
def test_predict_success(service):
    request = PredictionRequest(
        item_id="TEST_01",
        question="Сколько дней составляет ежегодный оплачиваемый отпуск?"
    )
    response = service.predict(request)
    
    assert response.manual_review_required is False
    assert response.probability >= 0.4
    assert len(response.sources) > 0
    assert response.sources[0].document_name == "Положение_об_отпусках.pdf"

# Тест 2: Разные запросы дают отличающийся результат
def test_predict_different_query(service):
    request = PredictionRequest(
        item_id="TEST_02",
        question="Как подать заявку на замену ноутбука и монитора?"
    )
    response = service.predict(request)
    
    assert response.manual_review_required is False
    assert response.sources[0].document_name == "Регламент_ИТ_поддержки.pdf"

# Тест 3: Низкая уверенность -> Передача человеку (Fallback)
def test_predict_low_confidence_fallback(service):
    request = PredictionRequest(
        item_id="TEST_03",
        question="Какая погода сегодня в офисе и какой рецепт пирога?"
    )
    response = service.predict(request)
    
    assert response.manual_review_required is True
    assert response.probability < 0.4
    assert len(response.sources) == 0

# Тест 4: Валидация входа (слишком короткий вопрос)
def test_predict_invalid_input():
    with pytest.raises(ValidationError):
        PredictionRequest(
            item_id="TEST_04",
            question="Как?"  # Меньше 5 символов -> Pydantic бросит ошибку
        )