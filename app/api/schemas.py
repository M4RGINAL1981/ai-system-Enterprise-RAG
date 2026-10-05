from pydantic import BaseModel, Field
from typing import Optional, List

class PredictionRequest(BaseModel):
    item_id: str = Field(description="Уникальный идентификатор объекта / транзакции / пользователя")
    question: str = Field(min_length=5, max_length=1000, description="Текстовый вопрос сотрудника по регламентам компании")
    department_scope: Optional[str] = Field(default=None, description="Код бизнес-отдела для фильтрации документов")

    class Config:
        json_schema_extra = {
            "example": {
                "item_id": "USER_SESSION_42",
                "question": "Какова продолжительность ежегодного оплачиваемого отпуска для сотрудников ИТ-отдела?",
                "department_scope": "HR"
            }
        }

class DocumentSource(BaseModel):
    document_name: str = Field(description="Наименование файла-первоисточника контекста")
    page_number: Optional[int] = Field(default=None, description="Номер страницы в документе")
    chunk_id: str = Field(description="Уникальный идентификатор текстового фрагмента в векторной БД")

class PredictionResponse(BaseModel):
    request_id: str = Field(description="Уникальный идентификатор запроса (UUIDv4) для аудита")
    item_id: str = Field(description="Идентификатор переданного объекта")
    prediction: str = Field(description="Текстовый вердикт (сгенерированный ответ языковой модели)")
    probability: float = Field(ge=0.0, le=1.0, description="Вероятность наступления целевого события (скор релевантности поиска)")
    model_version: str = Field(description="Семантическая версия используемой модели")
    manual_review_required: bool = Field(description="Флаг необходимости проверки оператором")
    sources: List[DocumentSource] = Field(description="Список верифицированных источников контекста")
