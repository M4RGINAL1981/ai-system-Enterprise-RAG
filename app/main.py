from app.api.schemas import PredictionRequest
from app.services.prediction import RAGPredictionService


def main():
    service = RAGPredictionService()

    # 1. Корректный запрос по регламенту БКС Банка (с маскированием ПДн)
    req1 = PredictionRequest(
        item_id="DEMO_01",
        question="Какие документы нужны физическому лицу для открытия счета депо? Ответ пришли на ivan@mail.ru",
    )
    print("--- ТЕСТ 1: Успешный поиск по регламенту БКС ---")
    print(service.predict(req1).model_dump_json(indent=2))

    # 2. Нецелевой запрос (срабатывание бизнес-правила fallback на оператора)
    req2 = PredictionRequest(
        item_id="DEMO_02",
        question="Какая сегодня погода на улице и какой рецепт яблочного пирога?",
    )
    print("\n--- ТЕСТ 2: Низкая уверенность -> Передача оператору ---")
    print(service.predict(req2).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
