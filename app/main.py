from app.api.schemas import PredictionRequest
from app.services.prediction import RAGPredictionService

def main():
    service = RAGPredictionService()
    
    # 1. Корректный запрос
    req1 = PredictionRequest(
        item_id="DEMO_01",
        question="Сколько дней составляет отпуск у сотрудников? Напиши на test@mail.ru"
    )
    print("--- ТЕСТ 1: Успешный запрос ---")
    print(service.predict(req1).model_dump_json(indent=2))

    # 2. Неизвестный запрос (срабатывание бизнес-правила fallback)
    req2 = PredictionRequest(
        item_id="DEMO_02",
        question="Какая сегодня погода на улице и какой рецепт пирога?"
    )
    print("\n--- ТЕСТ 2: Передача оператору (score < 0.4) ---")
    print(service.predict(req2).model_dump_json(indent=2))

if __name__ == "__main__":
    main()