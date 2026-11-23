FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1

WORKDIR /app 
# Установка зависимостей отдельным слоем (Кэширование)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода и артефактов
COPY app/ ./app
COPY artifacts/ ./artifacts
COPY data/processed/ ./data/processed

EXPOSE 8000 

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]docker run --rm --network ai-system-enterprise-rag_default -e QDRANT_URL=http://qdrant:6333 -v "${PWD}/tests:/app/tests" enterprise-rag-api:v0.1.0 python -m pytest tests/test_component.py tests/test_api.py tests/test_resilience.py -v
