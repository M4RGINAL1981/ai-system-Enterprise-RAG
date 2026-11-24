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

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
