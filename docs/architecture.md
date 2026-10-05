# Архитектура Enterprise-RAG: Диаграммы и компоненты

## Контекстная диаграмма (C4 Model — Level 1: System Context)

### 1. Код диаграммы для Mermaid
```mermaid
flowchart LR
    User[Сотрудник / Конечный пользователь] -->|1. HTTPS: Вопрос по регламенту| RAG[Проектируемая Enterprise RAG Система]
    RAG -->|2. JSON: Ответ + Ссылки на источники| User

    Admin[Администратор базы знаний] -->|3. HTTPS/Web: Загрузка новых регламентов| RAG

    subgraph Corporate_Infrastructure [Корпоративный контур]
        CorpStorage[(Корпоративное хранилище документов / S3)] -->|4. SMB/S3: Синхронизация файлов PDF/MD| RAG
    end

    subgraph AI_Compute_Layer [Контур вычислений ИИ]
        RAG -->|5. HTTPS/gRPC: Запрос на генерацию Prompt + Context| LLM[Внешний / Локальный LLM Сервис vLLM / YandexGPT]
        LLM -->|6. JSON / Stream: Сгенерированный текст ответа| RAG
    end

    subgraph Observability_Stack [Контур эксплуатации]
        RAG -.->|7. OpenMetrics: Экспорт метрик задержек и качества поиска Hit Rate| Monitoring[Система мониторинга Prometheus]
        RAG -.->|8. Syslog: Структурированные JSON-логи запросов| Logging[Централизованное хранилище логов Elastic/Loki]
    end

    style RAG fill:#f9f,stroke:#333,stroke-width:3px
    style LLM fill:#bbf,stroke:#333,stroke-width:1px
    style CorpStorage fill:#fbb,stroke:#333,stroke-width:1px
```

### 2. Описание информационных потоков
1. **User -> RAG:** HTTPS (REST API), формат JSON (текст `question` и `session_id`).
2. **RAG -> User:** HTTPS Response, формат JSON (текст `answer`, массив источников `sources`).
3. **Corp Storage -> RAG:** S3 API, асинхронный пакетный процесс загрузки бинарных PDF/MD файлов.
4. **RAG -> LLM:** HTTPS/gRPC, передача системного промта и найденного контекста документов для генерации.

---

## Компонентная декомпозиция (C4 Model — Level 2: Container / Component)

### 1. Код диаграммы компонентов для Mermaid
```mermaid
flowchart TB
    Client[Сотрудник / Web-интерфейс] -->|HTTP POST /api/v1/predict| API[FastAPI Gateway]
    Admin[Администратор] -->|Загрузка документов| API

    subgraph AppContainer ["Контейнер приложения (FastAPI Service)"]
        API --> Auth[Модуль аутентификации X-API-Key]
        
        subgraph OnlinePipeline [Онлайн-контур инференса]
            Auth --> Validator[Pydantic Request Validator]
            Validator --> QueryEmbedder[Query Embedding Engine]
            QueryEmbedder --> VectorSearch[Hybrid Retriever BM25 + Dense]
            VectorSearch --> PromptBuilder[Prompt Engineering Module]
            PromptBuilder --> LLMClient[LLM API Connector]
            LLMClient --> PostProcessor[Post-Processor & Guardrails]
        end

        subgraph OfflinePipeline [Офлайн-контур инжеста данных]
            API --> DocumentParser[Doc/PDF Parser & Text Splitter]
            DocumentParser --> ChunkEmbedder[Chunk Embedding Engine]
        end
    end

    subgraph ArtifactStore [Хранилище моделей]
        QueryEmbedder -.->|Загрузка локального эмбеддера| LocalEmbedder[(Local Embedder Artifacts / HuggingFace)]
        ChunkEmbedder -.->|Загрузка локального эмбеддера| LocalEmbedder
    end

    subgraph DataStore [Слой персистентности]
        ChunkEmbedder -->|Запись векторов и метаданных| VectorDB[(Векторная БД: Qdrant / pgvector)]
        VectorSearch -->|Поиск релевантного контекста| VectorDB
        PostProcessor -->|Запись истории диалогов и фактов поиска| AppDB[(Реляционная БД: PostgreSQL / SQLite)]
    end

    subgraph ObservabilityStack [Контур мониторинга]
        API -.->|Сбор метрик HTTP / Latency| MetricsEndpoint[/metrics Endpoint/]
        PostProcessor -.->|Логирование бизнес-метрик и скоров| LogsOutput[JSON Logger]
    end

    PostProcessor -->|HTTP 200: Ответ + Ссылки| Client
    
    style AppContainer fill:#fff,stroke:#333,stroke-width:2px
    style VectorDB fill:#fbb,stroke:#333,stroke-width:1px
    style AppDB fill:#fbb,stroke:#333,stroke-width:1px
```

### 2. Спецификация компонентов системы
* `app.api.routes` — Обработка запросов и маршрутизация (fastapi).
* `app.api.schemas` — Pydantic-валидация сырых данных (pydantic).
* `app.ml.parser` — Парсинг PDF/Markdown и деление на чанки (langchain-text-splitters).
* `app.ml.embedder` — Расчет векторных представлений текста (sentence-transformers).
* `app.ml.retriever` — Поиск документов в векторной БД (qdrant-client, rank-bm25).

---

## 💻 Системные эндпоинты (Этап 5.2)

### GET /health — Проверка статуса (Health Check)
```json
{
  "status": "healthy",
  "components": {
    "api_gateway": "up",
    "vector_database": "connected",
    "llm_provider": "connected"
  },
  "model_loaded": true,
  "model_version": "bge-m3_llama-3-8b-instruct",
  "uptime_seconds": 86400
}
```

### GET /metrics — Метрики для Prometheus
Формат OpenMetrics:
```text
http_requests_total{method="POST",endpoint="/api/v1/predict",status="200"} 1420
http_request_duration_seconds_bucket{le="1.0"} 1350
rag_retrieval_confidence_score 0.89
```
