import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from app.data.contracts import ReglamentChunkSchema
from app.ml.preprocessing import clean_and_sanitize_text

RAW_PDF = Path("data/raw/terms-of-depository-activity-180806.pdf")
OUT_DIR = Path("data/processed")
ART_DIR = Path("artifacts")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bcs_reglaments"


def extract_chunks_from_pdf(pdf_path: Path) -> pd.DataFrame:
    """Извлекает текст из PDF и разбивает его на перекрывающиеся чанки."""
    reader = PdfReader(str(pdf_path))
    records = []
    chunk_counter = 1

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = clean_and_sanitize_text(raw_text)

        if len(cleaned) < 20:
            continue

        # Разделение прав доступа для будущего RBAC (ЛР № 7):
        # 1-89 стр. — общие условия для клиентов; 90-172 стр. — анкеты и внутренние формы
        access_group = "client" if page_idx <= 89 else "staff"

        start = 0
        while start < len(cleaned):
            end = start + CHUNK_SIZE
            chunk_text = cleaned[start:end].strip()

            if len(chunk_text) >= 20:
                records.append(
                    {
                        "chunk_id": f"chunk_{chunk_counter:04d}",
                        "document_name": pdf_path.name,
                        "page_number": page_idx,
                        "access_group": access_group,
                        "text": chunk_text,
                    }
                )
                chunk_counter += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return pd.DataFrame(records)


def upload_to_qdrant(df: pd.DataFrame, model: SentenceTransformer) -> int:
    """Генерирует плотные эмбеддинги и загружает их в Qdrant."""
    client = QdrantClient(url=QDRANT_URL)
    vector_size = model.get_embedding_dimension()
    if vector_size is None:
        raise ValueError("Не удалось определить размерность векторов модели")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    # Для моделей семейства E5 рекомендуется префикс "passage: " для документов
    passages = [f"passage: {text}" for text in df["text"].tolist()]
    embeddings = model.encode(passages, show_progress_bar=True, normalize_embeddings=True)

    points = []
    for idx, (_, row) in enumerate(df.iterrows()):
        points.append(
            PointStruct(
                id=idx + 1,
                vector=embeddings[idx].tolist(),
                payload={
                    "chunk_id": row["chunk_id"],
                    "document_name": row["document_name"],
                    "page_number": int(row["page_number"]),
                    "access_group": row["access_group"],
                    "text": row["text"],
                },
            )
        )

    # Загружаем батчами по 100 точек
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i : i + batch_size])

    return len(points)


def main():
    print("1. Извлечение текста и чанкинг PDF...")
    raw_df = extract_chunks_from_pdf(RAW_PDF)

    print("2. Валидация схемы данных через Pandera...")
    validated_df = ReglamentChunkSchema.validate(raw_df)
    clean_df = validated_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print("3. Разделение на train/test до обучения препроцессора (защита от Data Leakage)...")
    train_df, test_df = train_test_split(clean_df, test_size=0.2, random_state=42)

    print("4. Обучение TF-IDF + OneHot препроцессора строго на train...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(max_features=3000), "text"),
            ("dept", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["access_group"]),
        ]
    )

    x_train = preprocessor.fit_transform(train_df)
    x_test = preprocessor.transform(test_df)

    # Сохраняем обработанные таблицы и единый препроцессор (защита от Train-Serving Skew)
    train_df.to_parquet(OUT_DIR / "train.parquet", index=False)
    test_df.to_parquet(OUT_DIR / "test.parquet", index=False)
    joblib.dump(preprocessor, ART_DIR / "preprocessor.joblib")

    print("5. Векторизация и загрузка базы знаний в Qdrant...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    qdrant_points_count = upload_to_qdrant(clean_df, embed_model)

    report = {
        "raw_chunks": len(raw_df),
        "clean_chunks": len(clean_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "dropped_duplicates": len(raw_df) - len(clean_df),
        "tfidf_features": int(x_train.shape[1]),
        "qdrant_points_indexed": qdrant_points_count,
    }

    report_path = OUT_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("6. Логирование эксперимента и артефактов в MLflow...")
    mlflow.set_experiment("enterprise-rag-data-pipeline")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "train_share": 0.8,
                "qdrant_collection": COLLECTION_NAME,
            }
        )
        mlflow.log_metrics({k: float(v) for k, v in report.items()})
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(ART_DIR / "preprocessor.joblib"))

    print("Готово! Отчёт:", json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
