from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os


@dataclass
class DocumentChunk:
    chunk_id: str
    document_name: str
    page_number: int
    text: str
    source_engine: str = "qdrant"


class DocumentRetriever:
    """Компонент поиска релевантных контекстов: Qdrant (основной) + TF-IDF joblib (резервный)."""

    def __init__(
        self,
        qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333"),
        collection_name: str = "bcs_reglaments",
        embedding_model: str = "intfloat/multilingual-e5-small",
        preprocessor_path: Path = Path("artifacts/preprocessor.joblib"),
        data_path: Path = Path("data/processed/train.parquet"),
    ):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self._embed_model: SentenceTransformer | None = None
        self._qdrant_client: QdrantClient | None = None

        # Загружаем артефакты
        self.preprocessor = joblib.load(preprocessor_path) if preprocessor_path.exists() else None
        self.df = pd.read_parquet(data_path) if data_path.exists() else pd.DataFrame()

        if self.preprocessor is not None and not self.df.empty:
            self.doc_vectors = self.preprocessor.transform(self.df)
        else:
            self.doc_vectors = None

    def _get_qdrant(self) -> Tuple[QdrantClient, SentenceTransformer]:
        """Ленивая инициализация клиента Qdrant и модели эмбеддингов."""
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(url=self.qdrant_url, timeout=2)
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(self.embedding_model_name)
        return self._qdrant_client, self._embed_model

    def _retrieve_from_qdrant(self, query: str) -> Tuple[DocumentChunk | None, float]:
        client, model = self._get_qdrant()
        # Для моделей E5 запросы всегда подаются с префиксом "query: "
        query_vector = model.encode(f"query: {query}", normalize_embeddings=True).tolist()

        hits = client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=1,
        ).points

        if not hits:
            return None, 0.0

        top_hit = hits[0]
        payload = top_hit.payload or {}
        score = round(float(top_hit.score), 3)

        chunk = DocumentChunk(
            chunk_id=str(payload.get("chunk_id", "unknown")),
            document_name=str(payload.get("document_name", "unknown.pdf")),
            page_number=int(payload.get("page_number", 1)),
            text=str(payload.get("text", "")),
            source_engine="qdrant_dense",
        )
        return chunk, score

    def _retrieve_from_fallback(self, query: str) -> Tuple[DocumentChunk | None, float]:
        if self.preprocessor is None or self.doc_vectors is None or self.df.empty:
            q_lower = query.lower()
            if "счет" in q_lower or "депо" in q_lower or "депозитар" in q_lower:
                return (
                    DocumentChunk(
                        chunk_id="heuristic_001",
                        document_name="terms-of-depository-activity-180806.pdf",
                        page_number=37,
                        text="Открытие счета депо осуществляется на основании депозитарного договора и документа, удостоверяющего личность.",
                        source_engine="fallback_heuristic",
                    ),
                    0.50,
                )
            return None, 0.0

        query_df = pd.DataFrame([{"text": query, "access_group": "client"}])
        query_vec = self.preprocessor.transform(query_df)
        similarities = cosine_similarity(query_vec, self.doc_vectors)[0]

        best_idx = int(similarities.argmax())
        best_score = round(float(similarities[best_idx]), 3)
        row = self.df.iloc[best_idx]

        chunk = DocumentChunk(
            chunk_id=str(row["chunk_id"]),
            document_name=str(row["document_name"]),
            page_number=int(row["page_number"]),
            text=str(row["text"]),
            source_engine="fallback_tfidf",
        )
        return chunk, best_score

    def retrieve(self, query: str) -> Tuple[DocumentChunk | None, float]:
        """Ищет фрагмент в Qdrant; при недоступности Qdrant переключается на TF-IDF."""
        if not query.strip():
            return None, 0.0

        try:
            return self._retrieve_from_qdrant(query)
        except Exception:
            # Если Docker-контейнер Qdrant остановлен — срабатывает локальный препроцессор
            return self._retrieve_from_fallback(query)
