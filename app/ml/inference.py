from dataclasses import dataclass 
from typing import List, Tuple 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass 
class DocumentChunk:
    chunk_id: str
    document_name: str 
    page_number: int 
    text: str 

KNOWLEDGE_BASE : List[DocumentChunk] = [
    DocumentChunk(
        chunk_id = "chunk_001",
        document_name = "Положение_об_отпусках.pdf",
        page_number = 3,
        text = "Продолжительность ежегодного основного оплачиваемого отпуска для всех сотрудников составляет 28 календарных дней."
    ),
    DocumentChunk(
        chunk_id="chunk_002",
        document_name="Положение_об_отпусках.pdf",
        page_number=5,
        text="Дополнительный оплачиваемый отпуск предоставляется работникам с ненормированным рабочим днем и составляет 3 календарных дня."
    ),
    DocumentChunk(
        chunk_id="chunk_003",
        document_name="Регламент_ИТ_поддержки.pdf",
        page_number=1,
        text="Заявки на выдачу и замену оборудования (ноутбуки, мониторы) оформляются через внутренний портал ServiceDesk."
    ),
]

class DocumentRetriever:
    """ Компонент поиска релевантных контекстов в базе знаний (Inference Engine)."""

    def __init__(self, documents: List[DocumentChunk] = KNOWLEDGE_BASE):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        corpus = [doc.text for doc in self.documents]
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str) -> Tuple[DocumentChunk | None, float]:
        """ Ищет наиболее подходящий фрагмент документа по запросу."""
        if not query.strip():
            return None, 0.0

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]

        best_index = similarities.argmax()
        best_score = float(similarities[best_index])

        if best_score < 0.05:
            return None, 0.0

        return self.documents[best_index], round(best_score, 2)
    