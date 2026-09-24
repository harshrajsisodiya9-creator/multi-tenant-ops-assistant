from __future__ import annotations

from typing import Sequence
from uuid import UUID

from fastembed import TextEmbedding
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import setting
from app.models import Chunk

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

_embedding_model: TextEmbedding | None = None


def get_embedding_model() -> TextEmbedding:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return _embedding_model


def split_text_into_chunks(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_text(text)


def _normalize_embedding(raw_embedding: object) -> list[float]:
    if raw_embedding is None:
        return []

    if hasattr(raw_embedding, "__iter__") and not isinstance(
        raw_embedding, (str, bytes, bytearray)
    ):
        items = list(raw_embedding)
        if not items:
            return []

        if isinstance(items[0], (list, tuple)):
            if len(items) == 1:
                return _normalize_embedding(items[0])
            flattened: list[float] = []
            for item in items:
                flattened.extend(_normalize_embedding(item))
            return flattened

        return [float(item) for item in items]

    return [float(raw_embedding)]


def embed_text(text: str) -> list[float]:
    model = get_embedding_model()
    embeddings = list(model.embed([text]))
    return _normalize_embedding(embeddings[0] if embeddings else [])


def ingest_document(
    db: Session, tenant_id: UUID, source_document_id: str, text: str
) -> int:
    chunks = split_text_into_chunks(text)
    if not chunks:
        return 0

    model = get_embedding_model()
    embeddings = list(model.embed(chunks))

    for chunk_index, (chunk_text, embedding_values) in enumerate(
        zip(chunks, embeddings)
    ):
        db.add(
            Chunk(
                tenant_id=tenant_id,
                source_document_id=source_document_id,
                chunk_index=chunk_index,
                content=chunk_text,
                embedding=_normalize_embedding(embedding_values),
            )
        )

    db.commit()
    return len(chunks)


def search_similar_chunks(
    db: Session, tenant_id: UUID, question: str, limit: int = 5
) -> list[Chunk]:
    if not question or not question.strip():
        return []

    query_embedding = embed_text(question)
    stmt = (
        select(Chunk)
        .where(Chunk.tenant_id == tenant_id)
        .order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())


def generate_answer(question: str, context: str) -> str:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context or not context.strip():
        return "I couldn't find enough relevant context to answer that question."

    if not setting.groq_api_key:
        raise ValueError("GROQ API key is not configured.")

    llm = ChatGroq(
        model=setting.groq_model,
        api_key=setting.groq_api_key,
        temperature=0,
    )

    prompt = (
        "Answer the user's question using only the provided context. "
        "If the answer is not in the context, say that the information is not available.\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )
    response = llm.invoke(prompt)
    return str(getattr(response, "content", response)).strip()
