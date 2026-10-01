from __future__ import annotations

from io import BytesIO
from uuid import UUID

from fastapi import UploadFile
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


def load_txt_document(file_name: str, content: bytes) -> str:
    del file_name
    return content.decode("utf-8")


def load_md_document(file_name: str, content: bytes) -> str:
    del file_name
    return content.decode("utf-8")


def load_pdf_document(file_name: str, content: bytes) -> str:
    del file_name
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(content))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def load_document_text(file_name: str, content: bytes) -> str:
    name = (file_name or "").lower()
    if name.endswith(".pdf"):
        return load_pdf_document(file_name, content)
    if name.endswith(".txt"):
        return load_txt_document(file_name, content)
    if name.endswith((".md", ".markdown")):
        return load_md_document(file_name, content)
    raise ValueError(
        f"Unsupported file type for '{file_name}'. Supported types: .pdf, .txt, .md"
    )


async def read_uploaded_document(file: UploadFile) -> str:
    if file is None or not getattr(file, "filename", None):
        raise ValueError("A file is required.")

    content = await file.read()
    if not content or not content.strip():
        raise ValueError("The uploaded file is empty or unreadable.")

    text = load_document_text(file.filename, content)
    if not text or not text.strip():
        raise ValueError("The uploaded file is empty or unreadable.")

    return text


def index_document(
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


def ingest_document(
    db: Session, tenant_id: UUID, source_document_id: str, text: str
) -> int:
    return index_document(
        db=db,
        tenant_id=tenant_id,
        source_document_id=source_document_id,
        text=text,
    )


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


def answer_question(
    db: Session, tenant_id: UUID, question: str, limit: int = 5
) -> dict:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    matches = search_similar_chunks(
        db=db,
        tenant_id=tenant_id,
        question=question,
        limit=limit,
    )

    if not matches:
        return {
            "tenant_id": str(tenant_id),
            "question": question,
            "matches_found": 0,
            "top_matches": [],
            "answer": "I couldn't find enough relevant context to answer that question.",
            "context_preview": "",
        }

    context = "\n\n".join(chunk.content for chunk in matches)
    answer = generate_answer(question=question, context=context)

    return {
        "tenant_id": str(tenant_id),
        "question": question,
        "matches_found": len(matches),
        "top_matches": [
            {
                "chunk_id": chunk.id,
                "source_document_id": chunk.source_document_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
            }
            for chunk in matches
        ],
        "answer": answer,
    }
