from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import get_current_tenant
from app.database import get_db
from app.models import Tenant
from app.retrieval import generate_answer, ingest_document, search_similar_chunks

app = FastAPI()


class DocumentIngestRequest(BaseModel):
    source_document_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


@app.get("/whoami")
def get_verfiy(tenant: Tenant = Depends(get_current_tenant)):
    return {"tenant_id": tenant.id, "tenant_name": tenant.name}


@app.post("/documents")
def ingest_document_route(
    payload: DocumentIngestRequest,
    db: Session = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    chunk_count = ingest_document(
        db=db,
        tenant_id=tenant.id,
        source_document_id=payload.source_document_id,
        text=payload.text,
    )
    return {
        "tenant_id": str(tenant.id),
        "source_document_id": payload.source_document_id,
        "chunks_stored": chunk_count,
    }


@app.post("/ask")
def ask_question(
    payload: AskRequest,
    db: Session = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    matches = search_similar_chunks(
        db=db,
        tenant_id=tenant.id,
        question=payload.question,
        limit=payload.limit,
    )

    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No matching chunks found for this tenant.",
        )

    context = "\n\n".join(chunk.content for chunk in matches)
    answer = generate_answer(question=payload.question, context=context)
    return {
        "tenant_id": str(tenant.id),
        "question": payload.question,
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
        "context_preview": context[:2000],
    }
