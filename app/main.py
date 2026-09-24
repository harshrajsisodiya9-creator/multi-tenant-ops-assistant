from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import get_current_tenant
from app.database import get_db
from app.knowledgebase import answer_question, index_document, read_uploaded_document
from app.models import Tenant

app = FastAPI()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


@app.get("/whoami")
def get_verfiy(tenant: Tenant = Depends(get_current_tenant)):
    return {"tenant_id": tenant.id, "tenant_name": tenant.name}


@app.post("/documents")
async def ingest_document_route(
    file: UploadFile = File(...),
    source_document_id: str | None = Form(None),
    db: Session = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    try:
        text = await read_uploaded_document(file)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    document_id = source_document_id or Path(file.filename).stem
    chunk_count = index_document(
        db=db,
        tenant_id=tenant.id,
        source_document_id=document_id,
        text=text,
    )
    return {
        "tenant_id": str(tenant.id),
        "source_document_id": document_id,
        "chunks_stored": chunk_count,
    }


@app.post("/ask")
def ask_question(
    payload: AskRequest,
    db: Session = Depends(get_db),
    tenant: Tenant = Depends(get_current_tenant),
):
    result = answer_question(
        db=db,
        tenant_id=tenant.id,
        question=payload.question,
        limit=payload.limit,
    )

    if result["matches_found"] == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No matching chunks found for this tenant.",
        )

    return result
