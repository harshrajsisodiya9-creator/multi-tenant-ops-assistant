from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import (
    TokenClaims,
    get_authenticated_customer,
    get_current_tenant,
    get_reviewer,
    mint_token,
)
from app.config import setting
from app.database import get_db
from app.graph.builder import build_graph
from app.hybrid_search import answer_question
from app.knowledgebase import index_document, read_uploaded_document
from app.models import Tenant

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
agent_graph = build_graph()

_THREAD_OWNERS: dict[str, tuple[str, str]] = {}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: str | None = None


class HumanDecision(BaseModel):
    index: int
    action: str = Field(..., pattern="^(approve|deny|override)$")
    note: str | None = None
    reason: str | None = None
    custom_response: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    decisions: list[HumanDecision]


class MintTokenRequest(BaseModel):
    tenant_id: str
    customer_id: str
    role: str | None = None
    expires_minutes: int = 60


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


@app.post("/agent")
def agent_endpoint(
    payload: AgentRequest,
    claims: TokenClaims = Depends(get_authenticated_customer),
):
    if payload.thread_id:
        thread_id = payload.thread_id
        owner = _THREAD_OWNERS.get(thread_id)
        if owner is not None and owner != (claims.tenant_id, claims.customer_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Thread belongs to a different customer",
            )
        if owner is None:
            _THREAD_OWNERS[thread_id] = (claims.tenant_id, claims.customer_id)
    else:
        thread_id = uuid4().hex
        _THREAD_OWNERS[thread_id] = (claims.tenant_id, claims.customer_id)

    config = {"configurable": {"thread_id": thread_id}}

    state = agent_graph.invoke(
        {
            "tenant_id": claims.tenant_id,
            "user_message": payload.message,
            "customer_id": claims.customer_id,
            "customer_email": claims.email,
            "conversation_id": thread_id,
        },
        config=config,
    )

    response_phase = state.get("response_phase", "complete")

    result: dict = {
        "tenant_id": state.get("tenant_id"),
        "user_message": state.get("user_message"),
        "router_output": state.get("router_output"),
        "policy_decision": state.get("policy_decision"),
        "response": state.get("response", ""),
        "thread_id": thread_id,
    }

    if response_phase == "partial":
        escalations = [
            r for r in (state.get("intent_results") or [])
            if r.get("status") == "awaiting_human"
        ]
        result["status"] = "interrupted"
        result["escalations"] = escalations

    return result


@app.post("/agent/resume")
def agent_resume_endpoint(
    payload: ResumeRequest,
    claims: TokenClaims = Depends(get_reviewer),
):
    owner = _THREAD_OWNERS.get(payload.thread_id)
    if owner is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found",
        )
    if owner[0] != claims.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Thread belongs to a different tenant",
        )

    config = {"configurable": {"thread_id": payload.thread_id}}

    decisions = [d.model_dump() for d in payload.decisions]

    state = agent_graph.invoke(
        Command(resume={"decisions": decisions}),
        config=config,
    )

    return {
        "tenant_id": state.get("tenant_id"),
        "response": state.get("response", ""),
        "intent_results": state.get("intent_results"),
    }


if setting.app_env in ("dev", "test"):

    @app.post("/dev/token")
    def dev_mint_token(payload: MintTokenRequest):
        token = mint_token(
            payload.tenant_id,
            payload.customer_id,
            role=payload.role,
            expires_minutes=payload.expires_minutes,
        )
        return {"token": token}
