from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import get_current_tenant
from app.database import get_db
from app.graph.builder import build_graph
from app.hybrid_search import answer_question
from app.knowledgebase import index_document, read_uploaded_document
from app.models import Tenant

app = FastAPI()
agent_graph = build_graph()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1)
    customer_email: str | None = None


class HumanDecision(BaseModel):
    index: int
    action: str = Field(..., pattern="^(approve|deny|override)$")
    note: str | None = None
    reason: str | None = None
    custom_response: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    decisions: list[HumanDecision]


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
    tenant: Tenant = Depends(get_current_tenant),
):
    thread_id = uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    state = agent_graph.invoke(
        {
            "tenant_id": str(tenant.id),
            "user_message": payload.message,
            "customer_email": payload.customer_email,
        },
        config=config,
    )

    response_phase = state.get("response_phase", "complete")

    result = {
        "tenant_id": state.get("tenant_id"),
        "user_message": state.get("user_message"),
        "router_output": state.get("router_output"),
        "policy_decision": state.get("policy_decision"),
        "response": state.get("response", ""),
    }

    if response_phase == "partial":
        escalations = [
            r for r in (state.get("intent_results") or [])
            if r.get("status") == "awaiting_human"
        ]
        result["status"] = "interrupted"
        result["thread_id"] = thread_id
        result["escalations"] = escalations

    return result


@app.post("/agent/resume")
def agent_resume_endpoint(
    payload: ResumeRequest,
    tenant: Tenant = Depends(get_current_tenant),
):
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
