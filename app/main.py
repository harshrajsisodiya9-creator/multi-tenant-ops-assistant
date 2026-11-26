from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
agent_graph = build_graph()

_THREAD_OWNERS: dict[str, tuple[str, str]] = {}
_THREAD_OWNERS_GUARD = threading.Lock()

THREAD_AWAITING_APPROVAL = {
    "code": "THREAD_AWAITING_APPROVAL",
    "message": "This conversation has a pending human approval. Please wait for it to be resolved.",
}
THREAD_NOT_AWAITING_APPROVAL = {
    "code": "THREAD_NOT_AWAITING_APPROVAL",
    "message": "This conversation has no pending human approval to resolve.",
}
# Same response for unknown and foreign threads so callers cannot probe which thread IDs exist.
THREAD_NOT_ACCESSIBLE = {
    "code": "THREAD_NOT_ACCESSIBLE",
    "message": "Thread not found or not accessible.",
}

APPROVAL_NODE = "escalation_handler"


def _get_thread_owner(thread_id: str) -> tuple[str, str] | None:
    with _THREAD_OWNERS_GUARD:
        return _THREAD_OWNERS.get(thread_id)


def _register_thread(thread_id: str, owner: tuple[str, str]) -> bool:
    with _THREAD_OWNERS_GUARD:
        if thread_id in _THREAD_OWNERS:
            return False
        _THREAD_OWNERS[thread_id] = owner
        return True


def _create_thread(owner: tuple[str, str]) -> str:
    while True:
        thread_id = uuid4().hex
        if _register_thread(thread_id, owner):
            return thread_id


# Per-thread locks are in-process only: they do not coordinate separate workers/processes.
@dataclass
class _ThreadLockEntry:
    lock: Any
    users: int = 0


_THREAD_LOCKS: dict[str, _ThreadLockEntry] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
_new_thread_lock = threading.Lock


@contextmanager
def _thread_lock(thread_id: str) -> Iterator[None]:
    # users counts holders and waiters, so an entry is only dropped when nobody can still be using it.
    with _THREAD_LOCKS_GUARD:
        entry = _THREAD_LOCKS.get(thread_id)
        if entry is None:
            entry = _THREAD_LOCKS[thread_id] = _ThreadLockEntry(lock=_new_thread_lock())
        entry.users += 1
    try:
        with entry.lock:
            yield
    finally:
        with _THREAD_LOCKS_GUARD:
            entry.users -= 1
            if entry.users == 0:
                del _THREAD_LOCKS[thread_id]


def _pending_interrupt_nodes(config: dict) -> set[str]:
    snapshot = agent_graph.get_state(config)
    return {task.name for task in snapshot.tasks if task.interrupts}


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


@app.get("/health", include_in_schema=False)
def health_check():
    return {"status": "ok"}


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
    caller = (claims.tenant_id, claims.customer_id)
    is_new_thread = not payload.thread_id
    thread_id = _create_thread(caller) if is_new_thread else payload.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    # One lock spans validation, the pending check and the whole invocation, so no other
    # request on this thread can interrupt, resume or start a turn in between.
    with _thread_lock(thread_id):
        if not is_new_thread:
            if _get_thread_owner(thread_id) != caller:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=THREAD_NOT_ACCESSIBLE,
                )
            # A new turn would restart the graph and abandon the interrupted approval.
            if _pending_interrupt_nodes(config):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=THREAD_AWAITING_APPROVAL,
                )

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
            r
            for r in (state.get("intent_results") or [])
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
    thread_id = payload.thread_id
    config = {"configurable": {"thread_id": thread_id}}
    decisions = [d.model_dump() for d in payload.decisions]

    with _thread_lock(thread_id):
        owner = _get_thread_owner(thread_id)
        if owner is None or owner[0] != claims.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=THREAD_NOT_ACCESSIBLE,
            )
        # Resuming an idle thread runs nothing and would return the last turn's results
        # as if this decision had been applied.
        if APPROVAL_NODE not in _pending_interrupt_nodes(config):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=THREAD_NOT_AWAITING_APPROVAL,
            )

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
