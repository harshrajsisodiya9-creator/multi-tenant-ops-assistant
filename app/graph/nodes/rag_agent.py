from __future__ import annotations

from app.graph.state import AgentState
from app.hybrid_search import answer_question as rag_answer


def rag_agent_node(state: AgentState) -> dict:
    from uuid import UUID

    from sqlalchemy.orm import Session as SASession

    from app.database import SessionLocal

    db: SASession = SessionLocal()
    try:
        result = rag_answer(
            db=db,
            tenant_id=UUID(state["tenant_id"]),
            question=state["user_message"],
        )
        return {"response": result.get("answer", "I couldn't find an answer.")}
    finally:
        db.close()
