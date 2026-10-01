from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import setting
from app.knowledgebase import embed_text, generate_answer
from app.models import Chunk

RRF_K = 60
DEFAULT_LIMIT = 5
VECTOR_CANDIDATES = 20
BM25_CANDIDATES = 20


def _vector_search(
    db: Session, tenant_id: UUID, query_embedding: list[float], limit: int
) -> list[tuple[int, float]]:
    stmt = (
        select(
            Chunk.id,
            Chunk.embedding.cosine_distance(query_embedding).label("distance"),
        )
        .where(Chunk.tenant_id == tenant_id)
        .order_by("distance")
        .limit(limit)
    )
    return [(row.id, row.distance) for row in db.execute(stmt).all()]


def _bm25_search(
    db: Session, tenant_id: UUID, question: str, limit: int
) -> list[tuple[int, float]]:
    query = func.plainto_tsquery("english", question)
    rank = func.ts_rank(Chunk.content_tsv, query).label("rank")

    stmt = (
        select(Chunk.id, rank)
        .where(Chunk.tenant_id == tenant_id)
        .where(Chunk.content_tsv.op("@@")(query))
        .order_by(rank.desc())
        .limit(limit)
    )
    results = [(row.id, row.rank) for row in db.execute(stmt).all()]

    if not results:
        similarity = func.similarity(Chunk.content, question).label("sim")
        stmt = (
            select(Chunk.id, similarity)
            .where(Chunk.tenant_id == tenant_id)
            .where(similarity > 0.05)
            .order_by(similarity.desc())
            .limit(limit)
        )
        results = [(row.id, row.sim) for row in db.execute(stmt).all()]

    return results


def _rrf_fuse(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = RRF_K,
) -> list[int]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for position, (doc_id, _score) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1])]


def search_hybrid(
    db: Session,
    tenant_id: UUID,
    question: str,
    limit: int = DEFAULT_LIMIT,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[Chunk]:
    if not question or not question.strip():
        return []

    query_embedding = embed_text(question)

    vector_results = _vector_search(db, tenant_id, query_embedding, VECTOR_CANDIDATES)
    bm25_results = _bm25_search(db, tenant_id, question, BM25_CANDIDATES)

    fused_ids = _rrf_fuse([vector_results, bm25_results])[:limit]

    if not fused_ids:
        return []

    chunks_by_id = {
        c.id: c
        for c in db.execute(
            select(Chunk).where(Chunk.id.in_(fused_ids))
        ).scalars().all()
    }
    return [chunks_by_id[cid] for cid in fused_ids if cid in chunks_by_id]


def answer_question(
    db: Session, tenant_id: UUID, question: str, limit: int = DEFAULT_LIMIT
) -> dict:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    matches = search_hybrid(db=db, tenant_id=tenant_id, question=question, limit=limit)

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
