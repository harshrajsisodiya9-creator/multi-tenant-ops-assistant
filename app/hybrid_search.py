from __future__ import annotations

from uuid import UUID

from fastembed.rerank.cross_encoder import TextCrossEncoder
from langsmith import traceable
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.knowledgebase import embed_text, generate_answer
from app.models import Chunk

RRF_K = 60
DEFAULT_LIMIT = 5
VECTOR_CANDIDATES = 40
BM25_CANDIDATES = 40
RERANK_POOL = 35

RERANK_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
_reranker: TextCrossEncoder | None = None

_RAG_PIPELINE_VERSION = "1.0"


def _log_trace_metadata(metadata: dict) -> None:
    try:
        from langsmith.run_helpers import get_current_run_tree
        rt = get_current_run_tree()
        if rt is not None:
            rt.metadata = {**(rt.metadata or {}), **metadata}
    except Exception:
        pass


def get_reranker() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(model_name=RERANK_MODEL_NAME)
    return _reranker


@traceable(name="vector_search", run_type="retriever")
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
    results = [(row.id, row.distance) for row in db.execute(stmt).all()]

    _log_trace_metadata({
        "retrieved_count": len(results),
        "document_ids": [doc_id for doc_id, _ in results],
        "similarity_scores": [
            {"chunk_id": doc_id, "cosine_distance": round(float(dist), 6)}
            for doc_id, dist in results
        ],
    })

    return results


@traceable(name="bm25_search", run_type="retriever")
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

    used_trigram_fallback = False
    if not results:
        used_trigram_fallback = True
        similarity = func.similarity(Chunk.content, question).label("sim")
        stmt = (
            select(Chunk.id, similarity)
            .where(Chunk.tenant_id == tenant_id)
            .where(similarity > 0.05)
            .order_by(similarity.desc())
            .limit(limit)
        )
        results = [(row.id, row.sim) for row in db.execute(stmt).all()]

    _log_trace_metadata({
        "search_query": question,
        "retrieved_count": len(results),
        "used_trigram_fallback": used_trigram_fallback,
        "document_ids": [doc_id for doc_id, _ in results],
        "bm25_scores": [
            {"chunk_id": doc_id, "rank": position, "score": round(float(score), 6)}
            for position, (doc_id, score) in enumerate(results)
        ],
    })

    return results


@traceable(name="reciprocal_rank_fusion", run_type="chain")
def _rrf_fuse(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = RRF_K,
) -> list[int]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for position, (doc_id, _score) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position + 1)
    sorted_items = sorted(scores.items(), key=lambda x: -x[1])

    _log_trace_metadata({
        "vector_input_ids": [doc_id for doc_id, _ in ranked_lists[0]] if ranked_lists else [],
        "bm25_input_ids": [doc_id for doc_id, _ in ranked_lists[1]] if len(ranked_lists) > 1 else [],
        "unique_documents": len(scores),
        "fusion_scores": [
            {"chunk_id": doc_id, "rrf_score": round(score, 6)}
            for doc_id, score in sorted_items
        ],
    })

    return [doc_id for doc_id, _ in sorted_items]


@traceable(name="cross_encoder_reranking", run_type="chain")
def _cross_encoder_rerank(
    question: str, pool: list[Chunk], limit: int
) -> list[Chunk]:
    reranker = get_reranker()
    docs = [chunk.content for chunk in pool]
    scores = list(reranker.rerank(question, docs))
    ranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)
    selected = [chunk for chunk, _score in ranked[:limit]]

    _log_trace_metadata({
        "input_candidate_count": len(pool),
        "output_count": len(selected),
        "reranking_scores": [
            {"chunk_id": chunk.id, "score": round(float(score), 6), "rank": i}
            for i, (chunk, score) in enumerate(ranked)
        ],
        "selected_ids": [c.id for c in selected],
        "discarded_ids": [chunk.id for chunk, _ in ranked[limit:]],
        "rerank_model": RERANK_MODEL_NAME,
    })

    return selected


@traceable(name="context_construction", run_type="chain")
def _construct_context(matches: list[Chunk]) -> str:
    context = "\n\n".join(chunk.content for chunk in matches)

    _log_trace_metadata({
        "num_chunks": len(matches),
        "chunk_ids": [chunk.id for chunk in matches],
        "source_document_ids": [chunk.source_document_id for chunk in matches],
        "context_char_length": len(context),
        "truncated": False,
    })

    return context


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

    fused_ids = _rrf_fuse([vector_results, bm25_results])[:RERANK_POOL]

    if not fused_ids:
        return []

    chunks_by_id = {
        c.id: c
        for c in db.execute(
            select(Chunk).where(Chunk.id.in_(fused_ids))
        ).scalars().all()
    }
    pool = [chunks_by_id[cid] for cid in fused_ids if cid in chunks_by_id]

    if len(pool) <= limit:
        return pool

    return _cross_encoder_rerank(question, pool, limit)


@traceable(name="faq_rag_pipeline", run_type="chain", tags=["faq", "rag"])
def answer_question(
    db: Session, tenant_id: UUID, question: str, limit: int = DEFAULT_LIMIT
) -> dict:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    from app.config import setting
    _log_trace_metadata({
        "tenant_id": str(tenant_id),
        "pipeline": "faq_rag",
        "version": _RAG_PIPELINE_VERSION,
        "environment": setting.app_env,
    })

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

    context = _construct_context(matches)
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
