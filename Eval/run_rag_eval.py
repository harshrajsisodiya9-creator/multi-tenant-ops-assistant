"""Golden-dataset evaluation runner for the FAQ RAG pipeline.

Runs every golden question through the production pipeline
(app.hybrid_search.answer_question: vector + BM25 -> RRF -> cross-encoder ->
Groq generation) without changing any of its parameters. It records the ranked
chunks and the generated answer, computes retrieval and generation metrics
separately, and writes a JSON report.

Usage (Postgres running, KB ingested for the tenant):

    python -m Eval.run_rag_eval                    # deterministic metrics only
    python -m Eval.run_rag_eval --judge            # + LLM judge (see Eval/judge.py)
    python -m Eval.run_rag_eval --limit 3          # smoke run

Exit codes: 0 = all items evaluated; 1 = at least one item (or judge call)
failed, and the failures are listed in the report; 2 = the dataset or the
pre-flight check is invalid, so nothing was evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from Eval import metrics as M
from Eval.golden import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_DATASET,
    KB_DIR,
    PROJECT_DIR,
    GoldenDatasetError,
    GoldenItem,
    load_golden,
    sha256_file,
    unresolved_evidence,
)

# Same tenant Eval/eval.py has always evaluated against ("Test Tenant").
DEFAULT_TENANT_ID = "f67e4814-c4bf-421b-94d8-0d5894703816"
DEFAULT_K_VALUES = (1, 3, 5)
RAG_MAX_ATTEMPTS = 3
RATE_LIMIT_BACKOFF_SECONDS = 10
FALSE_ABSTENTION_KEY_FACT_THRESHOLD = 0.5

RagFn = Callable[[str], dict]


# ---------------------------------------------------------------------------
# Per-item evaluation (pure: no DB, no network)
# ---------------------------------------------------------------------------


def evaluate_retrieval(
    item: GoldenItem,
    top_matches: Sequence[dict],
    corpus: Sequence[tuple[str, str]],
    k_values: Sequence[int],
) -> tuple[list[dict], dict]:
    evidence = item.evidence
    ranked = []
    for rank, m in enumerate(top_matches, start=1):
        hits = M.matched_evidence(m["source_document_id"], m["content"], evidence)
        ranked.append(
            {
                "rank": rank,
                "chunk_id": m.get("chunk_id"),
                "source_document_id": m["source_document_id"],
                "chunk_index": m.get("chunk_index"),
                "grade": max((evidence[i]["grade"] for i in hits), default=0),
                "matched_evidence": hits,
                "content": m["content"],
            }
        )

    required = M.required_evidence_indices(evidence)
    if not required:
        return ranked, {
            "applicable": False,
            "reason": "no grade-2 evidence exists for this question (expected abstention)",
        }

    unresolved = [
        evidence[i]["quote"]
        for i in required
        if not any(i in M.matched_evidence(doc, content, evidence) for doc, content in corpus)
    ]
    if unresolved:
        return ranked, {
            "applicable": False,
            "reason": "required evidence not found in indexed corpus",
            "unresolved_evidence": unresolved,
        }

    hits_per_rank = [r["matched_evidence"] for r in ranked]
    grades = [r["grade"] for r in ranked]
    corpus_grades = [M.chunk_grade(doc, content, evidence) for doc, content in corpus]
    covered = {i for hits in hits_per_rank for i in hits}
    out = {
        "applicable": True,
        "required_evidence_count": len(required),
        "relevant_chunks_in_corpus": sum(g >= M.REQUIRED_GRADE for g in corpus_grades),
        "mrr": M.reciprocal_rank(grades),
        "missing_required_evidence": [evidence[i]["quote"] for i in required if i not in covered],
        "retrieved_source_documents": [r["source_document_id"] for r in ranked],
    }
    for k in k_values:
        out[f"recall@{k}"] = M.recall_at_k(hits_per_rank, required, k)
        out[f"complete_evidence@{k}"] = M.complete_evidence_at_k(hits_per_rank, required, k)
        out[f"ndcg@{k}"] = M.ndcg_at_k(grades, corpus_grades, k)
    return ranked, out


def evaluate_generation(item: GoldenItem, answer: str, context: str) -> dict:
    kfr, missing = M.key_fact_recall(answer, item.key_facts)
    grounding, unsupported = M.numeric_grounding(answer, context, item.question)
    abstained = M.detect_abstention(answer)
    if item.expected_behavior == "abstain":
        abstention_correct, false_abstention = abstained, None
    else:
        abstention_correct = None
        false_abstention = abstained and (kfr is None or kfr < FALSE_ABSTENTION_KEY_FACT_THRESHOLD)
    return {
        "key_fact_recall": kfr,
        "missing_key_facts": [list(item.key_facts[i]) for i in missing],
        "abstained": abstained,
        "abstention_correct": abstention_correct,
        "false_abstention": false_abstention,
        "numeric_grounding": grounding,
        "unsupported_numbers": unsupported,
    }


def build_context(top_matches: Sequence[dict]) -> str:
    # Mirrors app.hybrid_search._construct_context, which joins chunk content
    # with blank lines; that is the exact string the generator was given.
    return "\n\n".join(m["content"] for m in top_matches)


def _is_rate_limit(error: Exception) -> bool:
    msg = str(error).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


def call_with_retry(fn: Callable[[], dict], sleep=time.sleep) -> dict:
    for attempt in range(1, RAG_MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as error:
            if not _is_rate_limit(error) or attempt == RAG_MAX_ATTEMPTS:
                raise
            wait = RATE_LIMIT_BACKOFF_SECONDS * attempt
            print(f"  rate limited, retrying in {wait}s ({attempt}/{RAG_MAX_ATTEMPTS})", flush=True)
            sleep(wait)
    raise AssertionError("unreachable")


def _error(stage: str, error: Exception) -> dict:
    return {"stage": stage, "type": type(error).__name__, "message": str(error)}


def evaluate_items(
    items: Sequence[GoldenItem],
    rag_fn: RagFn,
    corpus: Sequence[tuple[str, str]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    judge=None,
    sleep=time.sleep,
) -> list[dict]:
    results = []
    for n, item in enumerate(items, start=1):
        print(f"[{n}/{len(items)}] {item.id} ({item.category})", flush=True)
        result = {
            "id": item.id,
            "question": item.question,
            "reference_answer": item.answer,
            "category": item.category,
            "secondary_categories": list(item.secondary_categories),
            "expected_behavior": item.expected_behavior,
            "status": "ok",
            "error": None,
            "latency_seconds": None,
            "generated_answer": None,
            "retrieved": [],
            "retrieval": None,
            "generation": None,
            "judge": None,
        }
        started = time.perf_counter()
        try:
            response = call_with_retry(lambda: rag_fn(item.question), sleep=sleep)
            result["latency_seconds"] = round(time.perf_counter() - started, 3)
            top_matches = response.get("top_matches") or []
            answer = response.get("answer")
            if not isinstance(answer, str):
                raise TypeError(f"pipeline returned non-string answer: {answer!r}")
            result["generated_answer"] = answer
            context = build_context(top_matches)
            result["retrieved"], result["retrieval"] = evaluate_retrieval(item, top_matches, corpus, k_values)
            result["generation"] = evaluate_generation(item, answer, context)
        except Exception as error:  # recorded, never skipped
            result["status"] = "error"
            result["error"] = _error("rag", error)
            print(f"  FAILED: {type(error).__name__}: {error}", flush=True)
            results.append(result)
            continue

        if judge is not None:
            verdict: dict = {}
            try:
                verdict["correctness"] = call_with_retry(
                    lambda: judge.correctness(item.question, item.answer, answer), sleep=sleep
                )
                verdict["faithfulness"] = call_with_retry(
                    lambda: judge.faithfulness(item.question, context, answer), sleep=sleep
                )
            except Exception as error:
                verdict["error"] = _error("judge", error)
                print(f"  JUDGE FAILED: {type(error).__name__}: {error}", flush=True)
            result["judge"] = verdict
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _agg(values) -> dict:
    vals = [v for v in values if v is not None]
    return {"mean": round(sum(vals) / len(vals), 4) if vals else None, "n": len(vals)}


def summarize(results: Sequence[dict], k_values: Sequence[int]) -> dict:
    ok = [r for r in results if r["status"] == "ok"]
    retr = [r for r in ok if r["retrieval"] and r["retrieval"]["applicable"]]
    answerable = [r for r in ok if r["expected_behavior"] == "answer"]
    abstain = [r for r in ok if r["expected_behavior"] == "abstain"]
    judged = [r for r in ok if r["judge"] is not None]

    retrieval = {
        "items_evaluated": len(retr),
        "items_not_applicable": [
            {"id": r["id"], "reason": r["retrieval"]["reason"]}
            for r in ok
            if r["retrieval"] and not r["retrieval"]["applicable"]
        ],
        "mrr": _agg(r["retrieval"]["mrr"] for r in retr),
    }
    for k in k_values:
        for name in ("recall", "complete_evidence", "ndcg"):
            retrieval[f"{name}@{k}"] = _agg(r["retrieval"][f"{name}@{k}"] for r in retr)

    def judge_vals(rows, metric):
        return [
            r["judge"][metric]["normalized" if metric == "correctness" else "score"]
            for r in rows
            if r["judge"] and metric in r["judge"]
        ]

    false_abst = [r["id"] for r in answerable if r["generation"]["false_abstention"]]
    correct_abst = [r for r in abstain if r["generation"]["abstention_correct"]]
    generation = {
        "key_fact_recall_answerable": _agg(r["generation"]["key_fact_recall"] for r in answerable),
        "numeric_grounding": _agg(r["generation"]["numeric_grounding"] for r in ok),
        "abstention": {
            "expected_abstentions": len(abstain),
            "correct_abstentions": len(correct_abst),
            "abstention_accuracy": round(len(correct_abst) / len(abstain), 4) if abstain else None,
            "missed_abstention_ids": [r["id"] for r in abstain if not r["generation"]["abstention_correct"]],
            "answerable_items": len(answerable),
            "false_abstentions": len(false_abst),
            "false_abstention_ids": false_abst,
        },
    }
    judge = {
        "items_judged": len(judged),
        "errors": [r["id"] for r in judged if "error" in r["judge"]],
        "correctness": _agg(judge_vals(judged, "correctness")),
        "faithfulness": _agg(judge_vals(judged, "faithfulness")),
    }

    by_category: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in results}):
        rows = [r for r in ok if r["category"] == cat]
        rrows = [r for r in rows if r["retrieval"] and r["retrieval"]["applicable"]]
        kmax = max(k_values)
        by_category[cat] = {
            "items": sum(r["category"] == cat for r in results),
            "succeeded": len(rows),
            f"recall@{kmax}": _agg(r["retrieval"][f"recall@{kmax}"] for r in rrows),
            "mrr": _agg(r["retrieval"]["mrr"] for r in rrows),
            f"ndcg@{kmax}": _agg(r["retrieval"][f"ndcg@{kmax}"] for r in rrows),
            "key_fact_recall": _agg(r["generation"]["key_fact_recall"] for r in rows),
            "judge_correctness": _agg(judge_vals(rows, "correctness")),
            "judge_faithfulness": _agg(judge_vals(rows, "faithfulness")),
        }

    return {
        "items_total": len(results),
        "items_succeeded": len(ok),
        "items_failed": len(results) - len(ok),
        "failed_ids": [r["id"] for r in results if r["status"] != "ok"],
        "retrieval": retrieval,
        "generation": generation,
        "judge": judge,
        "by_category": by_category,
    }


# ---------------------------------------------------------------------------
# Run metadata / reproducibility
# ---------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_DIR, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def corpus_fingerprint(corpus_rows: Sequence[tuple[str, int, str]]) -> dict:
    """Hash of the indexed corpus plus a check against the source files.

    ``matches_source_files`` re-chunks files/<doc>.md with the production
    splitter and compares byte-for-byte, so drift between the KB files and the
    database index is visible in the report.
    """
    from app.knowledgebase import split_text_into_chunks

    h = hashlib.sha256()
    per_doc: dict[str, dict[int, str]] = {}
    for doc, idx, content in sorted(corpus_rows):
        h.update(f"{doc}\x00{idx}\x00{content}\x01".encode())
        per_doc.setdefault(doc, {})[idx] = content
    matches = {}
    for doc, chunks in per_doc.items():
        path = KB_DIR / f"{doc}.md"
        if not path.exists():
            matches[doc] = None
            continue
        expected = split_text_into_chunks(path.read_text(encoding="utf-8"))
        matches[doc] = expected == [chunks.get(i) for i in range(len(chunks))]
    return {
        "chunk_count": len(corpus_rows),
        "chunks_per_document": {d: len(c) for d, c in sorted(per_doc.items())},
        "corpus_sha256": h.hexdigest(),
        "matches_source_files": matches,
    }


def pipeline_config() -> dict:
    import app.hybrid_search as hs
    import app.knowledgebase as kb
    from app.config import setting

    return {
        "pipeline": "app.hybrid_search.answer_question",
        "pipeline_version": hs._RAG_PIPELINE_VERSION,
        "top_k": hs.DEFAULT_LIMIT,
        "vector_candidates": hs.VECTOR_CANDIDATES,
        "bm25_candidates": hs.BM25_CANDIDATES,
        "rrf_k": hs.RRF_K,
        "rerank_pool": hs.RERANK_POOL,
        "rerank_model": hs.RERANK_MODEL_NAME,
        "embedding_model": kb.EMBEDDING_MODEL_NAME,
        "chunk_size": kb.CHUNK_SIZE,
        "chunk_overlap": kb.CHUNK_OVERLAP,
        "generator_provider": "groq",
        "generator_model": setting.groq_model,
        "generator_temperature": 0,  # hard-coded in app.knowledgebase.generate_answer
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_summary(summary: dict, k_values: Sequence[int]) -> None:
    def f(agg):
        return "n/a" if agg["mean"] is None else f"{agg['mean']:.3f} (n={agg['n']})"

    s, r, g, j = summary, summary["retrieval"], summary["generation"], summary["judge"]
    print("\n=== RAG evaluation summary ===")
    print(f"items: {s['items_total']}  succeeded: {s['items_succeeded']}  failed: {s['items_failed']} {s['failed_ids'] or ''}")
    print(f"retrieval (n={r['items_evaluated']}, not applicable: {[x['id'] for x in r['items_not_applicable']]})")
    for k in k_values:
        print(f"  recall@{k}: {f(r[f'recall@{k}'])}   complete@{k}: {f(r[f'complete_evidence@{k}'])}   ndcg@{k}: {f(r[f'ndcg@{k}'])}")
    print(f"  mrr: {f(r['mrr'])}")
    a = g["abstention"]
    print("generation")
    print(f"  key_fact_recall (answerable): {f(g['key_fact_recall_answerable'])}")
    print(f"  numeric_grounding: {f(g['numeric_grounding'])}")
    print(f"  abstention: {a['correct_abstentions']}/{a['expected_abstentions']} correct, missed {a['missed_abstention_ids']}")
    print(f"  false abstentions: {a['false_abstentions']}/{a['answerable_items']} {a['false_abstention_ids']}")
    if j["items_judged"]:
        print(f"judge: correctness {f(j['correctness'])}  faithfulness {f(j['faithfulness'])}  errors {j['errors']}")
    else:
        print("judge: not run (pass --judge)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the golden dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--tenant-id", default=DEFAULT_TENANT_ID)
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K_VALUES), help="K values for @K metrics")
    parser.add_argument("--limit", type=int, help="evaluate only the first N items (smoke run)")
    parser.add_argument("--judge", action="store_true", help="also run the LLM judge (Eval/judge.py)")
    parser.add_argument("--output", type=Path, help="report path (default Eval/reports/rag_eval_<UTC>.json)")
    parser.add_argument(
        "--allow-unresolved-evidence",
        action="store_true",
        help="continue when annotated evidence is missing from the index (affected items are excluded from retrieval metrics and listed)",
    )
    args = parser.parse_args(argv)

    try:
        items = load_golden(args.dataset, args.annotations, expected_size=None if args.limit else 20)
    except GoldenDatasetError as error:
        print(error, file=sys.stderr)
        return 2
    if args.limit:
        items = items[: args.limit]

    import app.hybrid_search as hs
    from sqlalchemy import select

    from app.database import SessionLocal
    from app.models import Chunk

    k_values = sorted(set(args.k))
    if max(k_values) > hs.DEFAULT_LIMIT or min(k_values) < 1:
        print(f"--k values must be within 1..{hs.DEFAULT_LIMIT} (the pipeline's top_k)", file=sys.stderr)
        return 2

    tenant_id = UUID(args.tenant_id)
    db = SessionLocal()
    try:
        rows = db.execute(
            select(Chunk.source_document_id, Chunk.chunk_index, Chunk.content).where(Chunk.tenant_id == tenant_id)
        ).all()
        corpus_rows = [(r.source_document_id, r.chunk_index, r.content) for r in rows]
        if not corpus_rows:
            print(f"No chunks indexed for tenant {tenant_id}; ingest the KB first.", file=sys.stderr)
            return 2
        corpus = [(doc, content) for doc, _idx, content in corpus_rows]
        unresolved = unresolved_evidence(items, corpus)
        if unresolved and not args.allow_unresolved_evidence:
            print("Pre-flight failed: annotated evidence not found in the indexed corpus:", file=sys.stderr)
            for p in unresolved:
                print(f"  - {p}", file=sys.stderr)
            print("Re-ingest the KB, fix the annotations, or pass --allow-unresolved-evidence.", file=sys.stderr)
            return 2

        judge = None
        judge_config = {"enabled": False}
        if args.judge:
            from app.config import setting
            from Eval.judge import GroqJudge

            judge = GroqJudge(model=setting.groq_judge_model, api_key=setting.groq_api_key)
            judge_config = judge.config()

        started = datetime.now(timezone.utc)
        results = evaluate_items(
            items,
            rag_fn=lambda q: hs.answer_question(db=db, tenant_id=tenant_id, question=q),
            corpus=corpus,
            k_values=k_values,
            judge=judge,
        )
    finally:
        db.close()

    summary = summarize(results, k_values)
    kb_files = sorted(KB_DIR.glob("*.md")) if KB_DIR.exists() else []
    report = {
        "report_schema_version": 1,
        "run": {
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python": platform.python_version(),
            "tenant_id": str(tenant_id),
            "dataset": str(args.dataset.relative_to(PROJECT_DIR) if args.dataset.is_relative_to(PROJECT_DIR) else args.dataset),
            "dataset_sha256": sha256_file(args.dataset),
            "annotations_sha256": sha256_file(args.annotations),
            "limit": args.limit,
            "unresolved_evidence": unresolved,
            "kb_documents_sha256": {p.stem: sha256_file(p) for p in kb_files},
            "corpus": corpus_fingerprint(corpus_rows),
        },
        "config": {
            "retrieval_pipeline": pipeline_config(),
            "metrics": {
                "k_values": k_values,
                "required_grade": M.REQUIRED_GRADE,
                "ndcg_gain": "2^grade - 1, log2 discount, ideal DCG over the full indexed corpus",
                "false_abstention_key_fact_threshold": FALSE_ABSTENTION_KEY_FACT_THRESHOLD,
                "rag_max_attempts": RAG_MAX_ATTEMPTS,
                "rate_limit_backoff_seconds": RATE_LIMIT_BACKOFF_SECONDS,
            },
            "judge": judge_config,
        },
        "summary": summary,
        "items": results,
    }

    output = args.output or (
        PROJECT_DIR / "Eval" / "reports" / f"rag_eval_{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    _print_summary(summary, k_values)
    print(f"\nreport: {output}")
    failed = summary["items_failed"] or summary["judge"]["errors"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
