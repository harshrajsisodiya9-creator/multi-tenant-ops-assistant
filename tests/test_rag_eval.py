"""Tests for the golden RAG evaluation (Eval/golden*.json, metrics, judge, runner).

No database, network or LLM is used. The KB-dependent tests read files/,
which is gitignored, so they skip when it is absent (e.g. in CI).
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest

from Eval import metrics as M
from Eval.golden import (
    CATEGORIES,
    DEFAULT_ANNOTATIONS,
    DEFAULT_DATASET,
    KB_DIR,
    PROJECT_DIR,
    GoldenDatasetError,
    GoldenItem,
    load_golden,
    unresolved_evidence,
    validate,
)
from Eval.judge import JudgeParseError, parse_correctness, parse_faithfulness, prompt_fingerprint
from Eval.run_rag_eval import evaluate_items, summarize

requires_kb = pytest.mark.skipif(not KB_DIR.exists(), reason="files/ (gitignored KB) not present")


# ---------------------------------------------------------------------------
# Golden dataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def items() -> list[GoldenItem]:
    return load_golden()


def test_dataset_has_exactly_20_question_answer_pairs():
    data = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
    assert len(data) == 20
    assert all(set(d) == {"question", "answer"} for d in data)
    assert len({d["question"] for d in data}) == 20


def test_annotations_align_and_validate(items):
    assert [i.id for i in items] == [f"q{n:02d}" for n in range(1, 21)]


def test_all_categories_covered_with_reasonable_distribution(items):
    counts = Counter(i.category for i in items)
    assert set(counts) == set(CATEGORIES)
    assert all(3 <= c <= 5 for c in counts.values()), counts


def test_abstention_items_have_no_required_evidence_except_partial_context(items):
    for i in items:
        if i.expected_behavior == "abstain":
            required = M.required_evidence_indices(i.evidence)
            assert not required or "partial_context" in i.secondary_categories, i.id


def test_unmodified_items_are_verbatim_from_original_dataset(items):
    """Items whose origin is the current Eval/eval_dataset.json and are marked
    unmodified must match it exactly."""
    original = json.loads((PROJECT_DIR / "Eval" / "eval_dataset.json").read_text(encoding="utf-8"))
    checked = 0
    for i in items:
        o = i.origin
        if o["commit"] == "d7fea7f":
            src = original[o["index"] - 1]
            assert src["question"] == i.question, i.id
            if not o["answer_modified"]:
                assert src["answer"] == i.answer, i.id
            else:
                assert o["original_answer"] == src["answer"] and o["modification_reason"], i.id
            checked += 1
    assert checked >= 10


def test_reference_answers_pass_their_own_deterministic_checks(items):
    """Calibration: a reference answer must cover all its key facts, and only
    abstention references may trigger the abstention detector."""
    for i in items:
        score, missing = M.key_fact_recall(i.answer, i.key_facts)
        assert score in (None, 1.0), (i.id, [i.key_facts[m] for m in missing])
        assert M.detect_abstention(i.answer) is (i.expected_behavior == "abstain"), i.id


def test_original_dataset_left_untouched():
    data = json.loads((PROJECT_DIR / "Eval" / "eval_dataset.json").read_text(encoding="utf-8"))
    assert len(data) == 21


def test_validate_rejects_extra_fields_and_misalignment():
    ds = [{"question": "q", "answer": "a", "category": "x"}]
    ann = [{"id": "q01", "question": "q", "category": "direct_factual",
            "expected_behavior": "answer", "evidence": [], "key_facts": []}]
    with pytest.raises(GoldenDatasetError) as exc:
        validate(ds, ann, expected_size=1)
    assert "exactly the keys" in str(exc.value)

    ds = [{"question": "q", "answer": "a"}]
    ann[0]["question"] = "different"
    with pytest.raises(GoldenDatasetError) as exc:
        validate(ds, ann, expected_size=1)
    msg = str(exc.value)
    assert "does not match" in msg and "grade-2 evidence" in msg and "key_facts" in msg


@requires_kb
def test_every_evidence_quote_is_inside_one_production_chunk(items):
    """Quotes must survive the production chunker intact; otherwise no chunk can
    ever 'contain' them and Recall@K would silently be 0."""
    from app.knowledgebase import split_text_into_chunks

    corpus = [
        (p.stem, chunk)
        for p in sorted(KB_DIR.glob("*.md"))
        for chunk in split_text_into_chunks(p.read_text(encoding="utf-8"))
    ]
    assert unresolved_evidence(items, corpus) == []


@requires_kb
def test_superseded_and_non_authoritative_docs_are_labelled_consistently(items):
    roles = {"refund_policy_v1": "superseded", "support_internal_faq": "non_authoritative"}
    for i in items:
        for e in i.evidence:
            expected = roles.get(e["source_document_id"], "authoritative")
            assert e["role"] == expected, (i.id, e["source_document_id"])
            assert (KB_DIR / f"{e['source_document_id']}.md").exists()


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

EVIDENCE = [
    {"source_document_id": "a", "quote": "alpha  fact", "grade": 2},
    {"source_document_id": "b", "quote": "beta fact", "grade": 2},
    {"source_document_id": "c", "quote": "faq restatement", "grade": 1},
]


def test_chunk_grade_requires_matching_document_and_normalises_whitespace():
    assert M.matched_evidence("a", "x alpha\nfact y", EVIDENCE) == [0]
    assert M.chunk_grade("b", "alpha fact", EVIDENCE) == 0  # right text, wrong doc
    assert M.chunk_grade("c", "a faq restatement", EVIDENCE) == 1


def test_recall_and_complete_evidence():
    hits = [[0], [], [1, 0]]
    assert M.recall_at_k(hits, [0, 1], 1) == 0.5
    assert M.recall_at_k(hits, [0, 1], 3) == 1.0
    assert M.complete_evidence_at_k(hits, [0, 1], 2) == 0.0
    assert M.complete_evidence_at_k(hits, [0, 1], 3) == 1.0
    assert M.recall_at_k(hits, [], 3) is None


def test_reciprocal_rank_ignores_grade_one():
    assert M.reciprocal_rank([1, 0, 2]) == pytest.approx(1 / 3)
    assert M.reciprocal_rank([1, 1]) == 0.0


def test_ndcg_hand_computed():
    retrieved = [0, 2, 1]
    corpus = [2, 1, 0, 0]
    dcg = 3 / math.log2(3) + 1 / math.log2(4)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert M.ndcg_at_k(retrieved, corpus, 3) == pytest.approx(dcg / idcg)
    assert M.ndcg_at_k([2, 1], [2, 1, 0], 2) == pytest.approx(1.0)
    assert M.ndcg_at_k([0], [0, 0], 1) is None


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,phrase,expected",
    [
        ("free over $35 before tax", "$35", True),
        ("free over $350", "$35", False),
        ("costs $35.99", "$35", False),
        ("a fee of $4.99.", "$4.99", True),
        ("after 121 days", "21", False),
        ("after 21 consecutive business days", "21 consecutive", True),
        ("reviewed by support agents", "agent", True),
        ("a 10% fee", "10%", True),
        ("MANAGER approval", "manager", True),
    ],
)
def test_contains_phrase_boundaries(text, phrase, expected):
    assert M.contains_phrase(text, phrase) is expected


def test_key_fact_recall():
    facts = [["$200"], ["manager"], ["superseded", "previous version"]]
    score, missing = M.key_fact_recall("Refunds over $200 need manager approval.", facts)
    assert score == pytest.approx(2 / 3) and missing == [2]
    assert M.key_fact_recall("anything", []) == (None, [])


@pytest.mark.parametrize(
    "answer",
    [
        "I couldn't find enough relevant context to answer that question.",
        "The provided context does not specify which payment methods are accepted.",
        "This information is not available in the policy documents.",
        "The policies don't mention prepaid return labels.",
        "Your options cannot be determined from the policies.",
        # Observed baseline outputs the first version of the detector missed:
        "It is now being sent back to the warehouse. No further details or options are provided in the context.",
        "The information about return fees for the Percussive Massage Gun isn’t provided in the context.",
    ],
)
def test_detect_abstention_positive(answer):
    assert M.detect_abstention(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "No. Overnight and expedited shipping are not available for Hawaii.",
        "Refunds over $200 require manager approval.",
        "Free international shipping is not offered on any order.",
        "Refund gets issued as store credit. No information is needed from you.",
    ],
)
def test_detect_abstention_negative(answer):
    assert not M.detect_abstention(answer)


def test_markdown_emphasis_does_not_break_key_facts():
    answer = "No. Returning it does **not** automatically stop the subscription. They are `separate` actions."
    assert M.contains_phrase(answer, "does not automatically stop")
    assert M.contains_phrase(answer, "separate")


def test_numeric_grounding():
    score, unsupported = M.numeric_grounding(
        "Free over $35; otherwise $4.99, delivered in 24 days.",
        "Standard shipping is free on orders over $35. Orders under $35 incur $4.99.",
    )
    assert unsupported == ["24"] and score == pytest.approx(2 / 3)
    assert M.numeric_grounding("no numbers here", "ctx") == (None, [])


def test_numeric_grounding_accepts_question_numbers_and_ignores_list_markers():
    score, unsupported = M.numeric_grounding(
        "1. A $30 order pays $4.99.\n2. After 11 days it is lost.",
        "Orders under $35 incur a flat shipping fee of $4.99.",
        question="I have a $30 order and 11 days without movement.",
    )
    assert unsupported == [] and score == 1.0


# ---------------------------------------------------------------------------
# Judge parsing (no network)
# ---------------------------------------------------------------------------


def test_parse_correctness():
    out = parse_correctness('```json\n{"score": 1, "rationale": "missing fee"}\n```')
    assert out == {"score": 1, "normalized": 0.5, "rationale": "missing fee"}
    for bad in ('{"score": 3}', '{"score": true}', "no json"):
        with pytest.raises(JudgeParseError):
            parse_correctness(bad)


def test_parse_faithfulness():
    out = parse_faithfulness('{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]}')
    assert out["score"] == 0.5 and out["unsupported_claims"] == ["b"]
    assert parse_faithfulness('{"claims": []}')["score"] is None
    with pytest.raises(JudgeParseError):
        parse_faithfulness('{"claims": [{"claim": "a", "supported": "yes"}]}')


def test_prompt_fingerprint_is_stable():
    assert prompt_fingerprint() == prompt_fingerprint()


# ---------------------------------------------------------------------------
# Runner (fake pipeline)
# ---------------------------------------------------------------------------


def _item(id_, behavior="answer", evidence=None, key_facts=(("$35",),)):
    return GoldenItem(
        id=id_, question=f"question {id_}", answer="ref", category="direct_factual" if behavior == "answer" else "unanswerable",
        secondary_categories=(), expected_behavior=behavior,
        evidence=tuple(evidence if evidence is not None else [{"source_document_id": "d", "quote": "over $35", "grade": 2, "role": "authoritative"}]),
        key_facts=tuple(key_facts), origin={},
    )


CORPUS = [("d", "free on orders over $35 before tax"), ("d", "unrelated chunk"), ("faq", "over $40")]


def test_runner_records_failures_and_rate_limit_retries():
    calls = Counter()

    def rag(question):
        calls[question] += 1
        if question == "question q2":
            raise RuntimeError("database exploded")
        if question == "question q3" and calls[question] == 1:
            raise RuntimeError("Error code: 429 rate limit")
        if question == "question q4":
            return {"top_matches": [], "answer": "I couldn't find enough relevant context to answer that question."}
        return {
            "top_matches": [
                {"chunk_id": 9, "source_document_id": "d", "chunk_index": 1, "content": "unrelated chunk"},
                {"chunk_id": 8, "source_document_id": "d", "chunk_index": 0, "content": "free on orders over $35 before tax"},
            ],
            "answer": "Shipping is free over $35.",
        }

    items = [_item("q1"), _item("q2"), _item("q3"), _item("q4", "abstain", evidence=[], key_facts=())]
    results = evaluate_items(items, rag, CORPUS, k_values=(1, 2), sleep=lambda s: None)

    by_id = {r["id"]: r for r in results}
    assert len(results) == 4  # nothing skipped
    assert by_id["q2"]["status"] == "error" and by_id["q2"]["error"]["message"] == "database exploded"
    assert calls["question q3"] == 2 and by_id["q3"]["status"] == "ok"

    r1 = by_id["q1"]["retrieval"]
    assert r1["recall@1"] == 0.0 and r1["recall@2"] == 1.0 and r1["mrr"] == 0.5
    assert [c["grade"] for c in by_id["q1"]["retrieved"]] == [0, 2]
    assert by_id["q1"]["generation"]["key_fact_recall"] == 1.0

    assert by_id["q4"]["retrieval"]["applicable"] is False
    assert by_id["q4"]["generation"]["abstention_correct"] is True

    summary = summarize(results, (1, 2))
    assert summary["items_failed"] == 1 and summary["failed_ids"] == ["q2"]
    assert summary["retrieval"]["items_evaluated"] == 2
    assert summary["retrieval"]["mrr"] == {"mean": 0.5, "n": 2}
    assert summary["generation"]["abstention"]["correct_abstentions"] == 1
    assert summary["judge"]["items_judged"] == 0


def test_runner_flags_unindexed_evidence_instead_of_scoring_zero():
    item = _item("q1", evidence=[{"source_document_id": "missing_doc", "quote": "x", "grade": 2, "role": "authoritative"}])
    results = evaluate_items([item], lambda q: {"top_matches": [], "answer": "a"}, CORPUS, sleep=lambda s: None)
    assert results[0]["retrieval"]["applicable"] is False
    assert "not found in indexed corpus" in results[0]["retrieval"]["reason"]


def test_runner_records_judge_errors():
    class BrokenJudge:
        def correctness(self, *a):
            raise JudgeParseError("garbage")

    results = evaluate_items(
        [_item("q1")], lambda q: {"top_matches": [], "answer": "over $35"}, CORPUS,
        judge=BrokenJudge(), sleep=lambda s: None,
    )
    assert results[0]["status"] == "ok"
    assert results[0]["judge"]["error"]["type"] == "JudgeParseError"
    assert summarize(results, (1,))["judge"]["errors"] == ["q1"]
