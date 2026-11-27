"""Pure, deterministic metric functions for the RAG evaluation.

Nothing in this module touches the database, the network, or an LLM, so every
metric here is reproducible and unit-tested (tests/test_rag_eval.py).

Relevance model
---------------
Relevance labels live in Eval/golden_annotations.json as *evidence quotes*, not
chunk IDs (chunk IDs are autoincrement values that change on every re-ingest).
A retrieved chunk is relevant with grade ``g`` when it comes from the quote's
``source_document_id`` and contains the quote after whitespace normalisation.
A chunk's grade is the max grade of all quotes it contains.

- grade 2: required, authoritative evidence for the reference answer
- grade 1: related but non-authoritative (internal FAQ) or superseded context
- grade 0: not relevant

Metric definitions
------------------
- Recall@K (evidence recall): fraction of the item's grade-2 evidence quotes
  contained in at least one of the top-K chunks. Evidence-level rather than
  chunk-level, because the 100-char chunk overlap can put one quote in two
  chunks and a system should not need both copies.
- Complete-evidence@K: 1.0 if *every* grade-2 quote is in the top K, else 0.0.
- MRR: mean over items of 1 / rank of the first chunk with grade 2 (0 if none).
- NDCG@K: graded gain (2^g - 1), log2 discount. The ideal DCG is computed from
  the grades of *all* chunks in the indexed corpus, so it reflects what a
  perfect ranker could retrieve from this index.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

REQUIRED_GRADE = 2

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    return _WS.sub(" ", text or "").strip()


_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|`)")
_MD_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+", re.MULTILINE)


def normalize_answer(text: str) -> str:
    """Normalise a *generated answer* for lexical checks.

    Strips markdown emphasis and list markers ("does **not** stop" -> "does not
    stop"; "1. You have" -> "You have"), normalises curly apostrophes, then
    collapses whitespace. Evidence matching on chunks uses normalize_text only,
    because chunk content is compared verbatim against source quotes.
    """
    text = (text or "").replace("’", "'").replace("‑", "-")
    text = _MD_LIST_MARKER.sub("", text)
    return normalize_text(_MD_EMPHASIS.sub("", text))


# ---------------------------------------------------------------------------
# Relevance resolution
# ---------------------------------------------------------------------------


def matched_evidence(
    source_document_id: str, content: str, evidence: Sequence[dict]
) -> list[int]:
    """Indices of evidence items contained in this chunk."""
    norm = normalize_text(content)
    return [
        i
        for i, e in enumerate(evidence)
        if e["source_document_id"] == source_document_id
        and normalize_text(e["quote"]) in norm
    ]


def chunk_grade(
    source_document_id: str, content: str, evidence: Sequence[dict]
) -> int:
    hits = matched_evidence(source_document_id, content, evidence)
    return max((evidence[i]["grade"] for i in hits), default=0)


def required_evidence_indices(evidence: Sequence[dict]) -> list[int]:
    return [i for i, e in enumerate(evidence) if e["grade"] >= REQUIRED_GRADE]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def recall_at_k(
    matched_per_rank: Sequence[Iterable[int]], required: Iterable[int], k: int
) -> float | None:
    """Evidence recall over the top-k ranks. None if nothing is required."""
    required = set(required)
    if not required:
        return None
    covered: set[int] = set()
    for hits in matched_per_rank[:k]:
        covered.update(hits)
    return len(covered & required) / len(required)


def complete_evidence_at_k(
    matched_per_rank: Sequence[Iterable[int]], required: Iterable[int], k: int
) -> float | None:
    r = recall_at_k(matched_per_rank, required, k)
    return None if r is None else float(r == 1.0)


def reciprocal_rank(grades: Sequence[int], threshold: int = REQUIRED_GRADE) -> float:
    for rank, g in enumerate(grades, start=1):
        if g >= threshold:
            return 1.0 / rank
    return 0.0


def dcg(grades: Sequence[int], k: int) -> float:
    return sum(
        (2**g - 1) / math.log2(rank + 1)
        for rank, g in enumerate(grades[:k], start=1)
    )


def ndcg_at_k(
    retrieved_grades: Sequence[int], corpus_grades: Iterable[int], k: int
) -> float | None:
    """NDCG@k. None when the corpus holds no relevant chunk (undefined)."""
    ideal = dcg(sorted(corpus_grades, reverse=True), k)
    if ideal == 0:
        return None
    return dcg(retrieved_grades, k) / ideal


# ---------------------------------------------------------------------------
# Generation metrics (deterministic)
# ---------------------------------------------------------------------------


def _phrase_pattern(phrase: str) -> re.Pattern:
    """Case-insensitive match that won't fire inside a longer word or number.

    "$35" does not match "$350" or "$35.99"; "21" does not match "121".
    A phrase ending in a letter may take a plural "s"/"es" ("agent" matches
    "agents"), so lexical facts aren't defeated by inflection.
    """
    body = re.escape(normalize_text(phrase)).replace(r"\ ", r"\s+")
    start = r"(?<![\w$])" if re.match(r"[\w$]", phrase) else ""
    if re.search(r"\d$", phrase):
        end = r"(?![\d.,]*\d)"
    elif re.search(r"\w$", phrase):
        end = r"(?:s|es)?(?!\w)"
    else:
        end = ""
    return re.compile(start + body + end, re.IGNORECASE)


def contains_phrase(text: str, phrase: str) -> bool:
    return bool(_phrase_pattern(phrase).search(normalize_answer(text)))


def key_fact_recall(answer: str, key_facts: Sequence[Sequence[str]]) -> tuple[float | None, list[int]]:
    """Fraction of key-fact groups covered by the answer, and missing group indices.

    A group is covered if the answer contains any of its alternatives. Lexical
    only: it cannot detect contradictions or negation ("is NOT $35").
    """
    if not key_facts:
        return None, []
    missing = [
        i
        for i, group in enumerate(key_facts)
        if not any(contains_phrase(answer, alt) for alt in group)
    ]
    return 1 - len(missing) / len(key_facts), missing


# Phrases that signal the answer declines to answer from the documents. Kept
# specific on purpose: "overnight is not available for Hawaii" is a factual
# answer, not an abstention, so a bare "not available" does not count.
_NOT = r"(?:is|are|was|were) not|isn't|aren't|wasn't|weren't"
_SOURCE = r"(?:the )?(?:provided |available |given |current )?(?:context|polic(?:y|ies)|documents?|knowledge ?base|information|sources?)"
ABSTENTION_PATTERNS = [
    rf"information (?:{_NOT}) (?:available|provided|included)",
    rf"(?:{_NOT}) (?:available|provided|mentioned|specified|covered|included|stated|addressed|listed|described|found) in {_SOURCE}",
    # "No further details or options are provided in the context."
    rf"\bno (?:further |additional |specific |other )?(?:information|details|mention|guidance|options)\b[^.]{{0,60}}?(?:provided|mentioned|specified|available|given|found) in {_SOURCE}",
    rf"{_SOURCE} (?:provided |available )?(?:does|do|did)(?: not|n't) (?:say|specify|mention|cover|contain|include|state|address|list|describe|provide|indicate|explain|detail)",
    r"(?:doesn't|does not|don't|do not) (?:have|contain|include) (?:any |enough )?(?:information|details|data)",
    r"\bno (?:information|details|mention|guidance)\b(?! (?:is|are) (?:needed|required))",
    r"(?:couldn't|could not|can't|cannot|unable to) (?:find|locate|determine|confirm|tell|answer|say)",
    r"cannot be (?:answered|determined|confirmed)",
    r"not enough (?:relevant )?(?:context|information)",
]
_ABSTAIN_RE = re.compile("|".join(f"(?:{p})" for p in ABSTENTION_PATTERNS), re.IGNORECASE)


def detect_abstention(answer: str) -> bool:
    return bool(_ABSTAIN_RE.search(normalize_answer(answer)))


_NUMBER_RE = re.compile(r"\$?\d+(?:[.,]\d+)*%?")


def _numbers(text: str) -> set[str]:
    return {t.rstrip(".,").lstrip("$").rstrip("%") for t in _NUMBER_RE.findall(text or "")} - {""}


def numeric_grounding(
    answer: str, context: str, question: str = ""
) -> tuple[float | None, list[str]]:
    """Fraction of numbers/amounts in the answer that appear in the context or question.

    A cheap faithfulness proxy: unsupported numbers are the most common and
    most costly policy hallucination. Numbers the user supplied in the question
    ("11 business days", "$30") are grounded. Numbers derived by arithmetic
    (e.g. a 14 + 10 = 24 day total) are reported as unsupported even when correct.
    """
    tokens = {t.rstrip(".,") for t in _NUMBER_RE.findall(normalize_answer(answer))} - {""}
    if not tokens:
        return None, []
    known = _numbers(context) | _numbers(question)
    unsupported = sorted(t for t in tokens if t.lstrip("$").rstrip("%") not in known)
    return 1 - len(unsupported) / len(tokens), unsupported


def mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
