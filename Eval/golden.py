"""Load and validate the golden dataset and its annotations.

The golden dataset (Eval/golden_dataset.json) holds only ``question`` and
``answer`` so it stays format-compatible with Eval/eval_dataset.json. Everything
else -- category, provenance, evidence quotes, key facts -- lives in
Eval/golden_annotations.json, aligned by position and exact question text.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from Eval.metrics import normalize_text

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parent
DEFAULT_DATASET = EVAL_DIR / "golden_dataset.json"
DEFAULT_ANNOTATIONS = EVAL_DIR / "golden_annotations.json"
KB_DIR = PROJECT_DIR / "files"

CATEGORIES = (
    "direct_factual",
    "paraphrased",
    "numerical_threshold",
    "conditional",
    "multi_document",
    "unanswerable",
)
EVIDENCE_ROLES = ("authoritative", "non_authoritative", "superseded")


class GoldenDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    answer: str
    category: str
    secondary_categories: tuple[str, ...]
    expected_behavior: str  # "answer" | "abstain"
    evidence: tuple[dict, ...]
    key_facts: tuple[tuple[str, ...], ...]
    origin: dict


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_golden(
    dataset_path: Path = DEFAULT_DATASET,
    annotations_path: Path = DEFAULT_ANNOTATIONS,
    expected_size: int | None = 20,
) -> list[GoldenItem]:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    meta = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    annotations = meta["items"]
    validate(dataset, annotations, expected_size=expected_size)
    return [
        GoldenItem(
            id=a["id"],
            question=d["question"],
            answer=d["answer"],
            category=a["category"],
            secondary_categories=tuple(a.get("secondary_categories", [])),
            expected_behavior=a["expected_behavior"],
            evidence=tuple(a["evidence"]),
            key_facts=tuple(tuple(g) for g in a["key_facts"]),
            origin=a["origin"],
        )
        for d, a in zip(dataset, annotations)
    ]


def validate(dataset: list, annotations: list, expected_size: int | None = 20) -> None:
    """Raise GoldenDatasetError listing every problem found."""
    errors: list[str] = []
    if not isinstance(dataset, list):
        raise GoldenDatasetError("dataset must be a JSON list")
    if expected_size is not None and len(dataset) != expected_size:
        errors.append(f"dataset has {len(dataset)} items, expected {expected_size}")
    if len(dataset) != len(annotations):
        errors.append(
            f"dataset has {len(dataset)} items but annotations has {len(annotations)}"
        )

    seen_q, seen_id = set(), set()
    for pos, d in enumerate(dataset, start=1):
        if not isinstance(d, dict) or set(d) != {"question", "answer"}:
            errors.append(f"dataset[{pos}] must have exactly the keys question, answer")
            continue
        if not all(isinstance(d[k], str) and d[k].strip() for k in ("question", "answer")):
            errors.append(f"dataset[{pos}] question/answer must be non-empty strings")
        if d["question"] in seen_q:
            errors.append(f"dataset[{pos}] duplicate question")
        seen_q.add(d["question"])

    for pos, (d, a) in enumerate(zip(dataset, annotations), start=1):
        tag = f"annotations[{pos}] ({a.get('id')})"
        if a.get("id") in seen_id:
            errors.append(f"{tag} duplicate id")
        seen_id.add(a.get("id"))
        if isinstance(d, dict) and a.get("question") != d.get("question"):
            errors.append(f"{tag} question text does not match dataset[{pos}]")
        if a.get("category") not in CATEGORIES:
            errors.append(f"{tag} unknown category {a.get('category')!r}")
        abstain = a.get("category") == "unanswerable"
        if a.get("expected_behavior") != ("abstain" if abstain else "answer"):
            errors.append(f"{tag} expected_behavior inconsistent with category")
        evidence = a.get("evidence", [])
        for e in evidence:
            if e.get("grade") not in (1, 2) or e.get("role") not in EVIDENCE_ROLES:
                errors.append(f"{tag} evidence needs grade in (1,2) and a valid role")
            if not normalize_text(e.get("quote", "")):
                errors.append(f"{tag} evidence quote is empty")
            if e.get("grade") == 2 and e.get("role") != "authoritative":
                errors.append(f"{tag} grade-2 evidence must be authoritative")
        if not abstain and not any(e.get("grade") == 2 for e in evidence):
            errors.append(f"{tag} answerable item needs at least one grade-2 evidence quote")
        if not abstain and not a.get("key_facts"):
            errors.append(f"{tag} answerable item needs key_facts")

    if errors:
        raise GoldenDatasetError("Invalid golden dataset:\n  - " + "\n  - ".join(errors))


def unresolved_evidence(
    items: list[GoldenItem], corpus: list[tuple[str, str]]
) -> list[str]:
    """Evidence quotes that no chunk in ``corpus`` contains.

    ``corpus`` is a list of (source_document_id, content). An unresolved quote
    means the annotation, the KB documents, or the chunking config has drifted,
    and retrieval metrics for that item would be silently wrong.
    """
    by_doc: dict[str, list[str]] = {}
    for doc, content in corpus:
        by_doc.setdefault(doc, []).append(normalize_text(content))
    problems = []
    for item in items:
        for e in item.evidence:
            q = normalize_text(e["quote"])
            if not any(q in c for c in by_doc.get(e["source_document_id"], [])):
                problems.append(
                    f"{item.id}: {e['source_document_id']}: {e['quote'][:80]!r}"
                )
    return problems
