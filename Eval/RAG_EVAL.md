# Golden RAG evaluation

Evaluates the FAQ RAG pipeline (`app.hybrid_search.answer_question`) on a 20-question
golden dataset. Retrieval and generation are scored separately.

## Run

```bash
docker compose up -d                      # Postgres with the indexed KB
python -m Eval.run_rag_eval               # deterministic metrics only
python -m Eval.run_rag_eval --judge       # + Groq LLM judge
python -m Eval.run_rag_eval --limit 3     # smoke run
pytest tests/test_rag_eval.py             # offline tests (no DB / network)
```

Options: `--tenant-id` (default: `f67e4814-…`, the tenant `Eval/eval.py` uses),
`--k 1 3 5`, `--output PATH`, `--allow-unresolved-evidence`.
The default report path is `Eval/reports/rag_eval_<UTC>.json`.

Exit codes: `0` = every item was evaluated. `1` = at least one pipeline or judge call
failed; failures are recorded per item and nothing is skipped. `2` = the dataset or the
pre-flight check is invalid.

No new dependencies. The runner uses only packages already in
`requirements-runtime.txt`. The existing RAGAS scripts (`ragas.py`, `ragasGemini.py`)
are unchanged.

## Files

| File | Purpose |
|---|---|
| `golden_dataset.json` | 20 `{question, answer}` pairs, the same format as `eval_dataset.json` |
| `golden_annotations.json` | One entry per question, aligned by position and exact question text: category, provenance, evidence quotes, key facts |
| `metrics.py` | Pure metric functions |
| `golden.py` | Loading and validation, and evidence-quote resolution against a corpus |
| `judge.py` | Optional LLM judge: model, prompts, limitations |
| `run_rag_eval.py` | Runner |
| `eval_dataset.json` | **Original 21-pair dataset, kept unchanged** |

## Relevance annotations

Chunk IDs are autoincrement values, so they change on every re-ingest. Relevance is
therefore labelled with **evidence quotes** copied word-for-word from `files/*.md`:

```json
{"source_document_id": "refund_policy", "quote": "Refunds over $200 require manager approval ...",
 "grade": 2, "role": "authoritative"}
```

A retrieved chunk has grade *g* if it comes from that document and contains the quote
(whitespace is normalised).

- **Grade 2:** required authoritative evidence.
- **Grade 1:** related but `non_authoritative` (`support_internal_faq`) or `superseded`
  (`refund_policy_v1`).

There are two guards against stale labels:
- A test checks that every quote fits inside a single chunk produced by the production
  splitter.
- The runner checks before starting that every quote exists in the tenant's indexed
  chunks.

## Metrics

**Retrieval**, over the pipeline's top-5. Only questions with grade-2 evidence count;
the others are listed as "not applicable".

- **Recall@K:** share of the grade-2 quotes that appear in the top K. It is counted per
  quote rather than per chunk, because the 100-character overlap can put one quote in
  two chunks.
- **Complete-evidence@K:** 1 if every grade-2 quote is in the top K. This matters for
  multi-document questions.
- **MRR:** 1 / rank of the first grade-2 chunk.
- **NDCG@K:** gain 2^g − 1 with a log2 discount. The ideal DCG is computed over the
  whole indexed corpus.

**Generation**, deterministic:
- **Key-fact recall:** lexical, and blind to negation.
- **Abstention:** correct abstentions on the 3 unanswerable questions, plus false
  abstentions on answerable ones (abstention detected and key-fact recall below 0.5).
- **Numeric grounding:** share of numbers in the answer that appear in the retrieved
  context or the question.

**Generation**, LLM judge (`--judge`):
- **Correctness:** 0/1/2 against the reference, reported normalised to 0–1.
- **Faithfulness:** share of claims supported by the retrieved context.

The judge uses Groq `GROQ_JUDGE_MODEL` (default `openai/gpt-oss-120b`) at temperature 0.
Prompts and limitations are in `judge.py`. Each report records the judge model and the
SHA-256 of each prompt.

Limitations:
- The judge is not deterministic.
- The judge comes from the same model family as the generator (`gpt-oss-20b`).
- Faithfulness only checks against the retrieved context. An answer that faithfully
  repeats a superseded chunk still scores as faithful, so read it together with
  correctness.

## Reproducibility

Each report records:
- git commit and whether the tree was dirty
- SHA-256 of the dataset, the annotations and each KB document
- a hash of the indexed corpus, and whether each document's chunks match the current
  `files/` re-chunked by the production splitter
- every retrieval and generation parameter, read from the production modules
- the K values and the judge configuration

## Dataset provenance

No committed version of the original dataset had 20 pairs: `7f6b946` had 26 and
`d7fea7f` (current) has 21. The golden set is built as follows:
- **14 pairs** are copied word-for-word from those versions.
- **5 pairs** keep the original question but have a revised answer.
- **1 pair** is new.

`origin` in the annotations records the commit and index of each pair. For revised pairs
it also records the reason and the original answer.

| id | change |
|---|---|
| q08 | Dropped an unsupported sentence ("applies to all active orders") |
| q12 | Remote-region delay is now a condition, not stated as fact |
| q14 | The "2 failures were consecutive" assumption is made explicit; "remains paused" is only true if the payment isn't updated within 5 days |
| q18 | RETURNED_TO_SENDER: the original refund answer was unsupported. Now an abstention that still explains the status |
| q19 | Prepaid labels: abstain, and don't rely on the superseded v1 policy |
| q20 | New: payment methods (no billing/payments document exists) |

Original pairs left out because their answers aren't fully supported:
- d7fea7f #5: the "will still be delivered" inference
- d7fea7f #16: the 7-day window plus a conflict inside the refund policy
- d7fea7f #17: frequency change before billing
- d7fea7f #20: "the entire order cannot ship"
- 7f6b946 #11: "contact support"

The rest were left out to keep the category balance.

Category distribution: direct_factual 3, paraphrased 3, numerical_threshold 4,
conditional 4, multi_document 3, unanswerable 3. Secondary tags mark the 5 other
questions that also need more than one document (q04, q08, q09, q10, q13).

After the first baseline run, the deterministic answer checks were fixed (markdown
stripping, contractions in the abstention detector, question numbers counted as
grounded). The q02 and q16 key-fact synonym lists were also widened. Retrieval labels
were not changed. The annotations hash in each report shows which version produced it.
