import argparse
import asyncio
import random
from pathlib import Path

import instructor
import pandas as pd
from google import genai
from google.genai import types
from ragas.embeddings import GoogleEmbeddings
from ragas.llms import InstructorLLM
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.config import setting
from Eval.eval import generate_results

REQUESTS_PER_MINUTE = 15
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2.0
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "phase1_baseline_results_gemini.csv"
SCORES_PATH = Path(__file__).resolve().parents[1] / "phase1_scores_gemini.csv"
OUTPUT_COLUMNS = [
    "user_input",
    "response",
    "retrieved_contexts",
    "reference",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

genai_client = genai.Client(
    api_key=setting.gemini_api_key,
    http_options=types.HttpOptions(
        timeout=60_000,
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)

judge_llm = InstructorLLM(
    client=instructor.from_genai(genai_client, use_async=True),
    model=setting.gemini_model,
    provider="google",
    temperature=0,
)

judge_embeddings = GoogleEmbeddings(client=genai_client, model="gemini-embedding-001")

faithfulness = Faithfulness(llm=judge_llm)
answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
context_precision = ContextPrecision(llm=judge_llm)
context_recall = ContextRecall(llm=judge_llm)


class AsyncRateLimiter:
    def __init__(self, requests_per_minute):
        self.interval = 60.0 / requests_per_minute
        self._lock = asyncio.Lock()
        self._last_request_at = None

    async def wait(self):
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self._last_request_at is not None:
                delay = self.interval - (now - self._last_request_at)
                if delay > 0:
                    await asyncio.sleep(delay)
            self._last_request_at = loop.time()


RATE_LIMITER = AsyncRateLimiter(REQUESTS_PER_MINUTE)


def is_rate_limit_error(error):
    message = str(error).lower()
    status = getattr(error, "status_code", None)
    if status == 429:
        return True
    if any(s in message for s in ("rate limit", "too many requests", "resource_exhausted")):
        return True
    cause = getattr(error, "__cause__", None)
    if cause is not None:
        return is_rate_limit_error(cause)
    return False


async def rate_limited_call(operation, *args, **kwargs):
    for retry_number in range(MAX_RETRIES + 1):
        await RATE_LIMITER.wait()
        try:
            return await operation(*args, **kwargs)
        except Exception as error:
            if not is_rate_limit_error(error):
                raise
            if retry_number >= MAX_RETRIES:
                print(f"Rate-limit retries exhausted after {MAX_RETRIES} retries", flush=True)
                raise
            delay = random.uniform(0, BASE_BACKOFF_SECONDS * (2 ** retry_number))
            print(
                f"Rate limited; retry {retry_number + 1}/{MAX_RETRIES} in {delay:.1f}s",
                flush=True,
            )
            await asyncio.sleep(delay)


# Monkey-patch LLM and embedding methods to go through the shared rate limiter.
original_create = judge_llm.client.create


async def _rate_limited_create(*args, **kwargs):
    return await rate_limited_call(original_create, *args, **kwargs)


judge_llm.client.create = _rate_limited_create

original_aembed_text = judge_embeddings.aembed_text
original_aembed_texts = judge_embeddings.aembed_texts


async def _rate_limited_aembed_text(*args, **kwargs):
    return await rate_limited_call(original_aembed_text, *args, **kwargs)


async def _rate_limited_aembed_texts(*args, **kwargs):
    return await rate_limited_call(original_aembed_texts, *args, **kwargs)


judge_embeddings.aembed_text = _rate_limited_aembed_text
judge_embeddings.aembed_texts = _rate_limited_aembed_texts


async def evaluate_sample(item, index, total):
    async def score(metric, **kwargs):
        print(f"[{index}/{total}] Starting {metric.name}", flush=True)
        try:
            result = await metric.ascore(**kwargs)
        except Exception as error:
            print(f"[{index}/{total}] Failed {metric.name}: {error}", flush=True)
            raise
        print(f"[{index}/{total}] Finished {metric.name}: {result.value}", flush=True)
        return result

    common = {
        "user_input": item["user_input"],
        "retrieved_contexts": item["retrieved_contexts"],
    }

    faith_result = await score(faithfulness, response=item["response"], **common)
    relevancy_result = await score(
        answer_relevancy, response=item["response"], user_input=item["user_input"],
    )
    prec_result = await score(context_precision, reference=item["reference"], **common)
    recall_result = await score(context_recall, reference=item["reference"], **common)

    return {
        "user_input": item["user_input"],
        "response": item["response"],
        "retrieved_contexts": item["retrieved_contexts"],
        "reference": item["reference"],
        "faithfulness": faith_result.value,
        "answer_relevancy": relevancy_result.value,
        "context_precision": prec_result.value,
        "context_recall": recall_result.value,
    }


async def main(limit=None):
    print(f"Generating RAG results for {limit or 'all'} samples", flush=True)
    selected_results = generate_results(limit=limit)
    print(
        f"Evaluating {len(selected_results)} samples "
        f"(rate_limit={REQUESTS_PER_MINUTE} RPM)",
        flush=True,
    )

    evaluated = []
    for index, item in enumerate(selected_results, start=1):
        result = await evaluate_sample(item, index, len(selected_results))
        evaluated.append(result)

    df = pd.DataFrame(evaluated, columns=OUTPUT_COLUMNS)
    df.to_csv(OUTPUT_PATH, index=False)

    score_cols = ["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scores_df = df[score_cols].copy()
    scores_df.insert(0, "#", range(1, len(scores_df) + 1))
    scores_df.to_csv(SCORES_PATH, index=False)

    print(scores_df.to_string(index=False, max_colwidth=60))
    print("\nAverages:")
    for col in score_cols[1:]:
        print(f"  {col:20s}: {df[col].mean():.4f}")
    print(f"\nFull results:  {OUTPUT_PATH}")
    print(f"Scores only:   {SCORES_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG baseline with Ragas and Gemini."
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N results (useful for a smoke run).",
    )
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit))
