import argparse
import asyncio
import random
import re
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

MAX_WORKERS = 1
REQUESTS_PER_MINUTE = 10
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "phase1_baseline_results.csv"
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


# Ragas 0.4's collections metrics require its Instructor-based LLM interface.
# Build its async Google GenAI client explicitly: llm_factory currently wraps
# Google clients synchronously, while the collections metrics call agenerate().
genai_client = genai.Client(
    api_key=setting.gemini_api_key,
    http_options=types.HttpOptions(
        timeout=60_000,
        # Keep retries centralized below so every Gemini attempt goes through
        # the shared limiter and only rate-limit errors are retried.
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)
judge_llm = InstructorLLM(
    client=instructor.from_genai(genai_client, use_async=True),
    model=setting.gemini_model,
    provider="google",
    temperature=0,
)
judge_embeddings = GoogleEmbeddings(
    client=genai_client,
    model="gemini-embedding-001",
)

faithfulness = Faithfulness(llm=judge_llm)
answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
context_precision = ContextPrecision(llm=judge_llm)
context_recall = ContextRecall(llm=judge_llm)


class AsyncRateLimiter:
    """Space all metric attempts evenly under Gemini's free-tier RPM quota."""

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


# All four metrics, retries, and AnswerRelevancy embeddings share this limiter.
RATE_LIMITER = AsyncRateLimiter(REQUESTS_PER_MINUTE)


def _exception_chain(error):
    seen = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attribute in ("__cause__", "__context__"):
            nested = getattr(current, attribute, None)
            if nested is not None:
                pending.append(nested)


def is_rate_limit_error(error):
    for current in _exception_chain(error):
        for attribute in ("status_code", "status", "code"):
            value = getattr(current, attribute, None)
            if value == 429 or str(value).upper() in {"429", "RESOURCE_EXHAUSTED"}:
                return True
        message = str(current).lower()
        if "resource_exhausted" in message or "too many requests" in message:
            return True
        if re.search(r"(?m)^\s*(?:HTTP\s+)?429\b", message):
            return True
    return False


def _retry_delay_from_value(value):
    """Find Google's RetryInfo/Retry-After delay in SDK exceptions."""
    if value is None:
        return None
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).replace("_", "").replace("-", "").lower()
            if normalized in {
                "retrydelay",
                "retryafter",
                "retryafterseconds",
            } or isinstance(nested, (dict, list, tuple)):
                delay = _retry_delay_from_value(nested)
                if delay is not None:
                    return delay
        return None
    if isinstance(value, (list, tuple)):
        for nested in value:
            delay = _retry_delay_from_value(nested)
            if delay is not None:
                return delay
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if hasattr(value, "seconds"):
        return max(
            0.0,
            float(value.seconds) + float(getattr(value, "nanos", 0)) / 1_000_000_000,
        )
    if isinstance(value, str):
        match = re.fullmatch(
            r"\s*([0-9]+(?:\.[0-9]+)?)\s*(s|sec|secs|seconds)?\s*", value, re.I
        )
        if match:
            return float(match.group(1))
    return None


def google_retry_delay(error):
    for current in _exception_chain(error):
        for attribute in (
            "retry_delay",
            "retry_after",
            "details",
            "response",
            "headers",
        ):
            value = getattr(current, attribute, None)
            delay = _retry_delay_from_value(value)
            if delay is not None:
                return delay
            if attribute == "response" and value is not None:
                headers = getattr(value, "headers", None)
                delay = _retry_delay_from_value(headers)
                if delay is not None:
                    return delay
                try:
                    delay = _retry_delay_from_value(value.json())
                    if delay is not None:
                        return delay
                except (AttributeError, TypeError, ValueError):
                    pass

        message = str(current)
        patterns = (
            r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
            r"retryDelay['\"]?\s*[:=]\s*['\"]?([0-9]+(?:\.[0-9]+)?)\s*s",
        )
        for pattern in patterns:
            match = re.search(pattern, message, re.I)
            if match:
                return float(match.group(1))
    return None


async def rate_limited_call(operation, *args, **kwargs):
    """Rate limit and retry one underlying Gemini API request."""
    for retry_number in range(MAX_RETRIES + 1):
        await RATE_LIMITER.wait()
        try:
            return await operation(*args, **kwargs)
        except Exception as error:
            if not is_rate_limit_error(error):
                raise
            if retry_number >= MAX_RETRIES:
                print(
                    f"Gemini rate-limit retries exhausted after {MAX_RETRIES} retries",
                    flush=True,
                )
                raise

            retry_delay = google_retry_delay(error)
            if retry_delay is None:
                max_delay = min(
                    MAX_BACKOFF_SECONDS,
                    BASE_BACKOFF_SECONDS * (2**retry_number),
                )
                retry_delay = random.uniform(0.0, max_delay)
            print(
                f"Gemini rate limit; retry {retry_number + 1}/{MAX_RETRIES} "
                f"in {retry_delay:.1f}s",
                flush=True,
            )
            await asyncio.sleep(retry_delay)


def install_rate_limited_method(target, method_name):
    original = getattr(target, method_name)

    async def rate_limited(*args, **kwargs):
        # Instructor otherwise retries generation and validation errors itself.
        # Keep each call to one attempt; the wrapper retries rate limits only.
        if target is judge_llm.client:
            kwargs["max_retries"] = 1
        return await rate_limited_call(original, *args, **kwargs)

    setattr(target, method_name, rate_limited)


# Limit each actual Gemini call, including the multiple calls some metrics
# make internally and AnswerRelevancy's embedding requests.
install_rate_limited_method(judge_llm.client, "create")
install_rate_limited_method(judge_embeddings, "aembed_text")
install_rate_limited_method(judge_embeddings, "aembed_texts")


async def gather_fail_fast(*coroutines):
    """Cancel sibling work immediately if any evaluation task fails."""
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def evaluate_sample(item, semaphore, sample_number=None, total_samples=None):
    async def score(metric, **kwargs):
        # Each metric uses the shared rate-limited Gemini and embedding clients.
        async with semaphore:
            prefix = (
                f"[{sample_number}/{total_samples}] "
                if sample_number is not None and total_samples is not None
                else ""
            )
            print(f"{prefix}Starting {metric.name}", flush=True)
            try:
                result = await metric.ascore(**kwargs)
            except Exception as error:
                print(
                    f"{prefix}Failed {metric.name}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                raise
            print(f"{prefix}Finished {metric.name}: {result.value}", flush=True)
            return result

    common = {
        "user_input": item["user_input"],
        "retrieved_contexts": item["retrieved_contexts"],
    }
    (
        faithfulness_result,
        relevancy_result,
        precision_result,
        recall_result,
    ) = await gather_fail_fast(
        score(faithfulness, response=item["response"], **common),
        score(
            answer_relevancy,
            response=item["response"],
            user_input=item["user_input"],
        ),
        score(
            context_precision,
            reference=item["reference"],
            **common,
        ),
        score(
            context_recall,
            reference=item["reference"],
            **common,
        ),
    )

    return {
        "user_input": item["user_input"],
        "response": item["response"],
        "retrieved_contexts": item["retrieved_contexts"],
        "reference": item["reference"],
        "faithfulness": faithfulness_result.value,
        "answer_relevancy": relevancy_result.value,
        "context_precision": precision_result.value,
        "context_recall": recall_result.value,
    }


async def main(limit=None):
    print(
        f"Generating RAG results for {limit or 'all'} samples",
        flush=True,
    )
    selected_results = generate_results(limit=limit)
    semaphore = asyncio.Semaphore(MAX_WORKERS)
    print(
        f"Starting evaluation of {len(selected_results)} samples "
        f"(max_workers={MAX_WORKERS}, rate_limit={REQUESTS_PER_MINUTE} RPM, "
        "Gemini timeout=60s)",
        flush=True,
    )
    evaluated_results = await gather_fail_fast(
        *(
            evaluate_sample(item, semaphore, index, len(selected_results))
            for index, item in enumerate(selected_results, start=1)
        )
    )

    df = pd.DataFrame(evaluated_results, columns=OUTPUT_COLUMNS)
    df.to_csv(OUTPUT_PATH, index=False)
    print(
        df[
            [
                "user_input",
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]
        ]
    )
    print(f"Saved {len(df)} evaluated rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG baseline with Ragas."
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N generated results (useful for a smoke run).",
    )
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit))
