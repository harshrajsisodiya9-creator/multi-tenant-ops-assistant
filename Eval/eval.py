import json
import time
from pathlib import Path
from uuid import UUID

from app.database import get_db
from app.hybrid_search import answer_question

Path_Directory = Path(__file__).resolve().parent

RETRY_BASE_SECONDS = 10


def generate_results(limit=None):
    with open(Path_Directory / "eval_dataset.json") as f:
        dataset = json.load(f)

    if limit is not None:
        dataset = dataset[:limit]

    db = next(get_db())
    tenant_id = UUID("f67e4814-c4bf-421b-94d8-0d5894703816")
    results = []
    try:
        for pair, item in enumerate(dataset, start=1):
            for attempt in range(3):
                try:
                    response = answer_question(
                        db=db, tenant_id=tenant_id, question=item["question"]
                    )
                    break
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        wait = RETRY_BASE_SECONDS * (attempt + 1)
                        print(f"Groq rate limit hit, waiting {wait}s...", flush=True)
                        time.sleep(wait)
                    else:
                        raise
            else:
                raise RuntimeError(f"Failed after 3 retries on question {pair}")

            context = [match["content"] for match in response["top_matches"]]
            answer = response.get("answer") or ""
            if not answer.strip():
                answer = "No relevant information found."
            print(f"pair processed: {pair}/{len(dataset)}", flush=True)
            results.append(
                {
                    "user_input": item["question"],
                    "reference": item["answer"],
                    "retrieved_contexts": context,
                    "response": answer,
                }
            )
    finally:
        db.close()

    return results


if __name__ == "__main__":
    generate_results()
