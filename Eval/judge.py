"""Optional LLM judge for answer correctness and faithfulness.

Only runs when the evaluation is started with ``--judge``. All deterministic
metrics in Eval/metrics.py are computed whether or not the judge runs.

Model
    Groq, ``setting.groq_judge_model`` (default ``openai/gpt-oss-120b``; set
    GROQ_JUDGE_MODEL to override), temperature 0. The generator is
    ``setting.groq_model`` (default ``openai/gpt-oss-20b``). The judge is a
    larger model, but it belongs to the same family as the generator.

Prompts
    CORRECTNESS_PROMPT and FAITHFULNESS_PROMPT below. Each report records
    JUDGE_PROMPT_VERSION and the SHA-256 of both templates, so a changed prompt
    is visible when comparing runs.

Limitations
    - Not deterministic: temperature 0 on a hosted API does not guarantee
      identical output between runs or model revisions.
    - Self-preference bias: the judge belongs to the same model family as the
      generator and may grade its style leniently.
    - Correctness is graded only against the reference answer, so if the
      reference is incomplete, a correct extra detail can be penalised.
    - Faithfulness is checked only against the retrieved chunks. An answer can
      be faithful to a superseded or non-authoritative chunk (for example
      refund_policy_v1) and still be wrong; read it together with correctness.
    - If the judge returns unparseable output, the item gets a judge error.
      It is never scored 0.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

JUDGE_PROMPT_VERSION = "1"

CORRECTNESS_PROMPT = """You are grading a customer-support answer against a reference answer written from the company's policy documents.

Question:
{question}

Reference answer:
{reference}

Candidate answer:
{candidate}

Grade the candidate on this scale:
2 = correct: states every essential fact and conclusion in the reference, and contradicts nothing in it.
1 = partially correct: no contradiction of the reference, but one or more essential facts or conclusions are missing.
0 = incorrect: contradicts the reference, states a wrong number/threshold/condition, or gives an answer where the reference says the information is not available (or abstains where the reference gives an answer).

Extra detail that does not contradict the reference is not penalised. Judge meaning, not wording.

Respond with only a JSON object: {{"score": 0 | 1 | 2, "rationale": "<one sentence>"}}"""

FAITHFULNESS_PROMPT = """You are checking whether an answer is supported by the retrieved context it was generated from.

Question:
{question}

Retrieved context:
{context}

Answer:
{answer}

List each factual claim the answer makes about policies, numbers, timelines, conditions or products. Do not list statements that only say information is missing or unavailable. For each claim, decide whether the retrieved context explicitly supports it (simple arithmetic on numbers in the context counts as supported).

Respond with only a JSON object: {{"claims": [{{"claim": "<claim>", "supported": true | false}}]}}"""


def prompt_fingerprint() -> dict:
    return {
        "prompt_version": JUDGE_PROMPT_VERSION,
        "correctness_prompt_sha256": hashlib.sha256(CORRECTNESS_PROMPT.encode()).hexdigest(),
        "faithfulness_prompt_sha256": hashlib.sha256(FAITHFULNESS_PROMPT.encode()).hexdigest(),
    }


class JudgeParseError(ValueError):
    pass


def extract_json(text: str) -> dict:
    """Parse the first JSON object in ``text`` (tolerates code fences / prose)."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise JudgeParseError(f"no JSON object in judge output: {text[:200]!r}")


def parse_correctness(text: str) -> dict:
    obj = extract_json(text)
    score = obj.get("score")
    if isinstance(score, bool) or score not in (0, 1, 2):
        raise JudgeParseError(f"correctness score must be 0, 1 or 2, got {score!r}")
    return {"score": score, "normalized": score / 2, "rationale": str(obj.get("rationale", ""))}


def parse_faithfulness(text: str) -> dict:
    obj = extract_json(text)
    claims = obj.get("claims")
    if not isinstance(claims, list) or not all(
        isinstance(c, dict) and isinstance(c.get("supported"), bool) for c in claims
    ):
        raise JudgeParseError("faithfulness output needs a 'claims' list with boolean 'supported'")
    supported = sum(c["supported"] for c in claims)
    return {
        # No factual claims (e.g. a pure abstention) -> undefined, not 1.0.
        "score": supported / len(claims) if claims else None,
        "claims": claims,
        "unsupported_claims": [c.get("claim") for c in claims if not c["supported"]],
    }


@dataclass
class GroqJudge:
    model: str
    api_key: str
    temperature: float = 0.0

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for --judge")
        from langchain_groq import ChatGroq

        self._llm = ChatGroq(model=self.model, api_key=self.api_key, temperature=self.temperature)

    def config(self) -> dict:
        return {
            "enabled": True,
            "provider": "groq",
            "model": self.model,
            "temperature": self.temperature,
            **prompt_fingerprint(),
        }

    def _ask(self, prompt: str) -> str:
        return str(getattr(self._llm.invoke(prompt), "content", "")).strip()

    def correctness(self, question: str, reference: str, candidate: str) -> dict:
        return parse_correctness(
            self._ask(CORRECTNESS_PROMPT.format(question=question, reference=reference, candidate=candidate))
        )

    def faithfulness(self, question: str, context: str, answer: str) -> dict:
        return parse_faithfulness(
            self._ask(FAITHFULNESS_PROMPT.format(question=question, context=context, answer=answer))
        )
