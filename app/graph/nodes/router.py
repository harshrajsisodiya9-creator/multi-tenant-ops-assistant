from __future__ import annotations

import re

from langchain_groq import ChatGroq

from app.config import setting
from app.graph.schemas import RouterOutput
from app.graph.state import AgentState

ESCALATION_KEYWORDS = [
    "chargeback",
    "legal",
    "lawyer",
    "fraud",
    "sue",
    "attorney",
    "court",
    "lawsuit",
    "consumer forum",
    "police",
]

ORDER_ID_PATTERN = re.compile(r"(?:ORD-[\w]+-\d+|#\d{4,})", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

CONDITIONAL_PATTERNS = [
    "if it hasn't",
    "if it has not",
    "if it's not",
    "if it is not",
    "if my order hasn't",
    "if my order has not",
    "if the order hasn't",
    "if the order has not",
    "if it isn't",
    "if it's still",
    "if it is still",
    "only if",
    "in case it",
    "in case my",
    "if not shipped",
    "if not delivered",
    "if still processing",
    "if still pending",
    "hasn't shipped",
    "has not shipped",
    "hasn't been delivered",
    "has not been delivered",
    "hasn't arrived",
    "has not arrived",
]

CONDITIONAL_QUESTION_SIGNALS = [
    "is my order",
    "has my order",
    "has it been",
    "is it still",
    "can i still",
    "am i eligible",
    "is it refundable",
    "is it returnable",
    "can i return",
    "can i get a refund",
    "check if",
    "tell me if",
    "let me know if",
]

CONJUNCTION_SIGNALS = ["and also", "and", "also", "plus", "as well as", "along with"]

CONVERSATIONAL_RULES: list[tuple[list[str], str, float]] = [
    (
        [
            "hello",
            "hi ",
            "hi!",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ],
        "greeting",
        0.95,
    ),
    (["thank you", "thanks", "thx", "appreciate it", "thank u"], "thanks", 0.95),
    (["bye", "goodbye", "see you", "take care", "good night"], "goodbye", 0.95),
]

_KEYWORD_RULES: list[tuple[list[str], str, float]] = [
    (
        ["track my order", "where is my order", "order status", "shipping status"],
        "order_status",
        0.95,
    ),
    (["cancel my order", "cancel order", "cancel the order"], "cancel_order", 0.92),
    (["cancel my subscription", "cancel subscription"], "cancel_order", 0.92),
    (["refund", "money back", "return my", "return the"], "refund", 0.88),
]

FAQ_INDICATORS = [
    "how to",
    "how do",
    "how can",
    "what is",
    "what's the",
    "policy",
    "process",
    "procedure",
    "guide",
    "steps",
    "tell me about",
    "explain",
    "information",
]

ROUTER_SYSTEM_PROMPT = """\
You are an intent classifier for an e-commerce customer support system.

Classify the user message into one or more intents:
- faq: general questions about policies, products, shipping, etc.
- order_status: asking about where an order is, tracking, delivery ETA
- refund: requesting a refund or return
- cancel_order: requesting to cancel an order or subscription
- complaint: expressing dissatisfaction, reporting a problem
- out_of_scope: anything unrelated to e-commerce support (weather, jokes, etc.)
- greeting: hello, hi, good morning, etc.
- thanks: thank you, appreciate it, etc.
- chitchat: casual conversation not fitting other categories
- goodbye: bye, see you, take care, etc.

Extract entities:
- order_id: look for patterns like ORD-XX-1234 or #1234 (there may be multiple)
- email: any email address mentioned

Return a confidence score between 0 and 1.

Detect escalation signals: words like chargeback, legal, lawyer, fraud, sue, attorney, court.

Detect conditional requests — messages that say "if <condition>, do <action>" or ask \
whether a condition is true. When you detect one, populate the `conditional` field:
- request_type: "conditional_action" if the user wants an action performed when the condition \
  is met (e.g. "If my order hasn't shipped, cancel it"), or "conditional_question" if the \
  user is only asking whether the condition holds (e.g. "Has my order shipped yet?").
- condition.field: one of "order_status", "fulfillment_status", "is_refundable", "is_return_eligible"
- condition.operator: one of "equals", "not_equals", "in", "not_in", "is_true", "is_false"
- condition.value: the target value(s) to compare against (e.g. "shipped", ["shipped", "delivered"])

For boolean fields (is_refundable, is_return_eligible) use is_true/is_false operators.
For status fields use equals/not_equals/in/not_in operators.
Leave `conditional` as null when there is no conditional logic in the message.

Set the primary intent as the strongest one. If the message contains multiple distinct intents \
(e.g. "refund this AND check my other order"), list the additional ones as sub_intents with \
their own entities and confidence. Only split into sub_intents when there are genuinely \
separate requests — a single request with context is one intent."""


def _extract_entities(message: str) -> dict:
    entities: dict = {}
    order_matches = [m.upper() for m in ORDER_ID_PATTERN.findall(message)]
    if order_matches:
        entities["order_id"] = order_matches[0]
        if len(order_matches) > 1:
            entities["all_order_ids"] = order_matches
    email_match = EMAIL_PATTERN.search(message)
    if email_match:
        entities["email"] = email_match.group()
    return entities


def _normalize_order_entities(target: dict, regex_entities: dict) -> None:
    """Ensure order_id is always a single string; extras go to all_order_ids."""
    raw = target.get("order_id")

    if isinstance(raw, list):
        ids = list(dict.fromkeys(str(x).upper() for x in raw if x))
    elif raw:
        ids = [str(raw).upper()]
    else:
        ids = []

    for rid in regex_entities.get("all_order_ids", []):
        if rid not in ids:
            ids.append(rid)

    if not ids:
        target.pop("order_id", None)
        target.pop("all_order_ids", None)
        return

    target["order_id"] = ids[0]
    if len(ids) > 1:
        target["all_order_ids"] = ids
    else:
        target.pop("all_order_ids", None)


def _detect_escalation(message: str) -> list[str]:
    lower = message.lower()
    return [kw for kw in ESCALATION_KEYWORDS if kw in lower]


def _is_informational(message_lower: str) -> bool:
    return any(indicator in message_lower for indicator in FAQ_INDICATORS)


def _has_conjunction(message_lower: str) -> bool:
    return any(signal in message_lower for signal in CONJUNCTION_SIGNALS)


def _count_keyword_matches(message_lower: str) -> list[tuple[str, float]]:
    matches = []
    for keywords, intent, confidence in _KEYWORD_RULES:
        if any(kw in message_lower for kw in keywords):
            matches.append((intent, confidence))
    return matches


def _check_conversational(message_lower: str) -> tuple[str, float] | None:
    for keywords, intent, confidence in CONVERSATIONAL_RULES:
        if any(kw in message_lower for kw in keywords):
            return intent, confidence
    return None


def _detect_conditional(message_lower: str) -> bool:
    return any(pattern in message_lower for pattern in CONDITIONAL_PATTERNS)


def _detect_conditional_question(message_lower: str) -> bool:
    return any(signal in message_lower for signal in CONDITIONAL_QUESTION_SIGNALS)


def _pre_check(message: str) -> tuple[str | None, dict, float, list[str], bool]:
    entities = _extract_entities(message)
    escalation = _detect_escalation(message)
    lower = message.lower()
    is_conditional = _detect_conditional(lower) or _detect_conditional_question(lower)

    conv = _check_conversational(lower)
    if conv and not _count_keyword_matches(lower):
        intent, confidence = conv
        return intent, entities, confidence, escalation, False

    if _is_informational(lower):
        return None, entities, 0.0, escalation, is_conditional

    matches = _count_keyword_matches(lower)

    if len(matches) > 1 or (matches and _has_conjunction(lower)):
        return None, entities, 0.0, escalation, is_conditional

    if matches:
        intent, confidence = matches[0]
        return intent, entities, confidence, escalation, is_conditional

    return None, entities, 0.0, escalation, is_conditional


def _build_context_prefix(state: AgentState) -> str:
    parts: list[str] = []
    history = state.get("conversation_history") or []
    if history:
        lines = []
        for m in history:
            role = "Customer" if m["role"] == "human" else "Assistant"
            lines.append(f"{role}: {m['content']}")
        parts.append(
            "These are the previous conversations treat them as context:\n"
            + "\n".join(lines)
        )

    orders = state.get("orders_context") or []
    if orders:
        order_strs = [
            f"  {o.get('order_id', '?')} — {o.get('product', '?')} ({o.get('status', '?')})"
            for o in orders
        ]
        parts.append("Known orders:\n" + "\n".join(order_strs))

    return "\n\n".join(parts)


def router_node(state: AgentState) -> dict:
    message = state["user_message"]
    intent, entities, confidence, escalation, is_conditional = _pre_check(message)

    if intent and confidence >= 0.9 and not is_conditional:
        router_output = RouterOutput(
            intent=intent,
            entities=entities,
            confidence=confidence,
            escalation_signals=escalation,
        )
        return {"router_output": router_output.model_dump()}

    llm = ChatGroq(
        model=setting.groq_model,
        api_key=setting.groq_api_key,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(RouterOutput)

    context_prefix = _build_context_prefix(state)
    user_content = f"Current message: {message}"

    result: RouterOutput = structured_llm.invoke(
        [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "system", "content": "context:\n" + context_prefix}
            if context_prefix
            else {"role": "system", "content": "context:\n<none>"},
            {"role": "user", "content": user_content},
        ]
    )  # type: ignore

    if entities.get("order_id") and not result.entities.get("order_id"):
        result.entities["order_id"] = entities["order_id"]
    if entities.get("email") and not result.entities.get("email"):
        result.entities["email"] = entities["email"]

    if escalation and not result.escalation_signals:
        result.escalation_signals = escalation

    _normalize_order_entities(result.entities, entities)
    for sub in result.sub_intents:
        _normalize_order_entities(sub.entities, {})

    return {"router_output": result.model_dump()}
