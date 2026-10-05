from __future__ import annotations

from app.graph.agent.react_agent import run_react_agent
from app.graph.mock_data import get_order
from app.graph.state import AgentState


def _execute_order_status(tenant_id: str, entities: dict) -> str:
    order_id = entities.get("order_id", "")
    order = get_order(tenant_id, order_id)
    if not order:
        return f"I couldn't find order {order_id} in our system."
    status = order.get("status", "unknown")
    product = order.get("product", "your item")
    delivery = order.get("delivery_date")
    if delivery:
        return f"Order {order_id} ({product}) status: {status}. Delivered on {delivery}."
    return f"Order {order_id} ({product}) status: {status}."


def _execute_refund(tenant_id: str, entities: dict, constraints: dict) -> str:
    order_id = entities.get("order_id", "")
    order = get_order(tenant_id, order_id)
    if not order:
        return f"Refund initiated for order {order_id}."
    product = order.get("product", "your item")
    amount = order.get("amount", 0)
    refund_type = constraints.get("refund_type", "original payment method")
    condition = constraints.get("condition", "any")
    result = f"Refund of ₹{amount:,} for {product} ({order_id}) approved."
    result += f" Refund via {refund_type}."
    if condition and condition != "any":
        result += f" Condition: item must be {condition}."
    return result


def _execute_cancel(tenant_id: str, entities: dict) -> str:
    order_id = entities.get("order_id", "")
    order = get_order(tenant_id, order_id)
    if not order:
        return f"Cancellation initiated for order {order_id}."
    product = order.get("product", "your item")
    return f"Order {order_id} ({product}) has been cancelled."


def _execute_faq(tenant_id: str, question: str) -> str:
    from uuid import UUID

    from sqlalchemy.orm import Session as SASession

    from app.database import SessionLocal
    from app.hybrid_search import answer_question as rag_answer

    db: SASession = SessionLocal()
    try:
        result = rag_answer(
            db=db,
            tenant_id=UUID(tenant_id),
            question=question,
        )
        return result.get("answer", "I couldn't find an answer.")
    finally:
        db.close()


def _run_react_agent(
    tenant_id: str,
    intent: str,
    entities: dict,
    constraints: dict,
    user_message: str,
) -> tuple[str, list[dict]]:
    order_id = entities.get("order_id", "")
    agent_result = run_react_agent(
        tenant_id=tenant_id,
        order_id=order_id,
        user_message=user_message,
        intent=intent,
        constraints=constraints,
    )
    return agent_result["result"], agent_result.get("tool_calls", [])


def _build_block_context(reason_code: str, entities: dict) -> str:
    order_id = entities.get("order_id", "")
    messages = {
        "RETURNS_NOT_ALLOWED": f"Returns are not allowed for this product category (order {order_id}).",
        "RETURN_WINDOW_EXPIRED": f"The return window has expired for order {order_id}.",
        "ORDER_NOT_CANCELLABLE": f"Order {order_id} cannot be cancelled in its current status.",
        "SUBSCRIPTION_CANCEL_NOT_ALLOWED": "Subscription cancellation is not available for this plan.",
        "SUBSCRIPTION_MIN_COMMITMENT": "You are still within the minimum commitment period for this subscription.",
        "OUT_OF_SCOPE": "This request is outside the scope of our support system.",
        "TENANT_INACTIVE": "This service is currently unavailable.",
    }
    return messages.get(reason_code, f"This request cannot be processed ({reason_code}).")


def _build_escalation_context(reason_code: str, entities: dict) -> str:
    order_id = entities.get("order_id", "")
    messages = {
        "GLOBAL_REFUND_CEILING": f"Refund for order {order_id} exceeds the automatic processing limit.",
        "REFUND_OVER_LIMIT": f"Refund for order {order_id} exceeds the automatic refund limit.",
        "REFUND_RATE_LIMITED": "Multiple recent refund requests detected for this account.",
        "TENANT_REFUND_REVIEW_REQUIRED": f"Refund for order {order_id} requires manual review per store policy.",
        "ESCALATION_KEYWORDS_DETECTED": "Your message has been flagged for priority human review.",
        "DEFAULT_DENY": "This request requires human review.",
    }
    return messages.get(reason_code, f"This request has been escalated for review ({reason_code}).")


def _build_clarify_context(reason_code: str) -> str:
    messages = {
        "LOW_CONFIDENCE": "Could you rephrase that? I want to make sure I understand your request correctly.",
        "VERIFICATION_REQUIRED": "Could you share your order number or the email address associated with your account?",
        "IDENTITY_MISMATCH": "The order you referenced doesn't match the email address on file. Could you double-check?",
    }
    return messages.get(reason_code, "Could you provide more details so I can help?")


STATUS_MAP = {
    "allow": "fulfilled",
    "block": "blocked",
    "escalate": "awaiting_human",
    "clarify": "needs_clarification",
}

CONVERSATIONAL_INTENTS = {"greeting", "thanks", "chitchat", "goodbye"}

CONVERSATIONAL_RESPONSES = {
    "greeting": "Hello! How can I help you today?",
    "thanks": "You're welcome! Is there anything else I can help with?",
    "goodbye": "Goodbye! Feel free to reach out anytime.",
    "chitchat": "I'm here to help with your orders and account. What can I do for you?",
}

AGENT_INTENTS = {"refund", "cancel_order"}


def intent_executor_node(state: AgentState) -> dict:
    intent_results: list[dict] = list(state.get("intent_results") or [])
    tenant_id = state["tenant_id"]
    user_message = state["user_message"]
    has_escalations = False

    for result in intent_results:
        if result.get("status") != "pending":
            continue

        action = result["policy_action"]
        intent = result["intent"]
        entities = result.get("entities", {})
        constraints = result.get("constraints", {})

        if action == "allow":
            if intent in CONVERSATIONAL_INTENTS:
                result["domain_result"] = CONVERSATIONAL_RESPONSES[intent]
            elif intent in ("faq", "complaint"):
                result["domain_result"] = _execute_faq(tenant_id, user_message)
            elif intent == "order_status":
                result["domain_result"] = _execute_order_status(tenant_id, entities)
            elif intent in AGENT_INTENTS:
                domain_result, tool_calls = _run_react_agent(
                    tenant_id, intent, entities, constraints, user_message,
                )
                result["domain_result"] = domain_result
                result["tool_calls"] = tool_calls
            result["status"] = "fulfilled"

        elif action == "block":
            result["domain_result"] = _build_block_context(result["reason_code"], entities)
            result["status"] = "blocked"

        elif action == "escalate":
            result["domain_result"] = _build_escalation_context(result["reason_code"], entities)
            result["status"] = "awaiting_human"
            has_escalations = True

        elif action == "clarify":
            result["domain_result"] = _build_clarify_context(result["reason_code"])
            result["status"] = "needs_clarification"

    response_phase = "partial" if has_escalations else "complete"

    return {
        "intent_results": intent_results,
        "response_phase": response_phase,
    }
