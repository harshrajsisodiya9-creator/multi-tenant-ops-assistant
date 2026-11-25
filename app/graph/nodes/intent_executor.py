from __future__ import annotations

from app.graph.agent_factory import get_react_agent
from app.graph.conversation_store import conversation_store
from app.graph.data_access import get_order_scoped
from app.graph.state import AgentState


def _execute_order_status(tenant_id: str, customer_id: str, entities: dict) -> str:
    order_id = entities.get("order_id", "")
    order = get_order_scoped(tenant_id, customer_id, order_id)
    if not order:
        return f"I couldn't find order {order_id} in our system."
    status = order.get("status", "unknown")
    product = order.get("product", "your item")
    delivery = order.get("delivery_date")
    if delivery:
        return f"Order {order_id} ({product}) status: {status}. Delivered on {delivery}."
    return f"Order {order_id} ({product}) status: {status}."


def _execute_refund(tenant_id: str, customer_id: str, entities: dict, constraints: dict) -> str:
    order_id = entities.get("order_id", "")
    order = get_order_scoped(tenant_id, customer_id, order_id)
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


def _execute_cancel(tenant_id: str, customer_id: str, entities: dict) -> str:
    order_id = entities.get("order_id", "")
    order = get_order_scoped(tenant_id, customer_id, order_id)
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
    customer_id: str | None,
    intent: str,
    entities: dict,
    constraints: dict,
    user_message: str,
) -> dict:
    run_react_agent = get_react_agent()
    order_id = entities.get("order_id", "")

    from app.config import setting
    if setting.agent_mode == "prod":
        from app.graph.agent_prod.tool_context import ToolContext
        ctx = ToolContext(tenant_id=tenant_id, customer_id=customer_id or "")
        agent_result = run_react_agent(
            ctx=ctx,
            order_id=order_id,
            user_message=user_message,
            intent=intent,
            constraints=constraints,
        )
    else:
        agent_result = run_react_agent(
            tenant_id=tenant_id,
            order_id=order_id,
            user_message=user_message,
            intent=intent,
            constraints=constraints,
        )
    return agent_result


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


def _build_clarify_context(reason_code: str, constraints: dict | None = None) -> str:
    if reason_code == "AMBIGUOUS_ORDER" and constraints and constraints.get("matching_orders"):
        orders = constraints["matching_orders"]
        listing = "\n".join(f"  • {o}" for o in orders)
        return f"I found multiple orders that could match. Which one did you mean?\n{listing}"

    if reason_code == "MISSING_ORDER_ID" and constraints and constraints.get("order_count"):
        count = constraints["order_count"]
        return f"I see {count} orders on your account. Could you share the order number so I can pull up the right one?"

    messages = {
        "LOW_CONFIDENCE": "Could you rephrase that? I want to make sure I understand your request correctly.",
        "VERIFICATION_REQUIRED": "Could you share your order number or the email address associated with your account?",
        "IDENTITY_MISMATCH": "The order you referenced doesn't match the email address on file. Could you double-check?",
        "NO_ORDERS_FOUND": "I couldn't find any orders associated with your account. Could you check the email address or order number?",
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

AGENT_INTENTS: set[str] = set()


# ---------------------------------------------------------------------------
# Send()-targeted executor nodes
# ---------------------------------------------------------------------------

def _extract_order_context(tool_calls: list[dict]) -> list[dict]:
    orders: list[dict] = []
    for tc in tool_calls:
        if tc.get("blocked"):
            continue
        if tc.get("tool") == "get_order_details" and isinstance(tc.get("result"), dict):
            r = tc["result"]
            if r.get("order_id"):
                orders.append({
                    "order_id": r["order_id"],
                    "product": r.get("product", ""),
                    "status": r.get("status", ""),
                    "amount": r.get("amount", 0),
                })
    return orders


def lookup_agent_node(state: AgentState) -> dict:
    """Read-only ReAct agent: fetches order details and checks eligibility."""
    result = dict(state["current_intent"])
    tenant_id = state["tenant_id"]
    customer_id = state.get("customer_id")
    user_message = state["user_message"]
    entities = result.get("entities", {})
    constraints = result.get("constraints", {})

    agent_out = _run_react_agent(
        tenant_id, customer_id, result["intent"],
        entities, constraints, user_message,
    )
    result["tool_calls"] = agent_out.get("tool_calls", [])

    order_ctx = _extract_order_context(result["tool_calls"])
    if order_ctx:
        conversation_id = state.get("conversation_id", "")
        conversation_store.update_orders_context(
            tenant_id, customer_id or "", conversation_id, order_ctx,
        )

    if agent_out.get("pending_approval"):
        pending = agent_out["pending_approval"]
        result["pending_tool_call"] = pending
        result["domain_result"] = (
            f"Action '{pending['tool']}' with args {pending['args']} "
            f"requires human approval ({', '.join(pending['reasons'])})."
        )
        result["reason_code"] = pending["reasons"][0]
        result["status"] = "awaiting_human"
    else:
        result["domain_result"] = agent_out["result"]
        result["status"] = "fulfilled"

    return {"processed_results": [result]}


def knowledge_lookup_node(state: AgentState) -> dict:
    """Read-only RAG lookup: answers FAQs and complaints via hybrid search."""
    result = dict(state["current_intent"])
    tenant_id = state["tenant_id"]
    user_message = state["user_message"]

    result["domain_result"] = _execute_faq(tenant_id, user_message)
    result["status"] = "fulfilled"
    return {"processed_results": [result]}


def quick_resolver_node(state: AgentState) -> dict:
    """Fast read-only resolver: status lookups, canned responses, blocks, and clarifications."""
    result = dict(state["current_intent"])
    tenant_id = state["tenant_id"]
    customer_id = state.get("customer_id")
    entities = result.get("entities", {})
    constraints = result.get("constraints", {})
    action = result["policy_action"]
    intent = result["intent"]

    if action == "allow":
        if intent in CONVERSATIONAL_INTENTS:
            result["domain_result"] = CONVERSATIONAL_RESPONSES[intent]
        elif intent == "order_status":
            result["domain_result"] = _execute_order_status(
                tenant_id, customer_id or "", entities,
            )
        result["status"] = "fulfilled"

    elif action == "block":
        result["domain_result"] = _build_block_context(result["reason_code"], entities)
        result["status"] = "blocked"

    elif action == "escalate":
        result["domain_result"] = _build_escalation_context(result["reason_code"], entities)
        result["status"] = "awaiting_human"

    elif action == "clarify":
        result["domain_result"] = _build_clarify_context(result["reason_code"], constraints)
        result["status"] = "needs_clarification"

    return {"processed_results": [result]}


def bail_executor_node(state: AgentState) -> dict:
    """Handles pre_check bail results (block/clarify from tenant-inactive or low-confidence)."""
    intent_results = list(state.get("intent_results") or [])

    for result in intent_results:
        if result.get("status") != "pending":
            continue
        action = result.get("policy_action", "block")
        entities = result.get("entities", {})
        constraints = result.get("constraints", {})

        if action == "block":
            result["domain_result"] = _build_block_context(
                result.get("reason_code", "TENANT_INACTIVE"), entities,
            )
            result["status"] = "blocked"
        elif action == "clarify":
            result["domain_result"] = _build_clarify_context(
                result.get("reason_code", "LOW_CONFIDENCE"), constraints,
            )
            result["status"] = "needs_clarification"
        elif action == "escalate":
            result["domain_result"] = _build_escalation_context(
                result.get("reason_code", "DEFAULT_DENY"), entities,
            )
            result["status"] = "awaiting_human"
        else:
            result["domain_result"] = "This request cannot be processed at this time."
            result["status"] = "blocked"

    return {
        "intent_results": intent_results,
        "response_phase": "complete",
        "has_tool_calls": False,
    }


def merge_results_node(state: AgentState) -> dict:
    """Combines parallel processed_results into intent_results, computes routing flags."""
    processed = state.get("processed_results") or []
    has_tool_calls = any(r.get("tool_calls") for r in processed)
    has_escalations = any(r.get("status") == "awaiting_human" for r in processed)
    response_phase = "partial" if has_escalations else "complete"

    return {
        "intent_results": processed,
        "response_phase": response_phase,
        "has_tool_calls": has_tool_calls,
    }
