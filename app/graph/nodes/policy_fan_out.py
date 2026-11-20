from __future__ import annotations

from app.graph.mock_data import get_order, get_orders_by_email, get_refund_count_30d, get_tenant_info
from app.graph.nodes.pre_check import CONVERSATIONAL_INTENTS
from app.graph.policy_gate import GateContext, ORDER_INTENTS, evaluate_policy
from app.graph.schemas import IntentResult, PolicyDecision, RouterOutput, SubIntent
from app.graph.state import AgentState

GATE_SKIP_INTENTS = CONVERSATIONAL_INTENTS | {"faq"}
COMPLAINT_SKIP_WITHOUT_ESCALATION = True


_STOP_WORDS = {"the", "and", "for", "with", "from", "this", "that", "plus", "set", "box", "kit"}


def _product_matches_message(product_name: str, message_lower: str) -> bool:
    keywords = [w.lower() for w in product_name.split() if len(w) > 3 and w.lower() not in _STOP_WORDS]
    message_words = set(message_lower.split())
    return any(kw in message_words or kw.rstrip("s") in {w.rstrip("s") for w in message_words} for kw in keywords)


def _resolve_order_entity(
    tenant_id: str,
    intent: str,
    entities: dict,
    customer_email: str | None,
    user_message: str,
) -> tuple[dict, PolicyDecision | None]:
    if intent not in ORDER_INTENTS or entities.get("order_id"):
        return entities, None

    email = customer_email or entities.get("email")
    if not email:
        return entities, None

    customer_orders = get_orders_by_email(tenant_id, email)
    if not customer_orders:
        return entities, PolicyDecision(action="clarify", reason_code="NO_ORDERS_FOUND")

    msg_lower = user_message.lower()
    matching = []
    for order in customer_orders:
        if _product_matches_message(order.get("product", ""), msg_lower):
            matching.append(order)
            continue
        for item in order.get("items", []):
            if _product_matches_message(item.get("product", ""), msg_lower):
                matching.append(order)
                break

    if len(matching) == 1:
        return {**entities, "order_id": matching[0]["order_id"]}, None

    if len(matching) > 1:
        summaries = [
            f"{o['order_id']} — {o.get('product', 'unknown')} (₹{o.get('amount', 0):,})"
            for o in matching
        ]
        return entities, PolicyDecision(
            action="clarify",
            reason_code="AMBIGUOUS_ORDER",
            constraints={"matching_orders": summaries},
        )

    if len(customer_orders) == 1:
        return {**entities, "order_id": customer_orders[0]["order_id"]}, None

    return entities, PolicyDecision(
        action="clarify",
        reason_code="MISSING_ORDER_ID",
        constraints={"order_count": len(customer_orders)},
    )


def _build_gate_context(
    tenant_id: str,
    tenant_info: dict,
    intent: str,
    entities: dict,
    confidence: float,
    escalation_signals: list[str],
    customer_email: str | None,
) -> GateContext:
    order = None
    order_id = entities.get("order_id")
    if order_id:
        order = get_order(tenant_id, order_id)

    refund_count = 0
    email = customer_email or entities.get("email")
    if email:
        refund_count = get_refund_count_30d(tenant_id, email)

    ro = RouterOutput(
        intent=intent,
        entities=entities,
        confidence=confidence,
        escalation_signals=escalation_signals,
    )

    return GateContext(
        tenant_id=tenant_id,
        tenant_active=tenant_info.get("active", True),
        router_output=ro,
        customer_email=email,
        order=order,
        refund_count_30d=refund_count,
    )


def _skip_gate(intent: str, escalation_signals: list[str]) -> PolicyDecision | None:
    if intent in GATE_SKIP_INTENTS:
        reason_map = {
            "faq": "FAQ_ALLOWED",
            "greeting": "GREETING_ALLOWED",
            "thanks": "THANKS_ALLOWED",
            "chitchat": "CHITCHAT_ALLOWED",
            "goodbye": "GOODBYE_ALLOWED",
        }
        return PolicyDecision(
            action="allow",
            reason_code=reason_map[intent],
        )

    if intent == "complaint" and not escalation_signals:
        return PolicyDecision(
            action="allow",
            reason_code="COMPLAINT_ALLOWED",
        )

    return None


def policy_fan_out_node(state: AgentState) -> dict:
    if state.get("pre_check_bail"):
        return {}

    router_output = RouterOutput(**state["router_output"])
    tenant_id = state["tenant_id"]
    tenant_info = get_tenant_info(tenant_id) or {"active": True}
    customer_email = state.get("customer_email") or router_output.entities.get("email")

    all_intents: list[tuple[str, dict, float]] = [
        (router_output.intent, router_output.entities, router_output.confidence),
    ]
    for sub in router_output.sub_intents:
        all_intents.append((sub.intent, sub.entities, sub.confidence))

    intent_results: list[dict] = []
    primary_decision = None

    user_message = state.get("user_message", "")

    for i, (intent, entities, confidence) in enumerate(all_intents):
        skip = _skip_gate(intent, router_output.escalation_signals)
        if skip:
            decision = skip
        else:
            entities, early_clarify = _resolve_order_entity(
                tenant_id, intent, entities, customer_email, user_message,
            )
            if early_clarify:
                decision = early_clarify
            else:
                ctx = _build_gate_context(
                    tenant_id=tenant_id,
                    tenant_info=tenant_info,
                    intent=intent,
                    entities=entities,
                    confidence=confidence,
                    escalation_signals=router_output.escalation_signals,
                    customer_email=customer_email,
                )
                decision = evaluate_policy(ctx)

        if i == 0:
            primary_decision = decision

        result = IntentResult(
            intent=intent,
            entities=entities,
            policy_action=decision.action,
            reason_code=decision.reason_code,
            constraints=decision.constraints,
            status="pending",
        )
        intent_results.append(result.model_dump())

    return {
        "intent_results": intent_results,
        "policy_decision": primary_decision.model_dump() if primary_decision else {},
    }
