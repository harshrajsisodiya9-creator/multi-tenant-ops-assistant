from __future__ import annotations

from app.graph.data_access import (
    get_customer_orders,
    get_order_scoped,
    get_refund_count_30d,
    get_tenant_info,
)
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
    customer_id: str,
    intent: str,
    entities: dict,
    user_message: str,
) -> tuple[dict, PolicyDecision | None]:
    if intent not in ORDER_INTENTS or entities.get("order_id"):
        return entities, None

    if not customer_id:
        return entities, None

    customer_orders = get_customer_orders(tenant_id, customer_id)
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


def _expand_multi_order_intents(
    all_intents: list[tuple[str, dict, float]],
) -> list[tuple[str, dict, float]]:
    """When multiple order IDs are present, produce one intent per unique order."""
    if not all_intents:
        return all_intents

    primary_intent, primary_entities, primary_conf = all_intents[0]
    all_ids = primary_entities.get("all_order_ids", [])

    if len(all_ids) <= 1:
        return all_intents

    assigned: set[str] = set()
    expanded: list[tuple[str, dict, float]] = []

    for intent, entities, conf in all_intents:
        oid = entities.get("order_id")
        if oid and oid in all_ids and oid not in assigned:
            assigned.add(oid)
            clean = {k: v for k, v in entities.items() if k != "all_order_ids"}
            expanded.append((intent, clean, conf))

    for oid in all_ids:
        if oid not in assigned:
            clean = {k: v for k, v in primary_entities.items() if k != "all_order_ids"}
            clean["order_id"] = oid
            expanded.append((primary_intent, clean, primary_conf))

    return expanded


def _build_gate_context(
    tenant_id: str,
    customer_id: str,
    tenant_info: dict,
    intent: str,
    entities: dict,
    confidence: float,
    escalation_signals: list[str],
) -> GateContext:
    order = None
    order_id = entities.get("order_id")
    if order_id and customer_id:
        order = get_order_scoped(tenant_id, customer_id, order_id)

    refund_count = 0
    if customer_id:
        refund_count = get_refund_count_30d(tenant_id, customer_id)

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
        customer_id=customer_id,
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
    customer_id = state.get("customer_id") or ""
    tenant_info = get_tenant_info(tenant_id) or {"active": True}

    conditional_data = router_output.conditional.model_dump() if router_output.conditional else None

    all_intents: list[tuple[str, dict, float]] = [
        (router_output.intent, router_output.entities, router_output.confidence),
    ]
    for sub in router_output.sub_intents:
        all_intents.append((sub.intent, sub.entities, sub.confidence))

    all_intents = _expand_multi_order_intents(all_intents)

    intent_results: list[dict] = []
    primary_decision = None

    user_message = state.get("user_message", "")

    for i, (intent, entities, confidence) in enumerate(all_intents):
        skip = _skip_gate(intent, router_output.escalation_signals)
        if skip:
            decision = skip
        else:
            entities, early_clarify = _resolve_order_entity(
                tenant_id, customer_id, intent, entities, user_message,
            )
            if early_clarify:
                decision = early_clarify
            else:
                ctx = _build_gate_context(
                    tenant_id=tenant_id,
                    customer_id=customer_id,
                    tenant_info=tenant_info,
                    intent=intent,
                    entities=entities,
                    confidence=confidence,
                    escalation_signals=router_output.escalation_signals,
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
            conditional=conditional_data if i == 0 else None,
        )
        intent_results.append(result.model_dump())

    return {
        "intent_results": intent_results,
        "policy_decision": primary_decision.model_dump() if primary_decision else {},
    }
