from __future__ import annotations

from app.graph.mock_data import get_order, get_refund_count_30d, get_tenant_info
from app.graph.nodes.pre_check import CONVERSATIONAL_INTENTS
from app.graph.policy_gate import GateContext, evaluate_policy
from app.graph.schemas import IntentResult, PolicyDecision, RouterOutput, SubIntent
from app.graph.state import AgentState

GATE_SKIP_INTENTS = CONVERSATIONAL_INTENTS | {"faq"}
COMPLAINT_SKIP_WITHOUT_ESCALATION = True


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

    for i, (intent, entities, confidence) in enumerate(all_intents):
        skip = _skip_gate(intent, router_output.escalation_signals)
        if skip:
            decision = skip
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
