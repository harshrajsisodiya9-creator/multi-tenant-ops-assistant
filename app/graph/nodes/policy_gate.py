from __future__ import annotations

from app.graph.mock_data import get_order, get_refund_count_30d, get_tenant_info
from app.graph.policy_gate import GateContext, evaluate_policy
from app.graph.schemas import RouterOutput
from app.graph.state import AgentState


def policy_gate_node(state: AgentState) -> dict:
    ro = RouterOutput(**state["router_output"])
    tenant_id = state["tenant_id"]
    tenant_info = get_tenant_info(tenant_id) or {"active": True}

    order = None
    order_id = ro.entities.get("order_id")
    if order_id:
        order = get_order(tenant_id, order_id)

    customer_email = state.get("customer_email") or ro.entities.get("email")
    refund_count = 0
    if customer_email:
        refund_count = get_refund_count_30d(tenant_id, customer_email)

    ctx = GateContext(
        tenant_id=tenant_id,
        tenant_active=tenant_info.get("active", True),
        router_output=ro,
        customer_email=customer_email,
        order=order,
        refund_count_30d=refund_count,
    )

    decision = evaluate_policy(ctx)
    return {"policy_decision": decision.model_dump()}
