from __future__ import annotations

from app.graph.mock_data import get_order
from app.graph.policy_gate import CANCELLABLE_STATUSES, GLOBAL_REFUND_CEILING
from app.graph.state import AgentState

# Compound actions are evaluated independently; no rollback across actions.
# Production would wrap these in a transaction.


def _check_refund_tool_call(tool_call: dict, tenant_id: str) -> str | None:
    args = tool_call.get("args", {})
    amount = args.get("amount", 0)

    if amount > GLOBAL_REFUND_CEILING:
        return f"Refund amount ₹{amount:,.0f} exceeds global ceiling ₹{GLOBAL_REFUND_CEILING:,}."

    order_id = args.get("order_id", "")
    order = get_order(tenant_id, order_id)
    if order:
        order_amount = order.get("total_amount", order.get("amount", 0))
        if amount > order_amount:
            return f"Refund amount ₹{amount:,.0f} exceeds order total ₹{order_amount:,}."

    return None


def _check_cancel_tool_call(tool_call: dict, tenant_id: str) -> str | None:
    args = tool_call.get("args", {})
    order_id = args.get("order_id", "")
    order = get_order(tenant_id, order_id)

    if not order:
        return None

    status = order.get("order_status", order.get("status", "")).lower()
    if status not in CANCELLABLE_STATUSES:
        return f"Order {order_id} is '{status}' — not cancellable."

    return None


TOOL_CHECKERS = {
    "initiate_refund": _check_refund_tool_call,
    "cancel_order": _check_cancel_tool_call,
}


def action_check_node(state: AgentState) -> dict:
    intent_results: list[dict] = list(state.get("intent_results") or [])
    tenant_id = state.get("tenant_id", "")
    violations: list[str] = []

    for result in intent_results:
        if result.get("status") != "fulfilled":
            continue

        for tc in result.get("tool_calls", []):
            if tc.get("blocked"):
                continue

            tool_name = tc.get("tool", "")
            checker = TOOL_CHECKERS.get(tool_name)
            if checker:
                violation = checker(tc, tenant_id)
                if violation:
                    violations.append(violation)
                    result["status"] = "blocked"
                    result["domain_result"] = f"Action blocked by safety check: {violation}"
                    result["reason_code"] = "ACTION_CHECK_VIOLATION"
                    break

    return {"intent_results": intent_results}
