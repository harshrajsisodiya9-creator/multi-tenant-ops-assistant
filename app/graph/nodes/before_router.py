from __future__ import annotations

from app.graph.conversation_store import conversation_store
from app.graph.state import AgentState


def before_router_node(state: AgentState) -> dict:
    tenant_id = state["tenant_id"]
    customer_id = state.get("customer_id") or ""
    conversation_id = state.get("conversation_id", "")

    snapshot = conversation_store.get_snapshot(tenant_id, customer_id, conversation_id)

    conversation_store.add_human_message(
        tenant_id, customer_id, conversation_id, state["user_message"],
    )

    return {
        "conversation_history": snapshot["messages"],
        "customer_context": snapshot["customer_context"],
        "orders_context": snapshot["orders_context"],
        "turn_summaries": snapshot.get("turn_summaries", []),
        # Clear the previous turn's execution state. HITL resumes via Command skip this node.
        "processed_results": None,
        "intent_results": [],
        "router_output": None,
        "policy_decision": None,
        "has_tool_calls": False,
        "response_phase": "complete",
        "pre_check_bail": False,
        "response": "",
    }
