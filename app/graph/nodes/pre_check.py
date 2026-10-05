from __future__ import annotations

from app.graph.mock_data import get_tenant_info
from app.graph.schemas import IntentResult, RouterOutput
from app.graph.state import AgentState

CONVERSATIONAL_INTENTS = {"greeting", "thanks", "chitchat", "goodbye"}


def pre_check_node(state: AgentState) -> dict:
    router_output = RouterOutput(**state["router_output"])
    tenant_id = state["tenant_id"]
    tenant_info = get_tenant_info(tenant_id) or {"active": True}

    if not tenant_info.get("active", True):
        result = IntentResult(
            intent=router_output.intent,
            entities=router_output.entities,
            policy_action="block",
            reason_code="TENANT_INACTIVE",
            status="pending",
        )
        return {
            "intent_results": [result.model_dump()],
            "policy_decision": {"action": "block", "reason_code": "TENANT_INACTIVE", "constraints": {}},
            "pre_check_bail": True,
        }

    if router_output.confidence < 0.6:
        result = IntentResult(
            intent=router_output.intent,
            entities=router_output.entities,
            policy_action="clarify",
            reason_code="LOW_CONFIDENCE",
            status="pending",
        )
        return {
            "intent_results": [result.model_dump()],
            "policy_decision": {"action": "clarify", "reason_code": "LOW_CONFIDENCE", "constraints": {}},
            "pre_check_bail": True,
        }

    if router_output.intent == "out_of_scope":
        result = IntentResult(
            intent="out_of_scope",
            entities=router_output.entities,
            policy_action="block",
            reason_code="OUT_OF_SCOPE",
            status="pending",
        )
        return {
            "intent_results": [result.model_dump()],
            "policy_decision": {"action": "block", "reason_code": "OUT_OF_SCOPE", "constraints": {}},
            "pre_check_bail": True,
        }

    return {"pre_check_bail": False}
