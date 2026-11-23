from __future__ import annotations

import json
import logging

from langgraph.types import interrupt

from app.graph.state import AgentState

logger = logging.getLogger(__name__)


def _execute_pending_tool(pending: dict, tenant_id: str, customer_id: str) -> str:
    from app.graph.agent_prod.tool_context import ToolContext
    from app.graph.agent_prod.tools import TOOL_REGISTRY

    tool_name = pending["tool"]
    tool_fn = TOOL_REGISTRY.get(tool_name)
    if not tool_fn:
        return f"Unknown tool '{tool_name}'."

    ctx = ToolContext(tenant_id=tenant_id, customer_id=customer_id)
    try:
        result = tool_fn(ctx, **pending["args"])
        return json.dumps(result, default=str) if isinstance(result, dict) else str(result)
    except Exception:
        logger.exception("Failed to execute approved tool call %s", tool_name)
        return f"Tool '{tool_name}' execution failed. Please retry or process manually."


def escalation_handler_node(state: AgentState) -> dict:
    intent_results: list[dict] = list(state.get("intent_results") or [])

    pending_escalations = []
    for i, result in enumerate(intent_results):
        if result.get("status") == "awaiting_human":
            escalation = {
                "index": i,
                "intent": result["intent"],
                "reason_code": result.get("reason_code", ""),
                "domain_result": result.get("domain_result", ""),
                "entities": result.get("entities", {}),
            }
            if result.get("pending_tool_call"):
                escalation["pending_tool_call"] = result["pending_tool_call"]
            pending_escalations.append(escalation)

    if not pending_escalations:
        return {}

    human_decisions = interrupt(pending_escalations)

    decisions = human_decisions if isinstance(human_decisions, list) else human_decisions.get("decisions", [])
    tenant_id = state.get("tenant_id", "")
    customer_id = state.get("customer_id") or ""

    for decision in decisions:
        idx = decision.get("index")
        action = decision.get("action", "deny")

        if idx is None or idx >= len(intent_results):
            continue

        result = intent_results[idx]

        if action == "approve":
            pending = result.get("pending_tool_call")
            if pending:
                tool_result = _execute_pending_tool(pending, tenant_id, customer_id)
                result["domain_result"] = tool_result
                result["status"] = "fulfilled"
            else:
                note = decision.get("note", "")
                original = result.get("domain_result", "")
                result["domain_result"] = f"{original} Approved by support team." + (f" Note: {note}" if note else "")
                result["status"] = "fulfilled"
            result["_newly_resolved"] = True

        elif action == "deny":
            result["status"] = "blocked"
            result["reason_code"] = "HUMAN_DENIED"
            result["domain_result"] = decision.get("reason", "This request has been reviewed and cannot be processed.")
            result["_newly_resolved"] = True

        elif action == "override":
            result["status"] = "fulfilled"
            result["domain_result"] = decision.get("custom_response", "")
            result["_newly_resolved"] = True

    return {
        "intent_results": intent_results,
        "response_phase": "follow_up",
    }
