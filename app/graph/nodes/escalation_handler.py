from __future__ import annotations

import json
import logging

from langgraph.types import interrupt

from app.graph.nodes.write_executor import _invoke_read_tool, _invoke_write_tool
from app.graph.state import AgentState

logger = logging.getLogger(__name__)


def _revalidate_pending(
    pending: dict, tenant_id: str, customer_id: str,
) -> str | None:
    """Re-check order state before executing an approved write.

    Returns an error string if the operation is no longer valid, None if OK.
    """
    tool_name = pending["tool"]
    args = pending.get("args", {})
    order_id = args.get("order_id", "")

    if not order_id:
        return None

    order = _invoke_read_tool("get_order_details", tenant_id, customer_id, order_id=order_id)
    if order.get("error"):
        return f"Order is no longer accessible: {order['error']}"

    if tool_name == "cancel_order":
        status = order.get("order_status", "").lower()
        cancellable = {"pending", "processing", "confirmed"}
        if status not in cancellable:
            return f"Order can no longer be cancelled (current status: {status})."

    if tool_name == "initiate_refund":
        eligibility = _invoke_read_tool(
            "get_return_eligibility", tenant_id, customer_id, order_id=order_id,
        )
        if not eligibility.get("eligible"):
            return eligibility.get("reason", "Order is no longer eligible for refund.")

        approved_amount = args.get("amount", 0)
        current_amount = eligibility.get("amount") or order.get("total_amount", 0)
        if approved_amount != current_amount:
            return (
                f"Refund amount has changed since approval was requested "
                f"(approved: {approved_amount}, current: {current_amount})."
            )

    return None


def _execute_pending_tool(pending: dict, tenant_id: str, customer_id: str) -> dict:
    """Execute an approved tool call using the mode-aware factory."""
    tool_name = pending["tool"]
    args = pending.get("args", {})

    try:
        result = _invoke_write_tool(tool_name, tenant_id, customer_id, **args)
        return result
    except Exception:
        logger.exception("Failed to execute approved tool call %s", tool_name)
        return {"success": False, "error": f"Tool '{tool_name}' execution failed."}


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

        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or idx >= len(intent_results):
            continue

        result = intent_results[idx]

        if action == "approve":
            pending = result.get("pending_tool_call")
            if pending:
                stale_reason = _revalidate_pending(pending, tenant_id, customer_id)
                if stale_reason:
                    result["status"] = "failed"
                    result["reason_code"] = "STALE_APPROVAL"
                    result["domain_result"] = (
                        f"This request was approved but can no longer be processed: {stale_reason}"
                    )
                    result["_newly_resolved"] = True
                    continue

                tool_result = _execute_pending_tool(pending, tenant_id, customer_id)

                tool_call_record = {
                    "tool": pending["tool"],
                    "args": pending.get("args", {}),
                    "result": tool_result,
                }

                if tool_result.get("success"):
                    result["domain_result"] = tool_result.get(
                        "message", json.dumps(tool_result, default=str),
                    )
                    result["status"] = "fulfilled"
                else:
                    result["domain_result"] = tool_result.get(
                        "error", "The approved operation failed.",
                    )
                    result["status"] = "failed"

                existing_calls = result.get("tool_calls") or []
                result["tool_calls"] = existing_calls + [tool_call_record]
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
