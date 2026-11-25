"""Deterministic write executor — no LLM ReAct loop.

Receives structured intent information from the graph (tenant, customer,
intent, entities, constraints, policy decision) and executes the
appropriate predefined workflow using existing service/tool implementations.
"""
from __future__ import annotations

import json
import logging

from app.config import setting
from app.graph.execution_registry import WRITE_INTENTS, is_write_tool
from app.graph.policy_gate import GLOBAL_REFUND_CEILING
from app.graph.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Approval / constraint checks (reused from react_subgraph logic)
# ---------------------------------------------------------------------------

def _hard_constraint_check(tool_name: str, tool_args: dict, constraints: dict) -> str | None:
    if tool_name == "initiate_refund":
        required_type = constraints.get("refund_type")
        if required_type and tool_args.get("refund_type") != required_type:
            return (
                f"Refund type must be '{required_type}', "
                f"got '{tool_args.get('refund_type')}'."
            )

    if tool_name == "cancel_order":
        if constraints.get("cancel_blocked"):
            return "Cancellation is not allowed per policy constraints."

    return None


def _needs_approval(tool_name: str, tool_args: dict, constraints: dict) -> list[str] | None:
    if constraints.get("needs_approval"):
        return constraints.get("approval_reasons", ["REQUIRES_APPROVAL"])

    if tool_name == "initiate_refund":
        amount = tool_args.get("amount", 0)
        if amount > GLOBAL_REFUND_CEILING:
            return ["GLOBAL_REFUND_CEILING"]

    return None


# ---------------------------------------------------------------------------
# Tool invocation helpers (mode-aware via agent_factory)
# ---------------------------------------------------------------------------

def _invoke_tool(tool_name: str, tenant_id: str, customer_id: str, **kwargs) -> dict:
    if setting.agent_mode == "prod":
        from app.graph.agent_prod.tool_context import ToolContext
        from app.graph.agent_prod.tools import TOOL_REGISTRY

        ctx = ToolContext(tenant_id=tenant_id, customer_id=customer_id)
        fn = TOOL_REGISTRY.get(tool_name)
        if not fn:
            return {"success": False, "error": f"Unknown tool '{tool_name}'."}
        return fn(ctx, **kwargs)
    else:
        from app.graph.agent.tools import TOOL_REGISTRY

        fn = TOOL_REGISTRY.get(tool_name)
        if not fn:
            return {"success": False, "error": f"Unknown tool '{tool_name}'."}
        return fn(tenant_id, **kwargs)


def _invoke_read_tool(tool_name: str, tenant_id: str, customer_id: str, **kwargs) -> dict:
    return _invoke_tool(tool_name, tenant_id, customer_id, **kwargs)


def _invoke_write_tool(tool_name: str, tenant_id: str, customer_id: str, **kwargs) -> dict:
    if not is_write_tool(tool_name):
        return {"success": False, "error": f"Tool '{tool_name}' is not a write tool."}
    return _invoke_tool(tool_name, tenant_id, customer_id, **kwargs)


# ---------------------------------------------------------------------------
# Per-intent deterministic workflows
# ---------------------------------------------------------------------------

def _execute_refund_workflow(
    tenant_id: str,
    customer_id: str,
    entities: dict,
    constraints: dict,
) -> dict:
    """Deterministic refund workflow:
    1. Retrieve latest order state
    2. Check return eligibility
    3. Validate constraints
    4. Check approval requirement
    5. Execute refund
    """
    order_id = entities.get("order_id", "")
    tool_calls: list[dict] = []

    # Step 1: Retrieve latest order state
    order_details = _invoke_read_tool(
        "get_order_details", tenant_id, customer_id, order_id=order_id,
    )
    tool_calls.append({"tool": "get_order_details", "args": {"order_id": order_id}, "result": order_details})

    if order_details.get("error"):
        return {
            "domain_result": f"Could not process refund: {order_details['error']}",
            "status": "fulfilled",
            "tool_calls": tool_calls,
        }

    # Step 2: Check return eligibility
    eligibility = _invoke_read_tool(
        "get_return_eligibility", tenant_id, customer_id, order_id=order_id,
    )
    tool_calls.append({"tool": "get_return_eligibility", "args": {"order_id": order_id}, "result": eligibility})

    if not eligibility.get("eligible"):
        reason = eligibility.get("reason", "Not eligible for return/refund.")
        return {
            "domain_result": reason,
            "status": "fulfilled",
            "tool_calls": tool_calls,
        }

    # Step 3: Build refund parameters from policy constraints and order state
    amount = eligibility.get("amount") or order_details.get("total_amount", 0)
    refund_type = constraints.get("refund_type") or eligibility.get("refund_type", "original_method")

    refund_args = {
        "order_id": order_id,
        "amount": amount,
        "refund_type": refund_type,
    }

    # Step 4: Hard constraint check
    violation = _hard_constraint_check("initiate_refund", refund_args, constraints)
    if violation:
        tool_calls.append({
            "tool": "initiate_refund", "args": refund_args,
            "blocked": True, "reason": violation,
        })
        return {
            "domain_result": f"Refund cannot be processed: {violation}",
            "status": "blocked",
            "tool_calls": tool_calls,
        }

    # Step 5: Approval check
    approval_reasons = _needs_approval("initiate_refund", refund_args, constraints)
    if approval_reasons:
        pending = {"tool": "initiate_refund", "args": refund_args, "reasons": approval_reasons}
        tool_calls.append({
            "tool": "initiate_refund", "args": refund_args,
            "blocked": True, "reason": "needs_approval",
        })
        return {
            "domain_result": (
                f"Action 'initiate_refund' with args {refund_args} "
                f"requires human approval ({', '.join(approval_reasons)})."
            ),
            "status": "awaiting_human",
            "pending_tool_call": pending,
            "reason_code_override": approval_reasons[0],
            "tool_calls": tool_calls,
        }

    # Step 6: Execute the refund
    refund_result = _invoke_write_tool(
        "initiate_refund", tenant_id, customer_id, **refund_args,
    )
    tool_calls.append({"tool": "initiate_refund", "args": refund_args, "result": refund_result})

    if refund_result.get("success"):
        return {
            "domain_result": refund_result.get("message", json.dumps(refund_result, default=str)),
            "status": "fulfilled",
            "tool_calls": tool_calls,
        }
    else:
        return {
            "domain_result": refund_result.get("error", "Refund failed."),
            "status": "failed",
            "tool_calls": tool_calls,
        }


def _execute_cancel_workflow(
    tenant_id: str,
    customer_id: str,
    entities: dict,
    constraints: dict,
) -> dict:
    """Deterministic cancellation workflow:
    1. Retrieve latest order state
    2. Validate cancellability
    3. Execute cancellation
    """
    order_id = entities.get("order_id", "")
    tool_calls: list[dict] = []

    # Step 1: Retrieve latest order state
    order_details = _invoke_read_tool(
        "get_order_details", tenant_id, customer_id, order_id=order_id,
    )
    tool_calls.append({"tool": "get_order_details", "args": {"order_id": order_id}, "result": order_details})

    if order_details.get("error"):
        return {
            "domain_result": f"Could not process cancellation: {order_details['error']}",
            "status": "fulfilled",
            "tool_calls": tool_calls,
        }

    # Step 2: Hard constraint check
    cancel_args = {"order_id": order_id}
    violation = _hard_constraint_check("cancel_order", cancel_args, constraints)
    if violation:
        tool_calls.append({
            "tool": "cancel_order", "args": cancel_args,
            "blocked": True, "reason": violation,
        })
        return {
            "domain_result": f"Cancellation cannot be processed: {violation}",
            "status": "blocked",
            "tool_calls": tool_calls,
        }

    # Step 3: Approval check
    approval_reasons = _needs_approval("cancel_order", cancel_args, constraints)
    if approval_reasons:
        pending = {"tool": "cancel_order", "args": cancel_args, "reasons": approval_reasons}
        tool_calls.append({
            "tool": "cancel_order", "args": cancel_args,
            "blocked": True, "reason": "needs_approval",
        })
        return {
            "domain_result": (
                f"Action 'cancel_order' with args {cancel_args} "
                f"requires human approval ({', '.join(approval_reasons)})."
            ),
            "status": "awaiting_human",
            "pending_tool_call": pending,
            "reason_code_override": approval_reasons[0],
            "tool_calls": tool_calls,
        }

    # Step 4: Execute cancellation
    cancel_result = _invoke_write_tool(
        "cancel_order", tenant_id, customer_id, **cancel_args,
    )
    tool_calls.append({"tool": "cancel_order", "args": cancel_args, "result": cancel_result})

    if cancel_result.get("success"):
        return {
            "domain_result": cancel_result.get("message", json.dumps(cancel_result, default=str)),
            "status": "fulfilled",
            "tool_calls": tool_calls,
        }
    else:
        return {
            "domain_result": cancel_result.get("error", "Cancellation failed."),
            "status": "fulfilled",
            "tool_calls": tool_calls,
        }


_WRITE_WORKFLOWS: dict[str, callable] = {
    "refund": _execute_refund_workflow,
    "cancel_order": _execute_cancel_workflow,
}


# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------

def write_executor_node(state: AgentState) -> dict:
    """Deterministic write executor — dispatches to the appropriate workflow
    based on intent. Never uses an LLM ReAct loop."""
    result = dict(state["current_intent"])
    intent = result["intent"]

    if intent not in WRITE_INTENTS:
        logger.error("write_executor received non-write intent '%s' — failing closed", intent)
        result["domain_result"] = "This request cannot be processed."
        result["status"] = "blocked"
        return {"processed_results": [result]}

    workflow = _WRITE_WORKFLOWS.get(intent)
    if workflow is None:
        logger.error("No write workflow registered for intent '%s'", intent)
        result["domain_result"] = "This request cannot be processed."
        result["status"] = "blocked"
        return {"processed_results": [result]}

    tenant_id = state["tenant_id"]
    customer_id = state.get("customer_id") or ""
    entities = result.get("entities", {})
    constraints = result.get("constraints", {})

    workflow_result = workflow(tenant_id, customer_id, entities, constraints)

    result["tool_calls"] = workflow_result.get("tool_calls", [])
    result["domain_result"] = workflow_result["domain_result"]
    result["status"] = workflow_result["status"]

    if workflow_result.get("pending_tool_call"):
        result["pending_tool_call"] = workflow_result["pending_tool_call"]

    if workflow_result.get("reason_code_override"):
        result["reason_code"] = workflow_result["reason_code_override"]

    return {"processed_results": [result]}
