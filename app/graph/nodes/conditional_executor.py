"""Conditional executor — Read → Evaluate → Write (or answer).

Handles requests like "If my order hasn't shipped, cancel it."
1. Gathers evidence by invoking read tools
2. Evaluates the structured condition deterministically
3. For conditional_action + satisfied: delegates to write workflow
4. For conditional_action + not_satisfied: returns explanation only
5. For conditional_question: returns evaluation result, never writes
6. For unknown: returns clarification
"""
from __future__ import annotations

import logging

from app.graph.condition_evaluator import (
    ConditionResult,
    evaluate_condition,
    required_tools_for_condition,
)
from app.graph.execution_registry import WRITE_INTENTS
from app.graph.nodes.write_executor import (
    _WRITE_WORKFLOWS,
    _invoke_read_tool,
)
from app.graph.state import AgentState

logger = logging.getLogger(__name__)


def _gather_evidence(
    condition: dict,
    tenant_id: str,
    customer_id: str,
    entities: dict,
) -> tuple[dict[str, dict], list[dict]]:
    """Invoke the read tools needed for the condition and return evidence + tool log."""
    evidence: dict[str, dict] = {}
    tool_calls: list[dict] = []
    order_id = entities.get("order_id", "")

    for tool_name in required_tools_for_condition(condition):
        result = _invoke_read_tool(tool_name, tenant_id, customer_id, order_id=order_id)
        evidence[tool_name] = result
        tool_calls.append({"tool": tool_name, "args": {"order_id": order_id}, "result": result})

    return evidence, tool_calls


def conditional_executor_node(state: AgentState) -> dict:
    """Read → Evaluate → conditionally Write (or just answer)."""
    result = dict(state["current_intent"])
    intent = result["intent"]
    conditional = result.get("conditional") or {}
    request_type = conditional.get("request_type", "conditional_action")
    condition = conditional.get("condition") or {}

    if not condition:
        logger.error("conditional_executor received intent '%s' with no condition — failing closed", intent)
        result["domain_result"] = "Could not evaluate the condition for this request."
        result["status"] = "fulfilled"
        return {"processed_results": [result]}

    tenant_id = state["tenant_id"]
    customer_id = state.get("customer_id") or ""
    entities = result.get("entities", {})
    constraints = result.get("constraints", {})

    # ── Step 1: Gather evidence ──────────────────────────────────────
    evidence, tool_calls = _gather_evidence(condition, tenant_id, customer_id, entities)
    result["tool_calls"] = tool_calls

    # ── Step 2: Evaluate condition ───────────────────────────────────
    eval_result = evaluate_condition(condition, evidence)
    status = eval_result["status"]

    result["conditional"] = {
        **conditional,
        "evaluation": eval_result,
    }

    # ── Step 3: Branch on request_type × evaluation status ───────────

    if status == ConditionResult.UNKNOWN:
        result["domain_result"] = (
            f"I couldn't determine the answer: {eval_result.get('reason', 'unknown')} "
            "Could you provide more details?"
        )
        result["status"] = "fulfilled"
        return {"processed_results": [result]}

    if request_type == "conditional_question":
        if status == ConditionResult.SATISFIED:
            result["domain_result"] = (
                f"Yes — the condition is met. "
                f"The current value of '{condition.get('field', '')}' is "
                f"'{eval_result.get('actual_value', 'N/A')}'."
            )
        else:
            reason = eval_result.get("reason", "The condition is not met.")
            result["domain_result"] = f"No — {reason}"
        result["status"] = "fulfilled"
        return {"processed_results": [result]}

    # conditional_action from here on
    if status == ConditionResult.NOT_SATISFIED:
        reason = eval_result.get("reason", "The condition was not met.")
        result["domain_result"] = (
            f"The condition was not met, so no action was taken. {reason}"
        )
        result["status"] = "fulfilled"
        return {"processed_results": [result]}

    # ── Step 4: Condition satisfied → delegate to write workflow ──────
    if intent not in WRITE_INTENTS:
        logger.error(
            "conditional_executor: condition satisfied but intent '%s' is not a write intent", intent,
        )
        result["domain_result"] = "This request cannot be processed."
        result["status"] = "blocked"
        return {"processed_results": [result]}

    workflow = _WRITE_WORKFLOWS.get(intent)
    if workflow is None:
        logger.error("No write workflow registered for intent '%s'", intent)
        result["domain_result"] = "This request cannot be processed."
        result["status"] = "blocked"
        return {"processed_results": [result]}

    workflow_result = workflow(tenant_id, customer_id, entities, constraints)

    result["tool_calls"] = tool_calls + workflow_result.get("tool_calls", [])
    result["domain_result"] = workflow_result["domain_result"]
    result["status"] = workflow_result["status"]

    if workflow_result.get("pending_tool_call"):
        result["pending_tool_call"] = workflow_result["pending_tool_call"]
    if workflow_result.get("reason_code_override"):
        result["reason_code"] = workflow_result["reason_code_override"]

    return {"processed_results": [result]}
