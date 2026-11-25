from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.graph.agent_prod.react_subgraph import (  # noqa: F401
    WRITE_TOOLS,
    _hard_constraint_check,
    _needs_approval,
    build_react_subgraph,
)
from app.graph.agent_prod.tool_context import ToolContext

INTENT_TOOLS: dict[str, list[str]] = {
    "refund": [
        "get_order_details", "get_return_eligibility",
        "initiate_refund", "initiate_return", "get_shipping_label",
    ],
    "cancel_order": [
        "get_order_details", "get_order_tracking", "cancel_order",
    ],
}

REACT_SYSTEM_PROMPT = """\
You are a tool-calling agent for e-commerce customer support.

Rules:
- Always call get_order_details first to verify the order exists.
- For refunds: check return eligibility before initiating.
- For cancellations: the cancel_order tool checks status internally.
- Never skip eligibility checks.
- Respect the constraints given to you (refund_type, condition, etc.).
- Be concise in your final answer.
- When you have enough information, respond with a plain text final answer (no tool call).

Constraints for this request:
{constraints}"""

_subgraph = build_react_subgraph()


def run_react_agent(
    ctx: ToolContext,
    order_id: str,
    user_message: str,
    intent: str,
    constraints: dict | None = None,
    *,
    llm=None,
    conversation_history: list[dict] | None = None,
    customer_context: dict | None = None,
    orders_context: list[dict] | None = None,
) -> dict:
    constraints = constraints or {}

    allowed_tools = INTENT_TOOLS.get(intent)
    if allowed_tools is None:
        return {
            "result": f"No tool allowlist configured for intent '{intent}'.",
            "tool_calls": [],
            "iterations": 0,
        }

    system_prompt = REACT_SYSTEM_PROMPT.format(
        constraints=json.dumps(constraints) if constraints else "None",
    )

    config: RunnableConfig = {}
    if llm is not None:
        config = {"configurable": {"llm": llm}}

    result = _subgraph.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Order: {order_id}\nRequest: {user_message}"),
            ],
            "tool_call_log": [],
            "iteration": 0,
            "final_result": "",
            "allowed_tools": list(allowed_tools),
            "order_id": order_id,
            "constraints": constraints,
            "tenant_id": ctx.tenant_id,
            "customer_id": ctx.customer_id,
            "conversation_history": conversation_history or [],
            "customer_context": customer_context or {},
            "orders_context": orders_context or [],
        },
        config=config,
    )

    return {
        "result": result.get("final_result") or "Agent reached maximum iterations without a final answer.",
        "tool_calls": result.get("tool_call_log", []),
        "iterations": result.get("iteration", 0),
        "pending_approval": result.get("pending_approval"),
    }
