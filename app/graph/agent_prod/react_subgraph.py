from __future__ import annotations

import inspect
import json
from typing import Annotated, Literal, TypedDict, get_type_hints

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import Field, create_model

from app.graph.agent_prod.tool_context import ToolContext
from app.graph.agent_prod.tools import TOOL_DESCRIPTIONS, TOOL_REGISTRY
from app.graph.policy_gate import GLOBAL_REFUND_CEILING

MAX_ITERATIONS = 8

WRITE_TOOLS = frozenset({
    "initiate_refund", "cancel_order", "initiate_return", "get_shipping_label",
})

_IDENTITY_PARAMS = frozenset({"tenant_id", "customer_id", "customer_email", "ctx"})


def _build_lc_tools() -> dict[str, StructuredTool]:
    desc_map = {d["name"]: d["description"] for d in TOOL_DESCRIPTIONS}
    tools: dict[str, StructuredTool] = {}

    for name, fn in TOOL_REGISTRY.items():
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
        fields: dict = {}

        for pname, param in sig.parameters.items():
            if pname in _IDENTITY_PARAMS:
                continue
            ann = hints.get(pname, str)
            if ann is inspect.Parameter.empty:
                ann = str
            if param.default is inspect.Parameter.empty:
                fields[pname] = (ann, ...)
            else:
                fields[pname] = (ann, Field(default=param.default))

        input_model = create_model(f"{name}_input", **fields)
        tools[name] = StructuredTool(
            name=name,
            description=desc_map.get(name, fn.__doc__ or name),
            args_schema=input_model,
            func=lambda **kw: None,
        )

    return tools


_LC_TOOLS = _build_lc_tools()


class ReactAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_log: list[dict]
    iteration: int
    final_result: str
    allowed_tools: list[str]
    order_id: str
    constraints: dict
    tenant_id: str
    customer_id: str
    pending_approval: dict
    conversation_history: list[dict]
    customer_context: dict
    orders_context: list[dict]


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


def _build_react_context(state: ReactAgentState) -> str | None:
    parts: list[str] = []
    history = state.get("conversation_history") or []
    if history:
        lines = [
            f"{'Customer' if m['role'] == 'human' else 'Assistant'}: {m['content']}"
            for m in history
        ]
        parts.append("Previous conversation:\n" + "\n".join(lines))

    orders = state.get("orders_context") or []
    if orders:
        order_strs = [
            f"  {o.get('order_id', '?')} — {o.get('product', '?')} ({o.get('status', '?')})"
            for o in orders
        ]
        parts.append("Known orders:\n" + "\n".join(order_strs))

    return "\n\n".join(parts) if parts else None


def reason_node(state: ReactAgentState, config: RunnableConfig) -> dict:
    allowed = state["allowed_tools"]
    lc_tools = [_LC_TOOLS[n] for n in allowed if n in _LC_TOOLS]

    llm = (config.get("configurable") or {}).get("llm")
    if llm is None:
        from langchain_groq import ChatGroq

        from app.config import setting

        llm = ChatGroq(
            model=setting.groq_model,
            api_key=setting.groq_api_key,
            temperature=0,
        )

    messages = list(state["messages"])
    if state.get("iteration", 0) == 0:
        ctx_text = _build_react_context(state)
        if ctx_text:
            from langchain_core.messages import SystemMessage as _Sys
            messages.insert(1, _Sys(content=f"Context from prior turns:\n{ctx_text}"))

    response: AIMessage = llm.bind_tools(lc_tools).invoke(messages)

    iteration = state.get("iteration", 0) + 1
    update: dict = {"messages": [response], "iteration": iteration}

    if not response.tool_calls:
        update["final_result"] = response.content

    return update


def tool_execute_node(state: ReactAgentState) -> dict:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}

    allowed = set(state["allowed_tools"])
    order_id = state["order_id"]
    constraints = state.get("constraints") or {}
    ctx = ToolContext(tenant_id=state["tenant_id"], customer_id=state["customer_id"])

    messages: list[ToolMessage] = []
    log = list(state.get("tool_call_log") or [])

    for tc in last.tool_calls:
        name, args, call_id = tc["name"], tc["args"], tc["id"]

        if name not in allowed:
            messages.append(ToolMessage(
                content=f"BLOCKED: Tool '{name}' is not allowed for this task.",
                tool_call_id=call_id,
            ))
            log.append({"tool": name, "args": args, "blocked": True, "reason": "not in allowlist"})
            continue

        call_oid = args.get("order_id")
        if call_oid and order_id and call_oid != order_id:
            messages.append(ToolMessage(
                content=f"BLOCKED: This task is limited to order {order_id}.",
                tool_call_id=call_id,
            ))
            log.append({"tool": name, "args": args, "blocked": True, "reason": f"order lock: expected {order_id}, got {call_oid}"})
            continue

        if name in WRITE_TOOLS:
            violation = _hard_constraint_check(name, args, constraints)
            if violation:
                messages.append(ToolMessage(
                    content=f"BLOCKED: {violation}",
                    tool_call_id=call_id,
                ))
                log.append({"tool": name, "args": args, "blocked": True, "reason": violation})
                continue

            approval_reasons = _needs_approval(name, args, constraints)
            if approval_reasons:
                pending = {
                    "tool": name,
                    "args": args,
                    "reasons": approval_reasons,
                }
                messages.append(ToolMessage(
                    content="PENDING APPROVAL: This action requires human review before execution.",
                    tool_call_id=call_id,
                ))
                log.append({"tool": name, "args": args, "blocked": True, "reason": "needs_approval"})
                return {
                    "messages": messages,
                    "tool_call_log": log,
                    "pending_approval": pending,
                }

        result = TOOL_REGISTRY[name](ctx, **args)
        log.append({"tool": name, "args": args, "result": result})
        messages.append(ToolMessage(
            content=json.dumps(result, default=str),
            tool_call_id=call_id,
        ))

    return {"messages": messages, "tool_call_log": log}


def should_continue(state: ReactAgentState) -> Literal["done", "max_iterations", "continue_tools"]:
    if state.get("pending_approval"):
        return "done"
    if state.get("final_result"):
        return "done"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "max_iterations"
    return "continue_tools"


def build_react_subgraph():
    graph = StateGraph(ReactAgentState)

    graph.add_node("reason", reason_node)
    graph.add_node("tool_execute", tool_execute_node)

    graph.add_edge(START, "reason")
    graph.add_conditional_edges("reason", should_continue, {
        "done": "__end__",
        "max_iterations": "__end__",
        "continue_tools": "tool_execute",
    })
    graph.add_edge("tool_execute", "reason")

    return graph.compile()
