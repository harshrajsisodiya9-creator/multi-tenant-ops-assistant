from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.graph.nodes.action_check import action_check_node
from app.graph.nodes.before_router import before_router_node
from app.graph.nodes.escalation_handler import escalation_handler_node
from app.graph.nodes.intent_executor import (
    AGENT_INTENTS,
    agent_executor_node,
    bail_executor_node,
    faq_executor_node,
    merge_results_node,
    simple_executor_node,
)
from app.graph.nodes.policy_fan_out import policy_fan_out_node
from app.graph.nodes.pre_check import pre_check_node
from app.graph.nodes.response_agent import response_agent_node
from app.graph.nodes.router import router_node
from app.graph.state import AgentState


# ---------------------------------------------------------------------------
# Routing functions — Literal return types show in LangSmith traces
# ---------------------------------------------------------------------------

def after_pre_check(state: AgentState) -> Literal["bail", "continue"]:
    if state.get("pre_check_bail"):
        return "bail"
    return "continue"


def route_intents(state: AgentState) -> list[Send]:
    """Send() fan-out: one Send per pending intent, dispatched to the right executor."""
    sends: list[Send] = []
    base = {
        "tenant_id": state["tenant_id"],
        "customer_id": state.get("customer_id"),
        "user_message": state["user_message"],
        "conversation_id": state.get("conversation_id", ""),
        "conversation_history": state.get("conversation_history", []),
        "customer_context": state.get("customer_context", {}),
        "orders_context": state.get("orders_context", []),
    }

    for result in state.get("intent_results") or []:
        if result.get("status") != "pending":
            continue

        payload = {**base, "current_intent": result}
        action = result["policy_action"]
        intent = result["intent"]

        if action == "allow" and intent in AGENT_INTENTS:
            sends.append(Send("agent_executor", payload))
        elif action == "allow" and intent in ("faq", "complaint"):
            sends.append(Send("faq_executor", payload))
        else:
            sends.append(Send("simple_executor", payload))

    return sends


def after_merge(state: AgentState) -> Literal["needs_action_check", "skip_action_check"]:
    if state.get("has_tool_calls"):
        return "needs_action_check"
    return "skip_action_check"


def after_response_agent(state: AgentState) -> Literal["complete", "needs_escalation"]:
    if state.get("response_phase") == "partial":
        return "needs_escalation"
    return "complete"


def after_escalation(state: AgentState) -> Literal["follow_up", "end"]:
    if state.get("response_phase") == "follow_up":
        return "follow_up"
    return "end"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # --- nodes ---
    graph.add_node("before_router", before_router_node)
    graph.add_node("router", router_node)
    graph.add_node("pre_check", pre_check_node)
    graph.add_node("bail_executor", bail_executor_node)
    graph.add_node("policy_fan_out", policy_fan_out_node)
    graph.add_node("agent_executor", agent_executor_node)
    graph.add_node("faq_executor", faq_executor_node)
    graph.add_node("simple_executor", simple_executor_node)
    graph.add_node("merge_results", merge_results_node)
    graph.add_node("action_check", action_check_node)
    graph.add_node("response_agent", response_agent_node)
    graph.add_node("escalation_handler", escalation_handler_node)
    graph.add_node("follow_up_response", response_agent_node)

    # --- edges ---
    graph.add_edge(START, "before_router")
    graph.add_edge("before_router", "router")
    graph.add_edge("router", "pre_check")

    # 1) after pre_check: bail or continue to policy
    graph.add_conditional_edges("pre_check", after_pre_check, {
        "bail": "bail_executor",
        "continue": "policy_fan_out",
    })

    # bail path goes straight to response_agent
    graph.add_edge("bail_executor", "response_agent")

    # 2) after policy_fan_out: Send() fan-out per intent
    graph.add_conditional_edges("policy_fan_out", route_intents)

    # all executor branches converge at merge_results
    graph.add_edge("agent_executor", "merge_results")
    graph.add_edge("faq_executor", "merge_results")
    graph.add_edge("simple_executor", "merge_results")

    # 3) after merge: action_check needed?
    graph.add_conditional_edges("merge_results", after_merge, {
        "needs_action_check": "action_check",
        "skip_action_check": "response_agent",
    })
    graph.add_edge("action_check", "response_agent")

    # 4) after response_agent: done or escalation?
    graph.add_conditional_edges("response_agent", after_response_agent, {
        "complete": END,
        "needs_escalation": "escalation_handler",
    })

    # 5) after escalation: follow-up response or end?
    graph.add_conditional_edges("escalation_handler", after_escalation, {
        "follow_up": "follow_up_response",
        "end": END,
    })
    graph.add_edge("follow_up_response", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
