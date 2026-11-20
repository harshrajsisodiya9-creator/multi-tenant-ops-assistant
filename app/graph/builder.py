from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes.action_check import action_check_node
from app.graph.nodes.escalation_handler import escalation_handler_node
from app.graph.nodes.intent_executor import intent_executor_node
from app.graph.nodes.policy_fan_out import policy_fan_out_node
from app.graph.nodes.pre_check import pre_check_node
from app.graph.nodes.response_agent import response_agent_node
from app.graph.nodes.router import router_node
from app.graph.state import AgentState


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("pre_check", pre_check_node)
    graph.add_node("policy_fan_out", policy_fan_out_node)
    graph.add_node("intent_executor", intent_executor_node)
    graph.add_node("action_check", action_check_node)
    graph.add_node("response_agent", response_agent_node)
    graph.add_node("escalation_handler", escalation_handler_node)
    graph.add_node("follow_up_response", response_agent_node)

    graph.add_edge(START, "router")
    graph.add_edge("router", "pre_check")
    graph.add_edge("pre_check", "policy_fan_out")
    graph.add_edge("policy_fan_out", "intent_executor")
    graph.add_edge("intent_executor", "action_check")
    graph.add_edge("action_check", "response_agent")
    graph.add_edge("response_agent", "escalation_handler")
    graph.add_edge("escalation_handler", "follow_up_response")
    graph.add_edge("follow_up_response", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
