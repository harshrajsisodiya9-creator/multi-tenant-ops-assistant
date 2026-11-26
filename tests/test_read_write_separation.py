"""Tests for read/write execution separation (Step 1).

Covers: execution registry, write_executor deterministic workflows,
routing dispatch, ReAct read-only enforcement, HITL migration,
and graph-level behavior.
"""
from __future__ import annotations

import pytest

from app.graph.execution_registry import (
    INTENT_EXECUTION_TYPE,
    READ_TOOLS,
    TOOL_EXECUTION_TYPE,
    WRITE_INTENTS,
    WRITE_TOOLS,
    ExecutionType,
    get_execution_type,
    is_write_intent,
    is_write_tool,
)


# ===================================================================
# 1. Execution registry
# ===================================================================

class TestExecutionRegistry:
    def test_every_intent_has_classification(self):
        from app.graph.schemas import INTENT_TYPES
        import typing
        literal_args = typing.get_args(INTENT_TYPES)
        for intent in literal_args:
            assert intent in INTENT_EXECUTION_TYPE, f"Intent '{intent}' has no execution classification"

    def test_every_tool_has_classification(self):
        from app.graph.agent_prod.tools import TOOL_REGISTRY
        for tool_name in TOOL_REGISTRY:
            assert tool_name in TOOL_EXECUTION_TYPE, f"Tool '{tool_name}' has no execution classification"

    def test_write_intents_are_correct(self):
        assert WRITE_INTENTS == {"refund", "cancel_order"}

    def test_write_tools_are_correct(self):
        assert WRITE_TOOLS == {"initiate_refund", "cancel_order", "initiate_return", "get_shipping_label"}

    def test_read_tools_are_correct(self):
        assert READ_TOOLS == {"get_order_details", "get_return_eligibility", "get_order_tracking", "list_my_orders"}

    def test_is_write_intent(self):
        assert is_write_intent("refund") is True
        assert is_write_intent("cancel_order") is True
        assert is_write_intent("order_status") is False
        assert is_write_intent("faq") is False
        assert is_write_intent("unknown_intent") is False

    def test_is_write_tool(self):
        assert is_write_tool("initiate_refund") is True
        assert is_write_tool("cancel_order") is True
        assert is_write_tool("get_order_details") is False
        assert is_write_tool("unknown_tool") is False

    def test_get_execution_type_returns_none_for_unknown(self):
        assert get_execution_type("nonexistent") is None

    def test_unknown_operations_fail_closed(self):
        assert get_execution_type("unknown") is None
        assert is_write_intent("unknown") is False


# ===================================================================
# 2. ReAct agent is read-only
# ===================================================================

class TestReActReadOnly:
    def test_intent_tools_contain_no_write_tools(self):
        from app.graph.agent_prod.react_agent import INTENT_TOOLS
        for intent, tools in INTENT_TOOLS.items():
            for tool in tools:
                assert not is_write_tool(tool), (
                    f"INTENT_TOOLS['{intent}'] contains write tool '{tool}'"
                )

    def test_lc_tools_exclude_write_tools(self):
        from app.graph.agent_prod.react_subgraph import _LC_TOOLS
        for tool_name in _LC_TOOLS:
            assert not is_write_tool(tool_name), (
                f"LangChain tool registry contains write tool '{tool_name}'"
            )

    def test_tool_execute_blocks_write_tools(self):
        from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
        from app.graph.agent_prod.react_subgraph import tool_execute_node

        write_call = AIMessage(
            content="",
            tool_calls=[{
                "name": "initiate_refund",
                "args": {"order_id": "ORD-MM-1001", "amount": 4500, "refund_type": "original_method"},
                "id": "call_1",
            }],
        )

        state = {
            "messages": [SystemMessage(content="test"), write_call],
            "tool_call_log": [],
            "allowed_tools": ["get_order_details", "initiate_refund"],
            "order_id": "ORD-MM-1001",
            "constraints": {},
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
        }

        result = tool_execute_node(state)
        assert result.get("messages")
        msg = result["messages"][0]
        assert isinstance(msg, ToolMessage)
        assert "BLOCKED" in msg.content
        assert "write operation" in msg.content.lower() or "read-only" in msg.content.lower()

        log = result["tool_call_log"]
        assert len(log) == 1
        assert log[0]["blocked"] is True
        assert "write tool" in log[0]["reason"] or "read-only" in log[0]["reason"]

    def test_tool_execute_allows_read_tools(self):
        from unittest.mock import patch
        from langchain_core.messages import AIMessage, SystemMessage
        from app.graph.agent_prod.react_subgraph import tool_execute_node

        read_call = AIMessage(
            content="",
            tool_calls=[{
                "name": "get_order_details",
                "args": {"order_id": "ORD-MM-1001"},
                "id": "call_2",
            }],
        )

        state = {
            "messages": [SystemMessage(content="test"), read_call],
            "tool_call_log": [],
            "allowed_tools": ["get_order_details"],
            "order_id": "ORD-MM-1001",
            "constraints": {},
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
        }

        mock_result = {"order_id": "ORD-MM-1001", "order_status": "delivered"}
        with patch("app.graph.agent_prod.react_subgraph.TOOL_REGISTRY", {"get_order_details": lambda ctx, **kw: mock_result}):
            result = tool_execute_node(state)

        log = result.get("tool_call_log", [])
        assert len(log) == 1
        assert log[0].get("blocked") is not True
        assert "result" in log[0]


# ===================================================================
# 3. AGENT_INTENTS no longer contains write intents
# ===================================================================

class TestAgentIntentsEmpty:
    def test_agent_intents_has_no_write_intents(self):
        from app.graph.nodes.intent_executor import AGENT_INTENTS
        for intent in AGENT_INTENTS:
            assert not is_write_intent(intent), (
                f"AGENT_INTENTS contains write intent '{intent}'"
            )

    def test_write_intents_not_in_agent_intents(self):
        from app.graph.nodes.intent_executor import AGENT_INTENTS
        assert "refund" not in AGENT_INTENTS
        assert "cancel_order" not in AGENT_INTENTS


# ===================================================================
# 4. Write executor — refund workflow
# ===================================================================

class TestWriteExecutorRefund:
    def test_refund_fulfilled(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "refund ORD-MM-1005",
            "current_intent": {
                "intent": "refund",
                "entities": {"order_id": "ORD-MM-1005"},
                "policy_action": "allow",
                "reason_code": "REFUND_ALLOWED",
                "constraints": {"refund_type": "original_method", "condition": "any"},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        assert len(result["processed_results"]) == 1
        r = result["processed_results"][0]
        assert r["status"] == "fulfilled"
        assert r["tool_calls"]
        tool_names = [tc["tool"] for tc in r["tool_calls"]]
        assert "get_order_details" in tool_names
        assert "get_return_eligibility" in tool_names
        assert "initiate_refund" in tool_names

    def test_refund_order_not_found(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "refund ORD-FAKE",
            "current_intent": {
                "intent": "refund",
                "entities": {"order_id": "ORD-FAKE"},
                "policy_action": "allow",
                "reason_code": "REFUND_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        r = result["processed_results"][0]
        assert "not found" in r["domain_result"].lower()
        assert r["tool_calls"]

    def test_refund_not_eligible(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-003",
            "user_message": "refund ORD-MM-1003",
            "current_intent": {
                "intent": "refund",
                "entities": {"order_id": "ORD-MM-1003"},
                "policy_action": "allow",
                "reason_code": "REFUND_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        r = result["processed_results"][0]
        assert "not allowed" in r["domain_result"].lower() or "not eligible" in r["domain_result"].lower()

    def test_refund_needs_approval(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "refund ORD-MM-1005",
            "current_intent": {
                "intent": "refund",
                "entities": {"order_id": "ORD-MM-1005"},
                "policy_action": "allow",
                "reason_code": "REFUND_NEEDS_APPROVAL",
                "constraints": {
                    "refund_type": "original_method",
                    "needs_approval": True,
                    "approval_reasons": ["TENANT_REFUND_REVIEW_REQUIRED"],
                },
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        r = result["processed_results"][0]
        assert r["status"] == "awaiting_human"
        assert r.get("pending_tool_call")
        assert r["pending_tool_call"]["tool"] == "initiate_refund"
        assert "TENANT_REFUND_REVIEW_REQUIRED" in r["pending_tool_call"]["reasons"]

    def test_refund_uses_policy_refund_type(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "refund ORD-MM-1005",
            "current_intent": {
                "intent": "refund",
                "entities": {"order_id": "ORD-MM-1005"},
                "policy_action": "allow",
                "reason_code": "REFUND_ALLOWED",
                "constraints": {"refund_type": "store_credit", "condition": "any"},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        r = result["processed_results"][0]
        refund_calls = [tc for tc in r["tool_calls"] if tc["tool"] == "initiate_refund" and not tc.get("blocked")]
        assert refund_calls
        assert refund_calls[0]["args"]["refund_type"] == "store_credit"


# ===================================================================
# 5. Write executor — cancel workflow
# ===================================================================

class TestWriteExecutorCancel:
    def test_cancel_pending_order(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-006",
            "user_message": "cancel ORD-MM-1007",
            "current_intent": {
                "intent": "cancel_order",
                "entities": {"order_id": "ORD-MM-1007"},
                "policy_action": "allow",
                "reason_code": "CANCEL_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        r = result["processed_results"][0]
        assert r["status"] == "fulfilled"
        tool_names = [tc["tool"] for tc in r["tool_calls"]]
        assert "cancel_order" in tool_names

    def test_cancel_non_cancellable_order(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "cancel ORD-MM-1001",
            "current_intent": {
                "intent": "cancel_order",
                "entities": {"order_id": "ORD-MM-1001"},
                "policy_action": "allow",
                "reason_code": "CANCEL_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        r = result["processed_results"][0]
        assert "cannot be cancelled" in r["domain_result"].lower() or "cancel" in r["domain_result"].lower()

    def test_cancel_order_not_found(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "cancel ORD-FAKE",
            "current_intent": {
                "intent": "cancel_order",
                "entities": {"order_id": "ORD-FAKE"},
                "policy_action": "allow",
                "reason_code": "CANCEL_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        r = result["processed_results"][0]
        assert "not found" in r["domain_result"].lower()

    def test_cancel_constraint_blocked(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-006",
            "user_message": "cancel ORD-MM-1007",
            "current_intent": {
                "intent": "cancel_order",
                "entities": {"order_id": "ORD-MM-1007"},
                "policy_action": "allow",
                "reason_code": "CANCEL_ALLOWED",
                "constraints": {"cancel_blocked": True},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        r = result["processed_results"][0]
        assert r["status"] == "blocked"
        assert "not allowed" in r["domain_result"].lower()


# ===================================================================
# 6. Write executor — fail closed for unknown intents
# ===================================================================

class TestWriteExecutorFailClosed:
    def test_non_write_intent_fails_closed(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "where is my order?",
            "current_intent": {
                "intent": "order_status",
                "entities": {"order_id": "ORD-MM-1001"},
                "policy_action": "allow",
                "reason_code": "ORDER_STATUS_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        r = result["processed_results"][0]
        assert r["status"] == "blocked"

    def test_unknown_intent_fails_closed(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "do something new",
            "current_intent": {
                "intent": "new_unknown_intent",
                "entities": {},
                "policy_action": "allow",
                "reason_code": "UNKNOWN",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        r = result["processed_results"][0]
        assert r["status"] == "blocked"


# ===================================================================
# 7. Routing dispatch
# ===================================================================

class TestRoutingDispatch:
    def _make_state(self, intent_results):
        return {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "test",
            "conversation_id": "",
            "customer_context": {},
            "orders_context": [],
            "intent_results": intent_results,
        }

    def test_refund_routes_to_write_executor(self):
        from app.graph.builder import route_intents

        state = self._make_state([{
            "intent": "refund",
            "entities": {"order_id": "ORD-MM-1001"},
            "policy_action": "allow",
            "reason_code": "REFUND_ALLOWED",
            "constraints": {},
            "status": "pending",
        }])

        sends = route_intents(state)
        assert len(sends) == 1
        assert sends[0].node == "write_executor"

    def test_cancel_routes_to_write_executor(self):
        from app.graph.builder import route_intents

        state = self._make_state([{
            "intent": "cancel_order",
            "entities": {"order_id": "ORD-MM-1007"},
            "policy_action": "allow",
            "reason_code": "CANCEL_ALLOWED",
            "constraints": {},
            "status": "pending",
        }])

        sends = route_intents(state)
        assert len(sends) == 1
        assert sends[0].node == "write_executor"

    def test_order_status_routes_to_quick_resolver(self):
        from app.graph.builder import route_intents

        state = self._make_state([{
            "intent": "order_status",
            "entities": {"order_id": "ORD-MM-1001"},
            "policy_action": "allow",
            "reason_code": "ORDER_STATUS_ALLOWED",
            "constraints": {},
            "status": "pending",
        }])

        sends = route_intents(state)
        assert len(sends) == 1
        assert sends[0].node == "quick_resolver"

    def test_faq_routes_to_knowledge_lookup(self):
        from app.graph.builder import route_intents

        state = self._make_state([{
            "intent": "faq",
            "entities": {},
            "policy_action": "allow",
            "reason_code": "FAQ_ALLOWED",
            "constraints": {},
            "status": "pending",
        }])

        sends = route_intents(state)
        assert len(sends) == 1
        assert sends[0].node == "knowledge_lookup"

    def test_blocked_intent_routes_to_quick_resolver(self):
        from app.graph.builder import route_intents

        state = self._make_state([{
            "intent": "refund",
            "entities": {"order_id": "ORD-MM-1003"},
            "policy_action": "block",
            "reason_code": "RETURNS_NOT_ALLOWED",
            "constraints": {},
            "status": "pending",
        }])

        sends = route_intents(state)
        assert len(sends) == 1
        assert sends[0].node == "quick_resolver"

    def test_escalated_intent_routes_to_quick_resolver(self):
        from app.graph.builder import route_intents

        state = self._make_state([{
            "intent": "refund",
            "entities": {"order_id": "ORD-MM-1001"},
            "policy_action": "escalate",
            "reason_code": "REFUND_OVER_LIMIT",
            "constraints": {},
            "status": "pending",
        }])

        sends = route_intents(state)
        assert len(sends) == 1
        assert sends[0].node == "quick_resolver"

    def test_clarify_routes_to_quick_resolver(self):
        from app.graph.builder import route_intents

        state = self._make_state([{
            "intent": "refund",
            "entities": {},
            "policy_action": "clarify",
            "reason_code": "VERIFICATION_REQUIRED",
            "constraints": {},
            "status": "pending",
        }])

        sends = route_intents(state)
        assert len(sends) == 1
        assert sends[0].node == "quick_resolver"

    def test_multi_intent_mixed_routing(self):
        from app.graph.builder import route_intents

        state = self._make_state([
            {
                "intent": "order_status",
                "entities": {"order_id": "ORD-MM-1001"},
                "policy_action": "allow",
                "reason_code": "ORDER_STATUS_ALLOWED",
                "constraints": {},
                "status": "pending",
            },
            {
                "intent": "refund",
                "entities": {"order_id": "ORD-MM-1001"},
                "policy_action": "allow",
                "reason_code": "REFUND_ALLOWED",
                "constraints": {},
                "status": "pending",
            },
        ])

        sends = route_intents(state)
        assert len(sends) == 2
        nodes = {s.node for s in sends}
        assert "quick_resolver" in nodes
        assert "write_executor" in nodes

    def test_write_intent_never_reaches_lookup_agent(self):
        from app.graph.builder import route_intents

        state = self._make_state([{
            "intent": "refund",
            "entities": {"order_id": "ORD-MM-1001"},
            "policy_action": "allow",
            "reason_code": "REFUND_ALLOWED",
            "constraints": {},
            "status": "pending",
        }])

        sends = route_intents(state)
        for s in sends:
            assert s.node != "lookup_agent", "Write intent was routed to lookup_agent"

    def test_non_pending_skipped(self):
        from app.graph.builder import route_intents

        state = self._make_state([{
            "intent": "refund",
            "entities": {"order_id": "ORD-MM-1001"},
            "policy_action": "allow",
            "reason_code": "REFUND_ALLOWED",
            "constraints": {},
            "status": "fulfilled",
        }])

        sends = route_intents(state)
        assert len(sends) == 0


# ===================================================================
# 8. Write executor results compatible with merge_results
# ===================================================================

class TestMergeResultsCompat:
    def test_write_result_aggregates_via_merge(self):
        from app.graph.nodes.intent_executor import merge_results_node

        state = {
            "processed_results": [
                {
                    "intent": "refund",
                    "status": "fulfilled",
                    "domain_result": "Refund of ₹4,500 initiated.",
                    "tool_calls": [
                        {"tool": "get_order_details", "args": {"order_id": "ORD-MM-1001"}, "result": {}},
                        {"tool": "initiate_refund", "args": {"order_id": "ORD-MM-1001"}, "result": {}},
                    ],
                },
            ],
        }

        result = merge_results_node(state)
        assert result["intent_results"][0]["status"] == "fulfilled"
        assert result["has_tool_calls"] is True
        assert result["response_phase"] == "complete"

    def test_write_awaiting_human_triggers_partial_phase(self):
        from app.graph.nodes.intent_executor import merge_results_node

        state = {
            "processed_results": [
                {
                    "intent": "refund",
                    "status": "awaiting_human",
                    "domain_result": "Requires approval.",
                    "pending_tool_call": {"tool": "initiate_refund", "args": {}, "reasons": ["CEILING"]},
                    "tool_calls": [],
                },
            ],
        }

        result = merge_results_node(state)
        assert result["response_phase"] == "partial"

    def test_mixed_read_and_write_results(self):
        from app.graph.nodes.intent_executor import merge_results_node

        state = {
            "processed_results": [
                {
                    "intent": "order_status",
                    "status": "fulfilled",
                    "domain_result": "Delivered.",
                },
                {
                    "intent": "refund",
                    "status": "fulfilled",
                    "domain_result": "Refund initiated.",
                    "tool_calls": [{"tool": "initiate_refund", "args": {}, "result": {}}],
                },
            ],
        }

        result = merge_results_node(state)
        assert len(result["intent_results"]) == 2
        assert result["has_tool_calls"] is True
        assert result["response_phase"] == "complete"


# ===================================================================
# 9. Existing quick_resolver unchanged
# ===================================================================

class TestQuickResolverUnchanged:
    def test_order_status_still_works(self):
        from app.graph.nodes.intent_executor import quick_resolver_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "Where is ORD-MM-1001?",
            "current_intent": {
                "intent": "order_status",
                "entities": {"order_id": "ORD-MM-1001"},
                "policy_action": "allow",
                "reason_code": "ORDER_STATUS_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = quick_resolver_node(state)
        r = result["processed_results"][0]
        assert r["status"] == "fulfilled"
        assert "ORD-MM-1001" in r["domain_result"]

    def test_block_still_works(self):
        from app.graph.nodes.intent_executor import quick_resolver_node

        state = {
            "tenant_id": "megamart",
            "user_message": "return perishables",
            "current_intent": {
                "intent": "refund",
                "entities": {"order_id": "ORD-MM-1003"},
                "policy_action": "block",
                "reason_code": "RETURNS_NOT_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = quick_resolver_node(state)
        assert result["processed_results"][0]["status"] == "blocked"

    def test_escalate_still_works(self):
        from app.graph.nodes.intent_executor import quick_resolver_node

        state = {
            "tenant_id": "megamart",
            "user_message": "refund",
            "current_intent": {
                "intent": "refund",
                "entities": {},
                "policy_action": "escalate",
                "reason_code": "ESCALATION_KEYWORDS_DETECTED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = quick_resolver_node(state)
        assert result["processed_results"][0]["status"] == "awaiting_human"

    def test_greeting_still_works(self):
        from app.graph.nodes.intent_executor import quick_resolver_node

        state = {
            "tenant_id": "megamart",
            "user_message": "Hello",
            "current_intent": {
                "intent": "greeting",
                "entities": {},
                "policy_action": "allow",
                "reason_code": "GREETING_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = quick_resolver_node(state)
        r = result["processed_results"][0]
        assert r["status"] == "fulfilled"
        assert "hello" in r["domain_result"].lower() or "help" in r["domain_result"].lower()


# ===================================================================
# 10. Graph build succeeds
# ===================================================================

class TestGraphBuild:
    def test_graph_compiles_with_write_executor(self):
        from app.graph.builder import build_graph
        graph = build_graph()
        assert graph is not None
