"""Tests documenting the unreachable status of lookup_agent and ReAct subgraph.

AGENT_INTENTS is currently an empty set, so no intent is routed to
lookup_agent_node. These tests verify that status and document the
expected behavior if AGENT_INTENTS were populated.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.graph.builder import route_intents
from app.graph.nodes.intent_executor import AGENT_INTENTS, lookup_agent_node

from .conftest import CUSTOMER_ALICE, TENANT_MEGAMART, make_base_state, make_intent_result


class TestAgentIntentsEmpty:
    """Document that AGENT_INTENTS is empty and lookup_agent is unreachable."""

    def test_agent_intents_is_empty_set(self):
        assert isinstance(AGENT_INTENTS, set)
        assert len(AGENT_INTENTS) == 0

    def test_no_intent_routes_to_lookup_agent(self):
        """Exhaustively verify no known intent routes to lookup_agent."""
        from app.graph.schemas import INTENT_TYPES
        import typing

        all_intents = list(typing.get_args(INTENT_TYPES))

        for intent in all_intents:
            intents = [make_intent_result(
                intent=intent,
                policy_action="allow",
                reason_code="TEST",
                entities={"order_id": "ORD-1"},
            )]
            state = make_base_state()
            state["intent_results"] = intents
            sends = route_intents(state)
            for send in sends:
                assert send.node != "lookup_agent", (
                    f"Intent '{intent}' routed to lookup_agent, "
                    f"but AGENT_INTENTS is empty"
                )


class TestLookupAgentNodeDirect:
    """Test lookup_agent_node directly — it works, just nothing routes to it."""

    @patch("app.graph.nodes.intent_executor._run_react_agent")
    @patch("app.graph.nodes.intent_executor.conversation_store")
    def test_happy_path_if_invoked_directly(self, mock_store, mock_agent):
        mock_agent.return_value = {
            "result": "Order ORD-1 is delivered.",
            "tool_calls": [
                {"tool": "get_order_details", "args": {"order_id": "ORD-1"},
                 "result": {"order_id": "ORD-1", "product": "Widget", "status": "delivered", "amount": 1000}},
            ],
        }

        ir = make_intent_result(
            intent="order_status",
            policy_action="allow",
            reason_code="ORDER_STATUS_ALLOWED",
            entities={"order_id": "ORD-1"},
        )
        state = make_base_state()
        state["current_intent"] = ir

        result = lookup_agent_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert pr["domain_result"] == "Order ORD-1 is delivered."

    @patch("app.graph.nodes.intent_executor._run_react_agent")
    @patch("app.graph.nodes.intent_executor.conversation_store")
    def test_pending_approval_from_agent(self, mock_store, mock_agent):
        mock_agent.return_value = {
            "result": "",
            "tool_calls": [],
            "pending_approval": {
                "tool": "initiate_refund",
                "args": {"order_id": "ORD-1", "amount": 60000},
                "reasons": ["GLOBAL_REFUND_CEILING"],
            },
        }

        ir = make_intent_result(
            intent="refund",
            policy_action="allow",
            reason_code="REFUND_ALLOWED",
            entities={"order_id": "ORD-1"},
        )
        state = make_base_state()
        state["current_intent"] = ir

        result = lookup_agent_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "awaiting_human"
        assert "pending_tool_call" in pr

    @patch("app.graph.nodes.intent_executor._run_react_agent")
    @patch("app.graph.nodes.intent_executor.conversation_store")
    def test_orders_context_updated(self, mock_store, mock_agent):
        mock_agent.return_value = {
            "result": "Done.",
            "tool_calls": [
                {
                    "tool": "get_order_details",
                    "args": {"order_id": "ORD-1"},
                    "result": {
                        "order_id": "ORD-1",
                        "product": "Widget",
                        "status": "delivered",
                        "amount": 500,
                    },
                },
            ],
        }

        ir = make_intent_result(
            intent="order_status",
            policy_action="allow",
            reason_code="ORDER_STATUS_ALLOWED",
            entities={"order_id": "ORD-1"},
        )
        state = make_base_state(conversation_id="conv-1")
        state["current_intent"] = ir

        lookup_agent_node(state)
        mock_store.update_orders_context.assert_called_once()
        call_args = mock_store.update_orders_context.call_args
        orders = call_args[0][3]
        assert orders[0]["order_id"] == "ORD-1"
