"""Tests for before_router_node.

Covers: snapshot retrieval, human message recording, returned state keys,
conversation isolation between tenants/customers.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.graph.nodes.before_router import before_router_node

from .conftest import CUSTOMER_ALICE, CUSTOMER_BOB, TENANT_MEGAMART, make_base_state


class TestBeforeRouterNode:
    """Core happy-path and structure tests."""

    @patch("app.graph.nodes.before_router.conversation_store")
    def test_returns_expected_keys(self, mock_store):
        mock_store.get_snapshot.return_value = {
            "messages": [{"role": "human", "content": "hi"}],
            "customer_context": {"name": "Alice"},
            "orders_context": [{"order_id": "ORD-MM-1001"}],
            "turn_summaries": [],
        }
        state = make_base_state(user_message="Where is my order?")
        result = before_router_node(state)

        assert "conversation_history" in result
        assert "customer_context" in result
        assert "orders_context" in result
        assert "turn_summaries" in result

    @patch("app.graph.nodes.before_router.conversation_store")
    def test_snapshot_called_with_correct_args(self, mock_store):
        mock_store.get_snapshot.return_value = {
            "messages": [], "customer_context": {}, "orders_context": [], "turn_summaries": [],
        }
        state = make_base_state(
            tenant_id="megamart",
            customer_id="CUST-MM-001",
            conversation_id="conv-123",
            user_message="hi",
        )
        before_router_node(state)
        mock_store.get_snapshot.assert_called_once_with("megamart", "CUST-MM-001", "conv-123")

    @patch("app.graph.nodes.before_router.conversation_store")
    def test_human_message_recorded(self, mock_store):
        mock_store.get_snapshot.return_value = {
            "messages": [], "customer_context": {}, "orders_context": [], "turn_summaries": [],
        }
        state = make_base_state(user_message="I want a refund")
        before_router_node(state)
        mock_store.add_human_message.assert_called_once_with(
            TENANT_MEGAMART, CUSTOMER_ALICE, "conv-test-001", "I want a refund",
        )

    @patch("app.graph.nodes.before_router.conversation_store")
    def test_empty_snapshot_returns_empty_lists(self, mock_store):
        mock_store.get_snapshot.return_value = {
            "messages": [], "customer_context": {}, "orders_context": [], "turn_summaries": [],
        }
        state = make_base_state()
        result = before_router_node(state)
        assert result["conversation_history"] == []
        assert result["customer_context"] == {}
        assert result["orders_context"] == []
        assert result["turn_summaries"] == []

    @patch("app.graph.nodes.before_router.conversation_store")
    def test_conversation_history_passthrough(self, mock_store):
        messages = [
            {"role": "human", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        mock_store.get_snapshot.return_value = {
            "messages": messages,
            "customer_context": {},
            "orders_context": [],
            "turn_summaries": [],
        }
        state = make_base_state()
        result = before_router_node(state)
        assert result["conversation_history"] == messages


class TestBeforeRouterEdgeCases:
    """Missing fields and None customer_id."""

    @patch("app.graph.nodes.before_router.conversation_store")
    def test_none_customer_id_coerces_to_empty_string(self, mock_store):
        mock_store.get_snapshot.return_value = {
            "messages": [], "customer_context": {}, "orders_context": [], "turn_summaries": [],
        }
        state = make_base_state(customer_id=None)
        before_router_node(state)
        mock_store.get_snapshot.assert_called_once_with(TENANT_MEGAMART, "", "conv-test-001")

    @patch("app.graph.nodes.before_router.conversation_store")
    def test_missing_conversation_id_uses_empty_string(self, mock_store):
        mock_store.get_snapshot.return_value = {
            "messages": [], "customer_context": {}, "orders_context": [], "turn_summaries": [],
        }
        state = make_base_state()
        state.pop("conversation_id", None)
        before_router_node(state)
        mock_store.get_snapshot.assert_called_once_with(TENANT_MEGAMART, CUSTOMER_ALICE, "")

    @patch("app.graph.nodes.before_router.conversation_store")
    def test_turn_summaries_missing_from_snapshot(self, mock_store):
        """If the snapshot predates turn_summaries, the node defaults to []."""
        mock_store.get_snapshot.return_value = {
            "messages": [], "customer_context": {}, "orders_context": [],
        }
        state = make_base_state()
        result = before_router_node(state)
        assert result["turn_summaries"] == []


class TestBeforeRouterTurnReset:
    """Each new turn must start with no execution state left over from the previous turn."""

    @staticmethod
    def _stale_previous_turn_state() -> dict:
        state = make_base_state(user_message="Track my order ORD-MM-1002")
        state.update({
            "processed_results": [{"intent": "refund", "status": "awaiting_human"}],
            "intent_results": [{"intent": "refund", "status": "awaiting_human"}],
            "router_output": {"intent": "refund"},
            "policy_decision": {"action": "allow", "reason_code": "REFUND_ALLOWED", "constraints": {}},
            "has_tool_calls": True,
            "response_phase": "partial",
            "pre_check_bail": True,
            "response": "Old answer from the previous turn.",
        })
        return state

    @patch("app.graph.nodes.before_router.conversation_store")
    def test_execution_fields_reset(self, mock_store):
        mock_store.get_snapshot.return_value = {
            "messages": [], "customer_context": {}, "orders_context": [], "turn_summaries": [],
        }
        result = before_router_node(self._stale_previous_turn_state())

        assert result["processed_results"] is None  # None tells the reducer to clear
        assert result["intent_results"] == []
        assert result["router_output"] is None
        assert result["policy_decision"] is None
        assert result["has_tool_calls"] is False
        assert result["response_phase"] == "complete"
        assert result["pre_check_bail"] is False
        assert result["response"] == ""

    @patch("app.graph.nodes.before_router.conversation_store")
    def test_returns_exactly_memory_and_reset_fields(self, mock_store):
        """Request inputs (tenant_id, customer_id, user_message, ...) are never overwritten."""
        mock_store.get_snapshot.return_value = {
            "messages": [], "customer_context": {}, "orders_context": [], "turn_summaries": [],
        }
        result = before_router_node(self._stale_previous_turn_state())
        assert set(result) == {
            "conversation_history", "customer_context", "orders_context", "turn_summaries",
            "processed_results", "intent_results", "router_output", "policy_decision",
            "has_tool_calls", "response_phase", "pre_check_bail", "response",
        }

    @patch("app.graph.nodes.before_router.conversation_store")
    def test_conversation_memory_still_loaded_alongside_reset(self, mock_store):
        history = [{"role": "human", "content": "Refund ORD-MM-1001"}, {"role": "assistant", "content": "Done."}]
        summaries = [{"intent": "refund", "order_id": "ORD-MM-1001", "status": "fulfilled", "reason_code": ""}]
        mock_store.get_snapshot.return_value = {
            "messages": history,
            "customer_context": {"name": "Alice"},
            "orders_context": [{"order_id": "ORD-MM-1001"}],
            "turn_summaries": summaries,
        }
        result = before_router_node(self._stale_previous_turn_state())
        assert result["conversation_history"] == history
        assert result["customer_context"] == {"name": "Alice"}
        assert result["orders_context"] == [{"order_id": "ORD-MM-1001"}]
        assert result["turn_summaries"] == summaries
