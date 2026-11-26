"""Tests for quick_resolver_node.

Covers: conversational responses, order_status lookup, block/escalate/clarify
messages, missing orders, all CONVERSATIONAL_INTENTS, clarify with constraints.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.graph.nodes.intent_executor import (
    CONVERSATIONAL_INTENTS,
    CONVERSATIONAL_RESPONSES,
    quick_resolver_node,
)

from .conftest import CUSTOMER_ALICE, TENANT_MEGAMART, make_base_state, make_intent_result


def _make_quick_resolver_state(intent_result: dict) -> dict:
    state = make_base_state()
    state["current_intent"] = intent_result
    return state


class TestQuickResolverConversational:
    def test_greeting_response(self):
        ir = make_intent_result(intent="greeting", policy_action="allow", reason_code="GREETING_ALLOWED")
        state = _make_quick_resolver_state(ir)
        result = quick_resolver_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert pr["domain_result"] == CONVERSATIONAL_RESPONSES["greeting"]

    def test_thanks_response(self):
        ir = make_intent_result(intent="thanks", policy_action="allow", reason_code="THANKS_ALLOWED")
        state = _make_quick_resolver_state(ir)
        result = quick_resolver_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert pr["domain_result"] == CONVERSATIONAL_RESPONSES["thanks"]

    def test_goodbye_response(self):
        ir = make_intent_result(intent="goodbye", policy_action="allow", reason_code="GOODBYE_ALLOWED")
        state = _make_quick_resolver_state(ir)
        result = quick_resolver_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert pr["domain_result"] == CONVERSATIONAL_RESPONSES["goodbye"]

    def test_chitchat_response(self):
        ir = make_intent_result(intent="chitchat", policy_action="allow", reason_code="CHITCHAT_ALLOWED")
        state = _make_quick_resolver_state(ir)
        result = quick_resolver_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert pr["domain_result"] == CONVERSATIONAL_RESPONSES["chitchat"]


class TestQuickResolverOrderStatus:
    @patch("app.graph.nodes.intent_executor.get_order_scoped")
    def test_order_found(self, mock_order):
        mock_order.return_value = {
            "order_id": "ORD-MM-1001",
            "product": "Wireless Headphones",
            "status": "delivered",
            "delivery_date": "2026-09-20",
        }
        ir = make_intent_result(
            intent="order_status", policy_action="allow",
            reason_code="ORDER_STATUS_ALLOWED",
            entities={"order_id": "ORD-MM-1001"},
        )
        state = _make_quick_resolver_state(ir)
        result = quick_resolver_node(state)

        mock_order.assert_called_once_with(TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-MM-1001")

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert pr["intent"] == "order_status"
        assert pr["entities"]["order_id"] == "ORD-MM-1001"
        assert "ORD-MM-1001" in pr["domain_result"]
        assert "delivered" in pr["domain_result"]
        assert "Delivered on 2026-09-20" in pr["domain_result"]

    @patch("app.graph.nodes.intent_executor.get_order_scoped")
    def test_order_not_found(self, mock_order):
        mock_order.return_value = None
        ir = make_intent_result(
            intent="order_status", policy_action="allow",
            reason_code="ORDER_STATUS_ALLOWED",
            entities={"order_id": "ORD-FAKE"},
        )
        state = _make_quick_resolver_state(ir)
        result = quick_resolver_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "couldn't find" in pr["domain_result"].lower()

    @patch("app.graph.nodes.intent_executor.get_order_scoped")
    def test_order_without_delivery_date(self, mock_order):
        mock_order.return_value = {
            "order_id": "ORD-MM-1005",
            "product": "Winter Jacket",
            "status": "processing",
            "delivery_date": None,
        }
        ir = make_intent_result(
            intent="order_status", policy_action="allow",
            reason_code="ORDER_STATUS_ALLOWED",
            entities={"order_id": "ORD-MM-1005"},
        )
        state = _make_quick_resolver_state(ir)
        result = quick_resolver_node(state)
        pr = result["processed_results"][0]
        assert "processing" in pr["domain_result"]
        assert "Delivered on" not in pr["domain_result"]

    @patch("app.graph.nodes.intent_executor.get_order_scoped", return_value=None)
    def test_order_status_without_order_id(self, mock_order):
        ir = make_intent_result(
            intent="order_status", policy_action="allow",
            reason_code="ORDER_STATUS_ALLOWED", entities={},
        )
        state = _make_quick_resolver_state(ir)
        result = quick_resolver_node(state)
        mock_order.assert_called_once_with(TENANT_MEGAMART, CUSTOMER_ALICE, "")
        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "couldn't find" in pr["domain_result"].lower()


class TestQuickResolverBlock:
    def test_block_returns_not_allowed_message(self):
        ir = make_intent_result(
            intent="refund", policy_action="block",
            reason_code="RETURNS_NOT_ALLOWED",
            entities={"order_id": "ORD-MM-1003"},
        )
        state = _make_quick_resolver_state(ir)
        result = quick_resolver_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "blocked"
        assert "not allowed" in pr["domain_result"].lower() or "ORD-MM-1003" in pr["domain_result"]

    def test_block_return_window_expired(self):
        ir = make_intent_result(
            intent="refund", policy_action="block",
            reason_code="RETURN_WINDOW_EXPIRED",
            entities={"order_id": "ORD-MM-1010"},
        )
        state = _make_quick_resolver_state(ir)
        result = quick_resolver_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "blocked"
        assert "expired" in pr["domain_result"].lower()


class TestQuickResolverEscalate:
    def test_escalation_keywords(self):
        ir = make_intent_result(
            intent="complaint", policy_action="escalate",
            reason_code="ESCALATION_KEYWORDS_DETECTED",
        )
        state = _make_quick_resolver_state(ir)
        result = quick_resolver_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "awaiting_human"
        assert "review" in pr["domain_result"].lower()

    def test_default_deny_escalation(self):
        ir = make_intent_result(
            intent="refund", policy_action="escalate",
            reason_code="DEFAULT_DENY",
        )
        state = _make_quick_resolver_state(ir)
        result = quick_resolver_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "awaiting_human"


class TestQuickResolverClarify:
    def test_low_confidence_clarify(self):
        ir = make_intent_result(
            intent="refund", policy_action="clarify",
            reason_code="LOW_CONFIDENCE",
        )
        state = _make_quick_resolver_state(ir)
        result = quick_resolver_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "needs_clarification"
        assert "rephrase" in pr["domain_result"].lower()

    def test_ambiguous_order_clarify_with_constraints(self):
        ir = make_intent_result(
            intent="refund", policy_action="clarify",
            reason_code="AMBIGUOUS_ORDER",
            constraints={"matching_orders": ["ORD-1 — Widget", "ORD-2 — Gadget"]},
        )
        state = _make_quick_resolver_state(ir)
        result = quick_resolver_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "needs_clarification"
        assert "ORD-1" in pr["domain_result"]
        assert "ORD-2" in pr["domain_result"]

    def test_missing_order_id_clarify(self):
        ir = make_intent_result(
            intent="refund", policy_action="clarify",
            reason_code="MISSING_ORDER_ID",
            constraints={"order_count": 5},
        )
        state = _make_quick_resolver_state(ir)
        result = quick_resolver_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "needs_clarification"
        assert "5" in pr["domain_result"]
