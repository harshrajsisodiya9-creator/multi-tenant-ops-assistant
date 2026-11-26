"""Tests for policy_fan_out_node.

Covers: gate skipping for FAQ/conversational, single intent, multi-intent,
multi-order expansion, entity resolution, escalation, complaint handling,
pre_check_bail short-circuit.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.graph.nodes.policy_fan_out import (
    GATE_SKIP_INTENTS,
    _expand_multi_order_intents,
    _product_matches_message,
    _skip_gate,
    policy_fan_out_node,
)
from app.graph.schemas import PolicyDecision

from .conftest import (
    CUSTOMER_ALICE,
    CUSTOMER_BOB,
    TENANT_MEGAMART,
    make_base_state,
    make_router_output,
)


# ===================================================================
# _product_matches_message
# ===================================================================

class TestProductMatchesMessage:
    def test_exact_keyword_match(self):
        assert _product_matches_message("Wireless Headphones", "i want to return my wireless headphones") is True

    def test_no_match(self):
        assert _product_matches_message("Wireless Headphones", "cancel my subscription") is False

    def test_short_words_ignored(self):
        assert _product_matches_message("USB C Hub", "i have a hub") is False

    def test_hyphenated_keyword_does_not_match_partial(self):
        """'T-Shirt' (keyword) does not match 'shirt' (message word) — hyphen kept by split."""
        assert _product_matches_message("Cotton T-Shirt", "i received the shirt yesterday") is False

    def test_rstrip_s_stemming_matches(self):
        """rstrip('s') stems: keyword 'shirts' matches message word 'shirt'."""
        assert _product_matches_message("Cotton Shirts", "i received the shirt yesterday") is True


# ===================================================================
# _skip_gate
# ===================================================================

class TestSkipGate:
    def test_faq_skips(self):
        result = _skip_gate("faq", [])
        assert result is not None
        assert result.action == "allow"
        assert result.reason_code == "FAQ_ALLOWED"

    def test_greeting_skips(self):
        result = _skip_gate("greeting", [])
        assert result is not None
        assert result.reason_code == "GREETING_ALLOWED"

    def test_complaint_without_escalation_skips(self):
        result = _skip_gate("complaint", [])
        assert result is not None
        assert result.reason_code == "COMPLAINT_ALLOWED"

    def test_complaint_with_escalation_does_not_skip(self):
        result = _skip_gate("complaint", ["fraud"])
        assert result is None

    def test_refund_does_not_skip(self):
        result = _skip_gate("refund", [])
        assert result is None

    def test_order_status_does_not_skip(self):
        result = _skip_gate("order_status", [])
        assert result is None


# ===================================================================
# _expand_multi_order_intents
# ===================================================================

class TestExpandMultiOrderIntents:
    def test_single_order_no_expansion(self):
        intents = [("refund", {"order_id": "ORD-1"}, 0.9)]
        result = _expand_multi_order_intents(intents)
        assert len(result) == 1

    def test_multiple_order_ids_expanded(self):
        intents = [
            ("refund", {"order_id": "ORD-1", "all_order_ids": ["ORD-1", "ORD-2"]}, 0.9),
        ]
        result = _expand_multi_order_intents(intents)
        assert len(result) == 2
        assert result[0][1]["order_id"] == "ORD-1"
        assert result[1][1]["order_id"] == "ORD-2"
        assert "all_order_ids" not in result[0][1]
        assert "all_order_ids" not in result[1][1]

    def test_empty_list(self):
        result = _expand_multi_order_intents([])
        assert result == []


# ===================================================================
# policy_fan_out_node
# ===================================================================

class TestPolicyFanOutNode:
    def test_pre_check_bail_returns_empty(self):
        state = make_base_state()
        state["pre_check_bail"] = True
        state["router_output"] = make_router_output()
        result = policy_fan_out_node(state)
        assert result == {}

    @patch("app.graph.nodes.policy_fan_out.get_tenant_info")
    @patch("app.graph.nodes.policy_fan_out.get_order_scoped")
    @patch("app.graph.nodes.policy_fan_out.get_customer_orders")
    @patch("app.graph.nodes.policy_fan_out.get_refund_count_30d")
    def test_faq_intent_allowed(self, mock_refund, mock_orders, mock_order, mock_tenant):
        mock_tenant.return_value = {"active": True}
        state = make_base_state(user_message="How do I return an item?")
        state["pre_check_bail"] = False
        state["router_output"] = make_router_output(intent="faq", confidence=0.9)

        result = policy_fan_out_node(state)
        assert len(result["intent_results"]) == 1
        assert result["intent_results"][0]["policy_action"] == "allow"
        assert result["intent_results"][0]["reason_code"] == "FAQ_ALLOWED"

    @patch("app.graph.nodes.policy_fan_out.get_tenant_info")
    @patch("app.graph.nodes.policy_fan_out.get_order_scoped")
    @patch("app.graph.nodes.policy_fan_out.get_customer_orders")
    @patch("app.graph.nodes.policy_fan_out.get_refund_count_30d")
    def test_greeting_intent_allowed(self, mock_refund, mock_orders, mock_order, mock_tenant):
        mock_tenant.return_value = {"active": True}
        state = make_base_state(user_message="Hello")
        state["pre_check_bail"] = False
        state["router_output"] = make_router_output(intent="greeting", confidence=0.95)

        result = policy_fan_out_node(state)
        assert result["intent_results"][0]["policy_action"] == "allow"
        assert result["intent_results"][0]["reason_code"] == "GREETING_ALLOWED"

    @patch("app.graph.nodes.policy_fan_out.evaluate_policy")
    @patch("app.graph.nodes.policy_fan_out.get_tenant_info")
    @patch("app.graph.nodes.policy_fan_out.get_order_scoped")
    @patch("app.graph.nodes.policy_fan_out.get_customer_orders")
    @patch("app.graph.nodes.policy_fan_out.get_refund_count_30d")
    def test_order_status_goes_through_gate(self, mock_refund, mock_orders, mock_order, mock_tenant, mock_policy):
        mock_tenant.return_value = {"active": True}
        mock_order.return_value = {"order_id": "ORD-MM-1001", "status": "delivered"}
        mock_policy.return_value = PolicyDecision(action="allow", reason_code="ORDER_STATUS_ALLOWED")

        state = make_base_state(user_message="Where is order ORD-MM-1001?")
        state["pre_check_bail"] = False
        state["router_output"] = make_router_output(
            intent="order_status",
            entities={"order_id": "ORD-MM-1001"},
            confidence=0.95,
        )
        result = policy_fan_out_node(state)

        mock_policy.assert_called_once()
        gate_ctx = mock_policy.call_args[0][0]
        assert gate_ctx.tenant_id == TENANT_MEGAMART
        assert gate_ctx.router_output.intent == "order_status"
        assert gate_ctx.customer_id == CUSTOMER_ALICE

        assert len(result["intent_results"]) == 1
        ir = result["intent_results"][0]
        assert ir["intent"] == "order_status"
        assert ir["policy_action"] == "allow"
        assert ir["reason_code"] == "ORDER_STATUS_ALLOWED"
        assert ir["entities"]["order_id"] == "ORD-MM-1001"

    @patch("app.graph.nodes.policy_fan_out.evaluate_policy")
    @patch("app.graph.nodes.policy_fan_out.get_tenant_info")
    @patch("app.graph.nodes.policy_fan_out.get_order_scoped")
    @patch("app.graph.nodes.policy_fan_out.get_customer_orders")
    @patch("app.graph.nodes.policy_fan_out.get_refund_count_30d")
    def test_multi_intent_produces_multiple_results(
        self, mock_refund, mock_orders, mock_order, mock_tenant, mock_policy,
    ):
        mock_tenant.return_value = {"active": True}
        mock_order.return_value = {"order_id": "ORD-MM-1001"}
        mock_policy.return_value = PolicyDecision(action="allow", reason_code="ORDER_STATUS_ALLOWED")

        state = make_base_state(user_message="Check ORD-MM-1001 and cancel ORD-MM-1002")
        state["pre_check_bail"] = False
        state["router_output"] = make_router_output(
            intent="order_status",
            entities={"order_id": "ORD-MM-1001"},
            confidence=0.9,
            sub_intents=[
                {"intent": "cancel_order", "entities": {"order_id": "ORD-MM-1002"}, "confidence": 0.88},
            ],
        )
        result = policy_fan_out_node(state)
        assert len(result["intent_results"]) == 2

        by_intent = {ir["intent"]: ir for ir in result["intent_results"]}
        assert set(by_intent) == {"order_status", "cancel_order"}
        assert by_intent["order_status"]["entities"]["order_id"] == "ORD-MM-1001"
        assert by_intent["cancel_order"]["entities"]["order_id"] == "ORD-MM-1002"
        assert by_intent["order_status"]["policy_action"] == "allow"
        assert by_intent["cancel_order"]["policy_action"] == "allow"
        assert by_intent["order_status"]["reason_code"] == "ORDER_STATUS_ALLOWED"
        assert by_intent["cancel_order"]["reason_code"] == "ORDER_STATUS_ALLOWED"

    @patch("app.graph.nodes.policy_fan_out.evaluate_policy")
    @patch("app.graph.nodes.policy_fan_out.get_tenant_info")
    @patch("app.graph.nodes.policy_fan_out.get_order_scoped")
    @patch("app.graph.nodes.policy_fan_out.get_customer_orders")
    @patch("app.graph.nodes.policy_fan_out.get_refund_count_30d")
    def test_conditional_carried_on_primary_only(
        self, mock_refund, mock_orders, mock_order, mock_tenant, mock_policy,
    ):
        """Conditional data appears on the primary intent, not on the sub-intent."""
        mock_tenant.return_value = {"active": True}
        mock_order.return_value = {"order_id": "ORD-MM-1001", "status": "pending"}
        mock_refund.return_value = 0
        mock_policy.return_value = PolicyDecision(action="allow", reason_code="CANCEL_ALLOWED")

        state = make_base_state(
            user_message="If it hasn't shipped, cancel ORD-MM-1001 and also check ORD-MM-1002",
        )
        state["pre_check_bail"] = False
        state["router_output"] = make_router_output(
            intent="cancel_order",
            entities={"order_id": "ORD-MM-1001"},
            confidence=0.92,
            conditional={
                "request_type": "conditional_action",
                "condition": {"field": "order_status", "operator": "not_equals", "value": "shipped"},
            },
            sub_intents=[
                {"intent": "order_status", "entities": {"order_id": "ORD-MM-1002"}, "confidence": 0.88},
            ],
        )
        result = policy_fan_out_node(state)

        assert len(result["intent_results"]) == 2
        assert result["intent_results"][0]["conditional"] is not None
        assert result["intent_results"][0]["conditional"]["request_type"] == "conditional_action"
        assert result["intent_results"][1]["conditional"] is None


class TestPolicyFanOutEntityResolution:
    """Tests for order entity resolution within policy_fan_out_node."""

    @patch("app.graph.nodes.policy_fan_out.evaluate_policy")
    @patch("app.graph.nodes.policy_fan_out.get_tenant_info")
    @patch("app.graph.nodes.policy_fan_out.get_order_scoped")
    @patch("app.graph.nodes.policy_fan_out.get_customer_orders")
    @patch("app.graph.nodes.policy_fan_out.get_refund_count_30d")
    def test_single_order_auto_resolved(
        self, mock_refund, mock_orders, mock_order, mock_tenant, mock_policy,
    ):
        mock_tenant.return_value = {"active": True}
        mock_orders.return_value = [{"order_id": "ORD-MM-1001", "product": "Widget"}]
        mock_order.return_value = {"order_id": "ORD-MM-1001"}
        mock_refund.return_value = 0
        mock_policy.return_value = PolicyDecision(action="allow", reason_code="REFUND_ALLOWED")

        state = make_base_state(user_message="Refund my order")
        state["pre_check_bail"] = False
        state["router_output"] = make_router_output(
            intent="refund", entities={}, confidence=0.9,
        )
        result = policy_fan_out_node(state)
        assert result["intent_results"][0]["entities"]["order_id"] == "ORD-MM-1001"

    @patch("app.graph.nodes.policy_fan_out.get_tenant_info")
    @patch("app.graph.nodes.policy_fan_out.get_order_scoped")
    @patch("app.graph.nodes.policy_fan_out.get_customer_orders")
    @patch("app.graph.nodes.policy_fan_out.get_refund_count_30d")
    def test_no_customer_id_no_resolution(
        self, mock_refund, mock_orders, mock_order, mock_tenant,
    ):
        """Without customer_id, entity resolution is skipped."""
        mock_tenant.return_value = {"active": True}
        mock_order.return_value = None
        mock_refund.return_value = 0

        state = make_base_state(customer_id="", user_message="Refund my order")
        state["pre_check_bail"] = False
        state["router_output"] = make_router_output(
            intent="refund", entities={}, confidence=0.9,
        )
        result = policy_fan_out_node(state)
        assert "order_id" not in result["intent_results"][0]["entities"]

    @patch("app.graph.nodes.policy_fan_out.get_tenant_info")
    @patch("app.graph.nodes.policy_fan_out.get_order_scoped")
    @patch("app.graph.nodes.policy_fan_out.get_customer_orders")
    @patch("app.graph.nodes.policy_fan_out.get_refund_count_30d")
    def test_no_orders_found_clarifies(
        self, mock_refund, mock_orders, mock_order, mock_tenant,
    ):
        mock_tenant.return_value = {"active": True}
        mock_orders.return_value = []
        mock_refund.return_value = 0

        state = make_base_state(user_message="Refund my order")
        state["pre_check_bail"] = False
        state["router_output"] = make_router_output(
            intent="refund", entities={}, confidence=0.9,
        )
        result = policy_fan_out_node(state)
        assert result["intent_results"][0]["policy_action"] == "clarify"
        assert result["intent_results"][0]["reason_code"] == "NO_ORDERS_FOUND"

    @patch("app.graph.nodes.policy_fan_out.get_tenant_info")
    @patch("app.graph.nodes.policy_fan_out.get_customer_orders")
    def test_ambiguous_product_match_clarifies(self, mock_orders, mock_tenant):
        mock_tenant.return_value = {"active": True}
        mock_orders.return_value = [
            {"order_id": "ORD-1", "product": "Wireless Headphones", "amount": 1000},
            {"order_id": "ORD-2", "product": "Wireless Headphones Pro", "amount": 2000},
        ]
        state = make_base_state(user_message="Refund my wireless headphones")
        state["pre_check_bail"] = False
        state["router_output"] = make_router_output(intent="refund", entities={}, confidence=0.9)
        result = policy_fan_out_node(state)
        ir = result["intent_results"][0]
        assert ir["policy_action"] == "clarify"
        assert ir["reason_code"] == "AMBIGUOUS_ORDER"
        assert len(ir["constraints"]["matching_orders"]) == 2

    @patch("app.graph.nodes.policy_fan_out.get_tenant_info")
    @patch("app.graph.nodes.policy_fan_out.get_customer_orders")
    def test_multiple_orders_no_product_match_clarifies(self, mock_orders, mock_tenant):
        mock_tenant.return_value = {"active": True}
        mock_orders.return_value = [
            {"order_id": "ORD-1", "product": "Desk Lamp"},
            {"order_id": "ORD-2", "product": "Coffee Mug"},
        ]
        state = make_base_state(user_message="Refund my order")
        state["pre_check_bail"] = False
        state["router_output"] = make_router_output(intent="refund", entities={}, confidence=0.9)
        result = policy_fan_out_node(state)
        ir = result["intent_results"][0]
        assert ir["policy_action"] == "clarify"
        assert ir["reason_code"] == "MISSING_ORDER_ID"
        assert ir["constraints"]["order_count"] == 2

    @patch("app.graph.nodes.policy_fan_out.evaluate_policy")
    @patch("app.graph.nodes.policy_fan_out.get_tenant_info")
    @patch("app.graph.nodes.policy_fan_out.get_refund_count_30d")
    def test_complaint_with_escalation_goes_through_gate(
        self, mock_refund, mock_tenant, mock_policy,
    ):
        mock_tenant.return_value = {"active": True}
        mock_refund.return_value = 0
        mock_policy.return_value = PolicyDecision(
            action="escalate", reason_code="ESCALATION_KEYWORDS_DETECTED",
        )
        state = make_base_state(user_message="I will contact my lawyer")
        state["pre_check_bail"] = False
        state["router_output"] = make_router_output(
            intent="complaint", escalation_signals=["lawyer"], confidence=0.9,
        )
        result = policy_fan_out_node(state)
        mock_policy.assert_called_once()
        assert result["intent_results"][0]["policy_action"] == "escalate"
        assert result["intent_results"][0]["reason_code"] == "ESCALATION_KEYWORDS_DETECTED"
