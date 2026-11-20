from __future__ import annotations

import pytest

from app.graph.nodes.policy_fan_out import policy_fan_out_node


class TestSingleIntent:
    def test_faq_allowed(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "What is your return policy?",
            "router_output": {
                "intent": "faq",
                "entities": {},
                "confidence": 0.95,
                "escalation_signals": [],
                "sub_intents": [],
            },
        }
        result = policy_fan_out_node(state)
        assert len(result["intent_results"]) == 1
        assert result["intent_results"][0]["policy_action"] == "allow"
        assert result["intent_results"][0]["reason_code"] == "FAQ_ALLOWED"
        assert result["policy_decision"]["action"] == "allow"

    def test_order_status_allowed(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "Where is my order?",
            "customer_email": "alice@example.com",
            "router_output": {
                "intent": "order_status",
                "entities": {"order_id": "ORD-MM-1001"},
                "confidence": 0.95,
                "escalation_signals": [],
                "sub_intents": [],
            },
        }
        result = policy_fan_out_node(state)
        assert result["intent_results"][0]["policy_action"] == "allow"

    def test_refund_escalated_tiny_candles(self):
        state = {
            "tenant_id": "tiny-candles",
            "user_message": "I want a refund",
            "customer_email": "grace@example.com",
            "router_output": {
                "intent": "refund",
                "entities": {"order_id": "ORD-TC-2001"},
                "confidence": 0.9,
                "escalation_signals": [],
                "sub_intents": [],
            },
        }
        result = policy_fan_out_node(state)
        assert result["intent_results"][0]["policy_action"] == "escalate"
        assert result["intent_results"][0]["reason_code"] == "TENANT_REFUND_REVIEW_REQUIRED"

    def test_out_of_scope_blocked(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "Tell me a joke",
            "router_output": {
                "intent": "out_of_scope",
                "entities": {},
                "confidence": 0.9,
                "escalation_signals": [],
                "sub_intents": [],
            },
        }
        result = policy_fan_out_node(state)
        assert result["intent_results"][0]["policy_action"] == "block"

    def test_inactive_tenant_blocked(self):
        state = {
            "tenant_id": "inactive-co",
            "user_message": "I want a refund",
            "router_output": {
                "intent": "refund",
                "entities": {},
                "confidence": 0.9,
                "escalation_signals": [],
                "sub_intents": [],
            },
        }
        result = policy_fan_out_node(state)
        assert result["intent_results"][0]["policy_action"] == "block"
        assert result["intent_results"][0]["reason_code"] == "TENANT_INACTIVE"


class TestMultiIntent:
    def test_two_intents_both_evaluated(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "Check ORD-MM-1001 and refund ORD-MM-1005",
            "customer_email": "alice@example.com",
            "router_output": {
                "intent": "order_status",
                "entities": {"order_id": "ORD-MM-1001"},
                "confidence": 0.9,
                "escalation_signals": [],
                "sub_intents": [
                    {
                        "intent": "refund",
                        "entities": {"order_id": "ORD-MM-1005"},
                        "confidence": 0.85,
                    }
                ],
            },
        }
        result = policy_fan_out_node(state)
        assert len(result["intent_results"]) == 2
        assert result["intent_results"][0]["intent"] == "order_status"
        assert result["intent_results"][0]["policy_action"] == "allow"
        assert result["intent_results"][1]["intent"] == "refund"

    def test_primary_decision_backward_compat(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "Check and refund",
            "customer_email": "alice@example.com",
            "router_output": {
                "intent": "order_status",
                "entities": {"order_id": "ORD-MM-1001"},
                "confidence": 0.9,
                "escalation_signals": [],
                "sub_intents": [
                    {
                        "intent": "refund",
                        "entities": {"order_id": "ORD-MM-1005"},
                        "confidence": 0.85,
                    }
                ],
            },
        }
        result = policy_fan_out_node(state)
        assert result["policy_decision"]["action"] == "allow"
        assert result["policy_decision"]["reason_code"] == "ORDER_STATUS_ALLOWED"

    def test_mixed_allow_and_escalate(self):
        state = {
            "tenant_id": "tiny-candles",
            "user_message": "Check order and refund",
            "customer_email": "grace@example.com",
            "router_output": {
                "intent": "order_status",
                "entities": {"order_id": "ORD-TC-2001"},
                "confidence": 0.9,
                "escalation_signals": [],
                "sub_intents": [
                    {
                        "intent": "refund",
                        "entities": {"order_id": "ORD-TC-2001"},
                        "confidence": 0.85,
                    }
                ],
            },
        }
        result = policy_fan_out_node(state)
        assert result["intent_results"][0]["policy_action"] == "allow"
        assert result["intent_results"][1]["policy_action"] == "escalate"

    def test_all_statuses_pending_after_fan_out(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "FAQ and order check",
            "router_output": {
                "intent": "faq",
                "entities": {},
                "confidence": 0.9,
                "escalation_signals": [],
                "sub_intents": [
                    {
                        "intent": "order_status",
                        "entities": {"order_id": "ORD-MM-1001"},
                        "confidence": 0.85,
                    }
                ],
            },
            "customer_email": "alice@example.com",
        }
        result = policy_fan_out_node(state)
        for ir in result["intent_results"]:
            assert ir["status"] == "pending"
