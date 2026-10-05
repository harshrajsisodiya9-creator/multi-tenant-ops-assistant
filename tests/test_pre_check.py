from __future__ import annotations

import pytest

from app.graph.nodes.pre_check import pre_check_node


class TestTenantInactive:
    def test_inactive_tenant_bails(self):
        state = {
            "tenant_id": "inactive-co",
            "user_message": "I want a refund",
            "router_output": {
                "intent": "refund",
                "entities": {"order_id": "ORD-1"},
                "confidence": 0.9,
                "escalation_signals": [],
                "sub_intents": [],
            },
        }
        result = pre_check_node(state)
        assert result["pre_check_bail"] is True
        assert result["intent_results"][0]["policy_action"] == "block"
        assert result["intent_results"][0]["reason_code"] == "TENANT_INACTIVE"

    def test_active_tenant_passes(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "Where is my order?",
            "router_output": {
                "intent": "order_status",
                "entities": {},
                "confidence": 0.9,
                "escalation_signals": [],
                "sub_intents": [],
            },
        }
        result = pre_check_node(state)
        assert result["pre_check_bail"] is False
        assert "intent_results" not in result


class TestLowConfidence:
    def test_low_confidence_bails(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "something unclear",
            "router_output": {
                "intent": "faq",
                "entities": {},
                "confidence": 0.3,
                "escalation_signals": [],
                "sub_intents": [],
            },
        }
        result = pre_check_node(state)
        assert result["pre_check_bail"] is True
        assert result["intent_results"][0]["policy_action"] == "clarify"
        assert result["intent_results"][0]["reason_code"] == "LOW_CONFIDENCE"

    def test_adequate_confidence_passes(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "Return policy?",
            "router_output": {
                "intent": "faq",
                "entities": {},
                "confidence": 0.8,
                "escalation_signals": [],
                "sub_intents": [],
            },
        }
        result = pre_check_node(state)
        assert result["pre_check_bail"] is False

    def test_boundary_confidence_0_6_passes(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "Maybe return?",
            "router_output": {
                "intent": "refund",
                "entities": {},
                "confidence": 0.6,
                "escalation_signals": [],
                "sub_intents": [],
            },
        }
        result = pre_check_node(state)
        assert result["pre_check_bail"] is False


class TestOutOfScope:
    def test_out_of_scope_bails(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "Tell me a joke",
            "router_output": {
                "intent": "out_of_scope",
                "entities": {},
                "confidence": 0.95,
                "escalation_signals": [],
                "sub_intents": [],
            },
        }
        result = pre_check_node(state)
        assert result["pre_check_bail"] is True
        assert result["intent_results"][0]["policy_action"] == "block"
        assert result["intent_results"][0]["reason_code"] == "OUT_OF_SCOPE"

    def test_faq_does_not_bail(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "What is your return policy?",
            "router_output": {
                "intent": "faq",
                "entities": {},
                "confidence": 0.9,
                "escalation_signals": [],
                "sub_intents": [],
            },
        }
        result = pre_check_node(state)
        assert result["pre_check_bail"] is False


class TestPriorityOrder:
    def test_inactive_takes_priority_over_low_confidence(self):
        state = {
            "tenant_id": "inactive-co",
            "user_message": "hmm",
            "router_output": {
                "intent": "faq",
                "entities": {},
                "confidence": 0.3,
                "escalation_signals": [],
                "sub_intents": [],
            },
        }
        result = pre_check_node(state)
        assert result["intent_results"][0]["reason_code"] == "TENANT_INACTIVE"
