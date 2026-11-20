from __future__ import annotations

from unittest.mock import patch

import pytest

from app.graph.nodes.escalation_handler import escalation_handler_node


class TestNoEscalations:
    def test_pass_through_no_escalations(self):
        state = {
            "intent_results": [
                {"intent": "faq", "status": "fulfilled", "domain_result": "answer"},
            ],
        }
        result = escalation_handler_node(state)
        assert result == {}

    def test_pass_through_empty_results(self):
        state = {"intent_results": []}
        result = escalation_handler_node(state)
        assert result == {}


class TestSingleEscalation:
    def test_approve(self):
        state = {
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-TC-2001"},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "domain_result": "Refund requires review.",
                },
            ],
        }
        human_response = {"decisions": [{"index": 0, "action": "approve"}]}

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        assert result["intent_results"][0]["status"] == "fulfilled"
        assert "Approved by support team" in result["intent_results"][0]["domain_result"]
        assert result["intent_results"][0]["_newly_resolved"] is True
        assert result["response_phase"] == "follow_up"

    def test_approve_with_note(self):
        state = {
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-TC-2001"},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "domain_result": "Refund requires review.",
                },
            ],
        }
        human_response = {
            "decisions": [{"index": 0, "action": "approve", "note": "Exception granted"}]
        }

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        assert result["intent_results"][0]["status"] == "fulfilled"
        assert "Exception granted" in result["intent_results"][0]["domain_result"]

    def test_deny(self):
        state = {
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-TC-2001"},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "domain_result": "Refund requires review.",
                },
            ],
        }
        human_response = {
            "decisions": [{"index": 0, "action": "deny", "reason": "Item was used"}]
        }

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        assert result["intent_results"][0]["status"] == "blocked"
        assert result["intent_results"][0]["reason_code"] == "HUMAN_DENIED"
        assert "Item was used" in result["intent_results"][0]["domain_result"]

    def test_override(self):
        state = {
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-TC-2001"},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "domain_result": "Refund requires review.",
                },
            ],
        }
        human_response = {
            "decisions": [
                {
                    "index": 0,
                    "action": "override",
                    "custom_response": "We will process a store credit of ₹1,450.",
                }
            ]
        }

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        assert result["intent_results"][0]["status"] == "fulfilled"
        assert "store credit" in result["intent_results"][0]["domain_result"]


class TestMultipleEscalations:
    def test_two_escalations_batched(self):
        state = {
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-TC-2001"},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "domain_result": "Review needed.",
                },
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-TC-2005"},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "domain_result": "Review needed.",
                },
            ],
        }
        human_response = {
            "decisions": [
                {"index": 0, "action": "approve"},
                {"index": 1, "action": "deny", "reason": "Too old"},
            ]
        }

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        assert result["intent_results"][0]["status"] == "fulfilled"
        assert result["intent_results"][1]["status"] == "blocked"
        assert result["response_phase"] == "follow_up"

    def test_mixed_fulfilled_and_escalated(self):
        state = {
            "intent_results": [
                {
                    "intent": "order_status",
                    "entities": {"order_id": "ORD-TC-2001"},
                    "status": "fulfilled",
                    "reason_code": "ORDER_STATUS_ALLOWED",
                    "domain_result": "Shipped.",
                },
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-TC-2001"},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "domain_result": "Review needed.",
                },
            ],
        }
        human_response = {
            "decisions": [{"index": 1, "action": "approve"}]
        }

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        assert result["intent_results"][0]["status"] == "fulfilled"
        assert result["intent_results"][0].get("_newly_resolved") is None
        assert result["intent_results"][1]["status"] == "fulfilled"
        assert result["intent_results"][1]["_newly_resolved"] is True


class TestEdgeCases:
    def test_invalid_index_ignored(self):
        state = {
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "domain_result": "Review needed.",
                },
            ],
        }
        human_response = {
            "decisions": [{"index": 99, "action": "approve"}]
        }

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        assert result["intent_results"][0]["status"] == "awaiting_human"

    def test_list_format_decisions(self):
        state = {
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "domain_result": "Review needed.",
                },
            ],
        }
        human_response = [{"index": 0, "action": "approve"}]

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        assert result["intent_results"][0]["status"] == "fulfilled"
