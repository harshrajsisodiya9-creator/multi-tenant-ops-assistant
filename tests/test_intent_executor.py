from __future__ import annotations

import pytest

from app.graph.nodes.intent_executor import (
    _build_block_context,
    _build_clarify_context,
    _build_escalation_context,
    _execute_cancel,
    _execute_order_status,
    _execute_refund,
    intent_executor_node,
)


class TestExecuteOrderStatus:
    def test_found_delivered(self):
        result = _execute_order_status("megamart", {"order_id": "ORD-MM-1001"})
        assert "ORD-MM-1001" in result
        assert "delivered" in result
        assert "Wireless Headphones" in result

    def test_found_processing(self):
        result = _execute_order_status("megamart", {"order_id": "ORD-MM-1005"})
        assert "processing" in result

    def test_not_found(self):
        result = _execute_order_status("megamart", {"order_id": "ORD-FAKE"})
        assert "couldn't find" in result


class TestExecuteRefund:
    def test_refund_with_constraints(self):
        result = _execute_refund(
            "megamart",
            {"order_id": "ORD-MM-1001"},
            {"refund_type": "original_method", "condition": "unopened"},
        )
        assert "4,500" in result
        assert "original_method" in result
        assert "unopened" in result

    def test_refund_no_condition(self):
        result = _execute_refund(
            "megamart",
            {"order_id": "ORD-MM-1002"},
            {"refund_type": "original_method", "condition": "any"},
        )
        assert "1,200" in result
        assert "unopened" not in result


class TestExecuteCancel:
    def test_cancel_found(self):
        result = _execute_cancel("megamart", {"order_id": "ORD-MM-1007"})
        assert "cancelled" in result
        assert "USB-C Hub" in result

    def test_cancel_not_found(self):
        result = _execute_cancel("megamart", {"order_id": "ORD-FAKE"})
        assert "initiated" in result


class TestBlockContext:
    def test_returns_not_allowed(self):
        msg = _build_block_context("RETURNS_NOT_ALLOWED", {"order_id": "ORD-1"})
        assert "not allowed" in msg.lower()

    def test_return_window_expired(self):
        msg = _build_block_context("RETURN_WINDOW_EXPIRED", {"order_id": "ORD-1"})
        assert "expired" in msg.lower()

    def test_unknown_code(self):
        msg = _build_block_context("UNKNOWN_CODE", {})
        assert "UNKNOWN_CODE" in msg


class TestEscalationContext:
    def test_refund_ceiling(self):
        msg = _build_escalation_context("GLOBAL_REFUND_CEILING", {"order_id": "ORD-1"})
        assert "limit" in msg.lower()

    def test_rate_limited(self):
        msg = _build_escalation_context("REFUND_RATE_LIMITED", {})
        assert "recent" in msg.lower()


class TestClarifyContext:
    def test_low_confidence(self):
        msg = _build_clarify_context("LOW_CONFIDENCE")
        assert "rephrase" in msg.lower()

    def test_verification_required(self):
        msg = _build_clarify_context("VERIFICATION_REQUIRED")
        assert "order number" in msg.lower()


class TestIntentExecutorNode:
    def test_single_allow_order_status(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "Where is ORD-MM-1001?",
            "intent_results": [
                {
                    "intent": "order_status",
                    "entities": {"order_id": "ORD-MM-1001"},
                    "policy_action": "allow",
                    "reason_code": "ORDER_STATUS_ALLOWED",
                    "constraints": {},
                    "domain_result": "",
                    "status": "pending",
                }
            ],
        }
        result = intent_executor_node(state)
        assert result["intent_results"][0]["status"] == "fulfilled"
        assert "ORD-MM-1001" in result["intent_results"][0]["domain_result"]
        assert result["response_phase"] == "complete"

    def test_single_block(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "return perishables",
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-MM-1003"},
                    "policy_action": "block",
                    "reason_code": "RETURNS_NOT_ALLOWED",
                    "constraints": {},
                    "domain_result": "",
                    "status": "pending",
                }
            ],
        }
        result = intent_executor_node(state)
        assert result["intent_results"][0]["status"] == "blocked"
        assert result["response_phase"] == "complete"

    def test_single_escalate_sets_awaiting_human(self):
        state = {
            "tenant_id": "tiny-candles",
            "user_message": "refund ORD-TC-2001",
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-TC-2001"},
                    "policy_action": "escalate",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "constraints": {},
                    "domain_result": "",
                    "status": "pending",
                }
            ],
        }
        result = intent_executor_node(state)
        assert result["intent_results"][0]["status"] == "awaiting_human"
        assert result["response_phase"] == "partial"

    def test_single_clarify(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "help with something",
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {},
                    "policy_action": "clarify",
                    "reason_code": "VERIFICATION_REQUIRED",
                    "constraints": {},
                    "domain_result": "",
                    "status": "pending",
                }
            ],
        }
        result = intent_executor_node(state)
        assert result["intent_results"][0]["status"] == "needs_clarification"
        assert result["response_phase"] == "complete"

    def test_mixed_allow_and_escalate(self):
        state = {
            "tenant_id": "tiny-candles",
            "user_message": "check order and refund",
            "intent_results": [
                {
                    "intent": "order_status",
                    "entities": {"order_id": "ORD-TC-2001"},
                    "policy_action": "allow",
                    "reason_code": "ORDER_STATUS_ALLOWED",
                    "constraints": {},
                    "domain_result": "",
                    "status": "pending",
                },
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-TC-2001"},
                    "policy_action": "escalate",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "constraints": {},
                    "domain_result": "",
                    "status": "pending",
                },
            ],
        }
        result = intent_executor_node(state)
        assert result["intent_results"][0]["status"] == "fulfilled"
        assert result["intent_results"][1]["status"] == "awaiting_human"
        assert result["response_phase"] == "partial"

    def test_skips_non_pending(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "check order",
            "intent_results": [
                {
                    "intent": "order_status",
                    "entities": {"order_id": "ORD-MM-1001"},
                    "policy_action": "allow",
                    "reason_code": "ORDER_STATUS_ALLOWED",
                    "constraints": {},
                    "domain_result": "Already done",
                    "status": "fulfilled",
                }
            ],
        }
        result = intent_executor_node(state)
        assert result["intent_results"][0]["domain_result"] == "Already done"
        assert result["response_phase"] == "complete"
