from __future__ import annotations

import pytest

from app.graph.nodes.intent_executor import (
    _build_block_context,
    _build_clarify_context,
    _build_escalation_context,
    _execute_cancel,
    _execute_order_status,
    _execute_refund,
    bail_executor_node,
    merge_results_node,
    quick_resolver_node,
)


class TestExecuteOrderStatus:
    def test_found_delivered(self):
        result = _execute_order_status("megamart", "CUST-MM-001", {"order_id": "ORD-MM-1001"})
        assert "ORD-MM-1001" in result
        assert "delivered" in result
        assert "Wireless Headphones" in result

    def test_found_processing(self):
        result = _execute_order_status("megamart", "CUST-MM-001", {"order_id": "ORD-MM-1005"})
        assert "processing" in result

    def test_not_found(self):
        result = _execute_order_status("megamart", "CUST-MM-001", {"order_id": "ORD-FAKE"})
        assert "couldn't find" in result


class TestExecuteRefund:
    def test_refund_with_constraints(self):
        result = _execute_refund(
            "megamart",
            "CUST-MM-001",
            {"order_id": "ORD-MM-1001"},
            {"refund_type": "original_method", "condition": "unopened"},
        )
        assert "4,500" in result
        assert "original_method" in result
        assert "unopened" in result

    def test_refund_no_condition(self):
        result = _execute_refund(
            "megamart",
            "CUST-MM-002",
            {"order_id": "ORD-MM-1002"},
            {"refund_type": "original_method", "condition": "any"},
        )
        assert "1,200" in result
        assert "unopened" not in result


class TestExecuteCancel:
    def test_cancel_found(self):
        result = _execute_cancel("megamart", "CUST-MM-006", {"order_id": "ORD-MM-1007"})
        assert "cancelled" in result
        assert "USB-C Hub" in result

    def test_cancel_not_found(self):
        result = _execute_cancel("megamart", "CUST-MM-001", {"order_id": "ORD-FAKE"})
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


class TestSimpleExecutorNode:
    def test_allow_order_status(self):
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
        assert len(result["processed_results"]) == 1
        r = result["processed_results"][0]
        assert r["status"] == "fulfilled"
        assert "ORD-MM-1001" in r["domain_result"]

    def test_block(self):
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

    def test_escalate(self):
        state = {
            "tenant_id": "tiny-candles",
            "user_message": "refund ORD-TC-2001",
            "current_intent": {
                "intent": "refund",
                "entities": {"order_id": "ORD-TC-2001"},
                "policy_action": "escalate",
                "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }
        result = quick_resolver_node(state)
        assert result["processed_results"][0]["status"] == "awaiting_human"

    def test_clarify(self):
        state = {
            "tenant_id": "megamart",
            "user_message": "help with something",
            "current_intent": {
                "intent": "refund",
                "entities": {},
                "policy_action": "clarify",
                "reason_code": "VERIFICATION_REQUIRED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }
        result = quick_resolver_node(state)
        assert result["processed_results"][0]["status"] == "needs_clarification"


class TestMergeResultsNode:
    def test_mixed_allow_and_escalate(self):
        state = {
            "processed_results": [
                {
                    "intent": "order_status",
                    "status": "fulfilled",
                    "domain_result": "Order status info",
                },
                {
                    "intent": "refund",
                    "status": "awaiting_human",
                    "domain_result": "Escalated",
                },
            ],
        }
        result = merge_results_node(state)
        assert result["intent_results"][0]["status"] == "fulfilled"
        assert result["intent_results"][1]["status"] == "awaiting_human"
        assert result["response_phase"] == "partial"

    def test_all_fulfilled(self):
        state = {
            "processed_results": [
                {"intent": "order_status", "status": "fulfilled", "domain_result": "done"},
            ],
        }
        result = merge_results_node(state)
        assert result["response_phase"] == "complete"
        assert result["has_tool_calls"] is False

    def test_has_tool_calls(self):
        state = {
            "processed_results": [
                {"intent": "refund", "status": "fulfilled", "tool_calls": [{"tool": "initiate_refund"}]},
            ],
        }
        result = merge_results_node(state)
        assert result["has_tool_calls"] is True


class TestBailExecutorNode:
    def test_bail_block(self):
        state = {
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {},
                    "policy_action": "block",
                    "reason_code": "TENANT_INACTIVE",
                    "constraints": {},
                    "status": "pending",
                }
            ],
        }
        result = bail_executor_node(state)
        assert result["intent_results"][0]["status"] == "blocked"
        assert result["response_phase"] == "complete"
        assert result["has_tool_calls"] is False

    def test_bail_skips_non_pending(self):
        state = {
            "intent_results": [
                {
                    "intent": "order_status",
                    "entities": {},
                    "policy_action": "allow",
                    "reason_code": "OK",
                    "constraints": {},
                    "domain_result": "Already done",
                    "status": "fulfilled",
                }
            ],
        }
        result = bail_executor_node(state)
        assert result["intent_results"][0]["domain_result"] == "Already done"
