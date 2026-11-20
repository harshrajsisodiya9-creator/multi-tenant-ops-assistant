from __future__ import annotations

import pytest

from app.graph.nodes.action_check import action_check_node


class TestRefundChecks:
    def test_refund_within_ceiling_passes(self):
        state = {
            "tenant_id": "megamart",
            "intent_results": [
                {
                    "intent": "refund",
                    "status": "fulfilled",
                    "domain_result": "Refund processed.",
                    "reason_code": "REFUND_ALLOWED",
                    "tool_calls": [
                        {
                            "tool": "initiate_refund",
                            "args": {"order_id": "ORD-MM-1001", "amount": 4500},
                        }
                    ],
                }
            ],
        }
        result = action_check_node(state)
        assert result["intent_results"][0]["status"] == "fulfilled"

    def test_refund_exceeds_global_ceiling_blocked(self):
        state = {
            "tenant_id": "megamart",
            "intent_results": [
                {
                    "intent": "refund",
                    "status": "fulfilled",
                    "domain_result": "Refund processed.",
                    "reason_code": "REFUND_ALLOWED",
                    "tool_calls": [
                        {
                            "tool": "initiate_refund",
                            "args": {"order_id": "ORD-MM-1004", "amount": 55000},
                        }
                    ],
                }
            ],
        }
        result = action_check_node(state)
        assert result["intent_results"][0]["status"] == "blocked"
        assert "ceiling" in result["intent_results"][0]["domain_result"].lower()
        assert result["intent_results"][0]["reason_code"] == "ACTION_CHECK_VIOLATION"

    def test_refund_exceeds_order_total_blocked(self):
        state = {
            "tenant_id": "megamart",
            "intent_results": [
                {
                    "intent": "refund",
                    "status": "fulfilled",
                    "domain_result": "Refund processed.",
                    "reason_code": "REFUND_ALLOWED",
                    "tool_calls": [
                        {
                            "tool": "initiate_refund",
                            "args": {"order_id": "ORD-MM-1001", "amount": 9999},
                        }
                    ],
                }
            ],
        }
        result = action_check_node(state)
        assert result["intent_results"][0]["status"] == "blocked"
        assert "order total" in result["intent_results"][0]["domain_result"].lower()


class TestCancelChecks:
    def test_cancel_pending_passes(self):
        state = {
            "tenant_id": "megamart",
            "intent_results": [
                {
                    "intent": "cancel_order",
                    "status": "fulfilled",
                    "domain_result": "Order cancelled.",
                    "reason_code": "CANCEL_ALLOWED",
                    "tool_calls": [
                        {
                            "tool": "cancel_order",
                            "args": {"order_id": "ORD-MM-1007"},
                        }
                    ],
                }
            ],
        }
        result = action_check_node(state)
        assert result["intent_results"][0]["status"] == "fulfilled"

    def test_cancel_shipped_blocked(self):
        state = {
            "tenant_id": "megamart",
            "intent_results": [
                {
                    "intent": "cancel_order",
                    "status": "fulfilled",
                    "domain_result": "Order cancelled.",
                    "reason_code": "CANCEL_ALLOWED",
                    "tool_calls": [
                        {
                            "tool": "cancel_order",
                            "args": {"order_id": "ORD-MM-1006"},
                        }
                    ],
                }
            ],
        }
        result = action_check_node(state)
        assert result["intent_results"][0]["status"] == "blocked"
        assert "not cancellable" in result["intent_results"][0]["domain_result"].lower()

    def test_cancel_unknown_order_passes(self):
        state = {
            "tenant_id": "megamart",
            "intent_results": [
                {
                    "intent": "cancel_order",
                    "status": "fulfilled",
                    "domain_result": "Cancelled.",
                    "reason_code": "CANCEL_ALLOWED",
                    "tool_calls": [
                        {
                            "tool": "cancel_order",
                            "args": {"order_id": "ORD-UNKNOWN"},
                        }
                    ],
                }
            ],
        }
        result = action_check_node(state)
        assert result["intent_results"][0]["status"] == "fulfilled"


class TestEdgeCases:
    def test_no_tool_calls_passes(self):
        state = {
            "tenant_id": "megamart",
            "intent_results": [
                {
                    "intent": "faq",
                    "status": "fulfilled",
                    "domain_result": "FAQ answer.",
                    "reason_code": "FAQ_ALLOWED",
                    "tool_calls": [],
                }
            ],
        }
        result = action_check_node(state)
        assert result["intent_results"][0]["status"] == "fulfilled"

    def test_blocked_tool_calls_skipped(self):
        state = {
            "tenant_id": "megamart",
            "intent_results": [
                {
                    "intent": "refund",
                    "status": "fulfilled",
                    "domain_result": "Refund.",
                    "reason_code": "REFUND_ALLOWED",
                    "tool_calls": [
                        {
                            "tool": "initiate_refund",
                            "args": {"order_id": "ORD-MM-1004", "amount": 55000},
                            "blocked": True,
                            "reason": "already blocked",
                        }
                    ],
                }
            ],
        }
        result = action_check_node(state)
        assert result["intent_results"][0]["status"] == "fulfilled"

    def test_non_fulfilled_skipped(self):
        state = {
            "tenant_id": "megamart",
            "intent_results": [
                {
                    "intent": "refund",
                    "status": "awaiting_human",
                    "domain_result": "",
                    "reason_code": "ESCALATED",
                    "tool_calls": [
                        {
                            "tool": "initiate_refund",
                            "args": {"amount": 999999},
                        }
                    ],
                }
            ],
        }
        result = action_check_node(state)
        assert result["intent_results"][0]["status"] == "awaiting_human"

    def test_multiple_intents_one_violation(self):
        state = {
            "tenant_id": "megamart",
            "intent_results": [
                {
                    "intent": "order_status",
                    "status": "fulfilled",
                    "domain_result": "Order info.",
                    "reason_code": "ORDER_STATUS_ALLOWED",
                    "tool_calls": [],
                },
                {
                    "intent": "refund",
                    "status": "fulfilled",
                    "domain_result": "Refund.",
                    "reason_code": "REFUND_ALLOWED",
                    "tool_calls": [
                        {
                            "tool": "initiate_refund",
                            "args": {"order_id": "ORD-MM-1001", "amount": 99999},
                        }
                    ],
                },
            ],
        }
        result = action_check_node(state)
        assert result["intent_results"][0]["status"] == "fulfilled"
        assert result["intent_results"][1]["status"] == "blocked"
