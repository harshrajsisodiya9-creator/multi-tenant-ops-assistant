from __future__ import annotations

import pytest

from app.graph.nodes.action_check import action_check_node


class TestAuditFlags:
    def test_near_ceiling_flagged(self):
        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "refund",
                    "status": "fulfilled",
                    "domain_result": "Refund processed.",
                    "entities": {"order_id": "ORD-MM-1004"},
                    "tool_calls": [
                        {
                            "tool": "initiate_refund",
                            "args": {"order_id": "ORD-MM-1004", "amount": 45000},
                        }
                    ],
                }
            ],
        }
        result = action_check_node(state)
        r = result["intent_results"][0]
        assert r["status"] == "fulfilled"
        assert any(f["flag"] == "NEAR_CEILING" for f in r.get("audit_flags", []))

    def test_below_threshold_no_flag(self):
        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "refund",
                    "status": "fulfilled",
                    "domain_result": "Refund processed.",
                    "entities": {"order_id": "ORD-MM-1001"},
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
        r = result["intent_results"][0]
        assert r["status"] == "fulfilled"
        assert "audit_flags" not in r

    def test_multi_write_flagged(self):
        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "refund",
                    "status": "fulfilled",
                    "domain_result": "Done.",
                    "entities": {"order_id": "ORD-MM-1001"},
                    "tool_calls": [
                        {
                            "tool": "initiate_refund",
                            "args": {"order_id": "ORD-MM-1001", "amount": 1000},
                        },
                        {
                            "tool": "cancel_order",
                            "args": {"order_id": "ORD-MM-1001"},
                        },
                    ],
                }
            ],
        }
        result = action_check_node(state)
        r = result["intent_results"][0]
        assert any(f["flag"] == "MULTI_WRITE" for f in r.get("audit_flags", []))

    def test_blocked_tool_calls_not_audited(self):
        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "refund",
                    "status": "fulfilled",
                    "domain_result": "Refund.",
                    "entities": {"order_id": "ORD-MM-1004"},
                    "tool_calls": [
                        {
                            "tool": "initiate_refund",
                            "args": {"order_id": "ORD-MM-1004", "amount": 55000},
                            "blocked": True,
                            "reason": "already blocked in-loop",
                        }
                    ],
                }
            ],
        }
        result = action_check_node(state)
        r = result["intent_results"][0]
        assert r["status"] == "fulfilled"
        assert "audit_flags" not in r


class TestPassthrough:
    def test_no_tool_calls_passes(self):
        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "faq",
                    "status": "fulfilled",
                    "domain_result": "FAQ answer.",
                    "tool_calls": [],
                }
            ],
        }
        result = action_check_node(state)
        assert result["intent_results"][0]["status"] == "fulfilled"

    def test_non_fulfilled_skipped(self):
        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "refund",
                    "status": "awaiting_human",
                    "domain_result": "",
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

    def test_cancel_no_flags_when_clean(self):
        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "cancel_order",
                    "status": "fulfilled",
                    "domain_result": "Order cancelled.",
                    "entities": {"order_id": "ORD-MM-1007"},
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
        r = result["intent_results"][0]
        assert r["status"] == "fulfilled"
        assert "audit_flags" not in r

    def test_multiple_intents_flags_only_on_flagged(self):
        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "order_status",
                    "status": "fulfilled",
                    "domain_result": "Order info.",
                    "tool_calls": [],
                },
                {
                    "intent": "refund",
                    "status": "fulfilled",
                    "domain_result": "Refund.",
                    "entities": {"order_id": "ORD-MM-1001"},
                    "tool_calls": [
                        {
                            "tool": "initiate_refund",
                            "args": {"order_id": "ORD-MM-1001", "amount": 45000},
                        }
                    ],
                },
            ],
        }
        result = action_check_node(state)
        assert result["intent_results"][0]["status"] == "fulfilled"
        assert "audit_flags" not in result["intent_results"][0]
        assert result["intent_results"][1]["status"] == "fulfilled"
        assert any(f["flag"] == "NEAR_CEILING" for f in result["intent_results"][1].get("audit_flags", []))


class TestVerifySkippedInMock:
    def test_no_verification_in_mock_mode(self):
        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "refund",
                    "status": "fulfilled",
                    "domain_result": "Refund processed.",
                    "entities": {"order_id": "ORD-MM-1001"},
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
        tc = result["intent_results"][0]["tool_calls"][0]
        assert "verification" not in tc
