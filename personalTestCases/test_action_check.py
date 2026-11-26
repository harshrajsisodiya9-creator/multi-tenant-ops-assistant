"""Tests for action_check_node.

Covers: fulfilled result with no tool_calls skipped, audit flags for
multi-write, near-ceiling, non-fulfilled status skipped, empty intent_results,
verification logic (prod only, so we test the skip for mock mode).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.graph.nodes.action_check import (
    NEAR_CEILING_THRESHOLD,
    _audit_tool_calls,
    action_check_node,
)
from app.graph.policy_gate import GLOBAL_REFUND_CEILING

from .conftest import CUSTOMER_ALICE, TENANT_MEGAMART, make_base_state


# ===================================================================
# _audit_tool_calls
# ===================================================================

class TestAuditToolCalls:
    def test_multi_write_flagged(self):
        tool_calls = [
            {"tool": "initiate_refund", "args": {"order_id": "ORD-1", "amount": 1000}, "result": {"success": True}},
            {"tool": "cancel_order", "args": {"order_id": "ORD-2"}, "result": {"success": True}},
        ]
        flags = _audit_tool_calls(tool_calls, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1")
        flag_types = [f["flag"] for f in flags]
        assert "MULTI_WRITE" in flag_types

    def test_near_ceiling_flagged(self):
        near_amount = int(GLOBAL_REFUND_CEILING * NEAR_CEILING_THRESHOLD) + 1
        tool_calls = [
            {"tool": "initiate_refund", "args": {"order_id": "ORD-1", "amount": near_amount}, "result": {"success": True}},
        ]
        flags = _audit_tool_calls(tool_calls, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1")
        flag_types = [f["flag"] for f in flags]
        assert "NEAR_CEILING" in flag_types

    def test_blocked_calls_skipped(self):
        tool_calls = [
            {"tool": "initiate_refund", "args": {"amount": 999999}, "blocked": True, "reason": "needs_approval"},
        ]
        flags = _audit_tool_calls(tool_calls, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1")
        flag_types = [f["flag"] for f in flags]
        assert "NEAR_CEILING" not in flag_types

    def test_no_flags_on_clean_calls(self):
        tool_calls = [
            {"tool": "get_order_details", "args": {"order_id": "ORD-1"}, "result": {"order_id": "ORD-1"}},
        ]
        flags = _audit_tool_calls(tool_calls, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1")
        assert flags == []

    def test_empty_tool_calls(self):
        flags = _audit_tool_calls([], TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1")
        assert flags == []


# ===================================================================
# action_check_node
# ===================================================================

class TestActionCheckNode:
    def test_non_fulfilled_skipped(self):
        state = make_base_state()
        state["intent_results"] = [
            {
                "intent": "refund",
                "status": "blocked",
                "domain_result": "Not allowed.",
                "tool_calls": [{"tool": "initiate_refund", "args": {"amount": 999999}, "result": {"success": True}}],
            },
        ]
        result = action_check_node(state)
        assert "audit_flags" not in result["intent_results"][0]

    def test_fulfilled_without_tool_calls_skipped(self):
        state = make_base_state()
        state["intent_results"] = [
            {
                "intent": "greeting",
                "status": "fulfilled",
                "domain_result": "Hello!",
                "tool_calls": [],
            },
        ]
        result = action_check_node(state)
        assert "audit_flags" not in result["intent_results"][0]

    def test_audit_flags_added(self):
        state = make_base_state()
        near_amount = int(GLOBAL_REFUND_CEILING * NEAR_CEILING_THRESHOLD) + 1
        state["intent_results"] = [
            {
                "intent": "refund",
                "status": "fulfilled",
                "domain_result": "Refund done.",
                "entities": {"order_id": "ORD-1"},
                "tool_calls": [
                    {"tool": "initiate_refund", "args": {"order_id": "ORD-1", "amount": near_amount}, "result": {"success": True}},
                ],
            },
        ]
        result = action_check_node(state)
        assert "audit_flags" in result["intent_results"][0]
        flag_types = [f["flag"] for f in result["intent_results"][0]["audit_flags"]]
        assert "NEAR_CEILING" in flag_types

    def test_empty_intent_results(self):
        state = make_base_state()
        state["intent_results"] = []
        result = action_check_node(state)
        assert result["intent_results"] == []

    def test_missing_intent_results(self):
        state = make_base_state()
        result = action_check_node(state)
        assert result["intent_results"] == []

    def test_verification_skipped_in_mock_mode(self):
        """In mock mode, verification is skipped (is_prod=False)."""
        state = make_base_state()
        state["intent_results"] = [
            {
                "intent": "refund",
                "status": "fulfilled",
                "domain_result": "Refund done.",
                "entities": {"order_id": "ORD-1"},
                "tool_calls": [
                    {"tool": "initiate_refund", "args": {"order_id": "ORD-1", "amount": 1000}, "result": {"success": True}},
                ],
            },
        ]
        result = action_check_node(state)
        tc = result["intent_results"][0]["tool_calls"][0]
        assert "verification" not in tc

    def test_exception_does_not_crash_node(self):
        """If audit raises, the node logs but continues."""
        state = make_base_state()
        state["intent_results"] = [
            {
                "intent": "refund",
                "status": "fulfilled",
                "domain_result": "Done.",
                "entities": {"order_id": "ORD-1"},
                "tool_calls": [{"tool": "initiate_refund", "args": {"order_id": "ORD-1", "amount": 1000}, "result": {"success": True}}],
            },
        ]
        with patch("app.graph.nodes.action_check._audit_tool_calls", side_effect=RuntimeError("boom")):
            result = action_check_node(state)
        assert result["intent_results"][0]["status"] == "fulfilled"
