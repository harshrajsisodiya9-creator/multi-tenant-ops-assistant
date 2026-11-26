"""Tests for escalation_handler_node.

Covers: no pending escalations, approve with pending_tool_call,
approve without pending_tool_call (manual override), deny decision,
override decision, stale approval, tool execution failure,
index out of bounds, mixed decisions, revalidation logic.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.graph.nodes.escalation_handler import (
    _revalidate_pending,
    escalation_handler_node,
)

from .conftest import CUSTOMER_ALICE, TENANT_MEGAMART, make_base_state


def _make_escalation_state(intent_results: list[dict]) -> dict:
    state = make_base_state()
    state["intent_results"] = intent_results
    return state


# ===================================================================
# _revalidate_pending
# ===================================================================

class TestRevalidatePending:
    @patch("app.graph.nodes.escalation_handler._invoke_read_tool")
    def test_cancel_still_valid(self, mock_read):
        mock_read.return_value = {"order_status": "pending"}
        result = _revalidate_pending(
            {"tool": "cancel_order", "args": {"order_id": "ORD-1"}},
            TENANT_MEGAMART, CUSTOMER_ALICE,
        )
        assert result is None

    @patch("app.graph.nodes.escalation_handler._invoke_read_tool")
    def test_cancel_no_longer_valid(self, mock_read):
        mock_read.return_value = {"order_status": "shipped"}
        result = _revalidate_pending(
            {"tool": "cancel_order", "args": {"order_id": "ORD-1"}},
            TENANT_MEGAMART, CUSTOMER_ALICE,
        )
        assert result is not None
        assert "shipped" in result

    @patch("app.graph.nodes.escalation_handler._invoke_read_tool")
    def test_refund_amount_changed(self, mock_read):
        mock_read.side_effect = [
            {"order_id": "ORD-1", "total_amount": 2000},
            {"eligible": True, "amount": 2000},
        ]
        result = _revalidate_pending(
            {"tool": "initiate_refund", "args": {"order_id": "ORD-1", "amount": 1000}},
            TENANT_MEGAMART, CUSTOMER_ALICE,
        )
        assert result is not None
        assert "changed" in result

    @patch("app.graph.nodes.escalation_handler._invoke_read_tool")
    def test_order_error_on_revalidation(self, mock_read):
        mock_read.return_value = {"error": "Order not found"}
        result = _revalidate_pending(
            {"tool": "cancel_order", "args": {"order_id": "ORD-FAKE"}},
            TENANT_MEGAMART, CUSTOMER_ALICE,
        )
        assert result is not None
        assert "accessible" in result.lower()

    def test_no_order_id_skips_revalidation(self):
        result = _revalidate_pending(
            {"tool": "cancel_order", "args": {}},
            TENANT_MEGAMART, CUSTOMER_ALICE,
        )
        assert result is None


# ===================================================================
# escalation_handler_node
# ===================================================================

class TestEscalationHandlerNoEscalations:
    def test_no_pending_escalations(self):
        state = _make_escalation_state([
            {"intent": "order_status", "status": "fulfilled", "domain_result": "Done."},
        ])
        result = escalation_handler_node(state)
        assert result == {}

    def test_empty_intent_results(self):
        state = _make_escalation_state([])
        result = escalation_handler_node(state)
        assert result == {}


class TestEscalationHandlerApproval:
    @patch("app.graph.nodes.escalation_handler.interrupt")
    @patch("app.graph.nodes.escalation_handler._invoke_write_tool")
    @patch("app.graph.nodes.escalation_handler._invoke_read_tool")
    def test_approve_with_pending_tool_call_success(self, mock_read, mock_write, mock_interrupt):
        mock_interrupt.return_value = {"decisions": [
            {"index": 0, "action": "approve"},
        ]}
        mock_read.return_value = {"order_status": "pending"}
        mock_write.return_value = {"success": True, "message": "Order cancelled."}

        intent_results = [
            {
                "intent": "cancel_order",
                "status": "awaiting_human",
                "reason_code": "REQUIRES_APPROVAL",
                "domain_result": "Needs approval.",
                "entities": {"order_id": "ORD-1"},
                "pending_tool_call": {"tool": "cancel_order", "args": {"order_id": "ORD-1"}, "reasons": ["REQUIRES_APPROVAL"]},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)

        ir = result["intent_results"][0]
        assert ir["status"] == "fulfilled"
        assert ir.get("_newly_resolved") is True
        assert result["response_phase"] == "follow_up"

        mock_read.assert_called_once_with(
            "get_order_details", TENANT_MEGAMART, CUSTOMER_ALICE, order_id="ORD-1",
        )
        mock_write.assert_called_once_with(
            "cancel_order", TENANT_MEGAMART, CUSTOMER_ALICE, order_id="ORD-1",
        )
        assert len(ir["tool_calls"]) == 1
        assert ir["tool_calls"][0]["tool"] == "cancel_order"
        assert ir["tool_calls"][0]["result"]["success"] is True

    @patch("app.graph.nodes.escalation_handler.interrupt")
    @patch("app.graph.nodes.escalation_handler._invoke_write_tool")
    @patch("app.graph.nodes.escalation_handler._invoke_read_tool")
    def test_approve_with_tool_failure(self, mock_read, mock_write, mock_interrupt):
        mock_interrupt.return_value = {"decisions": [
            {"index": 0, "action": "approve"},
        ]}
        mock_read.return_value = {"order_status": "pending"}
        mock_write.return_value = {"success": False, "error": "DB error"}

        intent_results = [
            {
                "intent": "cancel_order",
                "status": "awaiting_human",
                "reason_code": "REQUIRES_APPROVAL",
                "domain_result": "Needs approval.",
                "entities": {"order_id": "ORD-1"},
                "pending_tool_call": {"tool": "cancel_order", "args": {"order_id": "ORD-1"}, "reasons": ["REQUIRES_APPROVAL"]},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)

        ir = result["intent_results"][0]
        assert ir["status"] == "failed"

    @patch("app.graph.nodes.escalation_handler.interrupt")
    def test_approve_without_pending_tool_call(self, mock_interrupt):
        mock_interrupt.return_value = {"decisions": [
            {"index": 0, "action": "approve", "note": "Looks good."},
        ]}
        intent_results = [
            {
                "intent": "complaint",
                "status": "awaiting_human",
                "reason_code": "ESCALATION_KEYWORDS_DETECTED",
                "domain_result": "Flagged for review.",
                "entities": {},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)

        ir = result["intent_results"][0]
        assert ir["status"] == "fulfilled"
        assert "Approved by support" in ir["domain_result"]
        assert "Looks good" in ir["domain_result"]


class TestEscalationHandlerDeny:
    @patch("app.graph.nodes.escalation_handler.interrupt")
    def test_deny_decision(self, mock_interrupt):
        mock_interrupt.return_value = {"decisions": [
            {"index": 0, "action": "deny", "reason": "Refund not warranted."},
        ]}
        intent_results = [
            {
                "intent": "refund",
                "status": "awaiting_human",
                "reason_code": "GLOBAL_REFUND_CEILING",
                "domain_result": "Exceeds limit.",
                "entities": {"order_id": "ORD-1"},
                "pending_tool_call": {"tool": "initiate_refund", "args": {"order_id": "ORD-1", "amount": 60000}, "reasons": ["GLOBAL_REFUND_CEILING"]},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)

        ir = result["intent_results"][0]
        assert ir["status"] == "blocked"
        assert ir["reason_code"] == "HUMAN_DENIED"
        assert ir["domain_result"] == "Refund not warranted."
        assert ir["_newly_resolved"] is True

    @patch("app.graph.nodes.escalation_handler.interrupt")
    def test_deny_without_reason(self, mock_interrupt):
        mock_interrupt.return_value = {"decisions": [
            {"index": 0, "action": "deny"},
        ]}
        intent_results = [
            {
                "intent": "refund",
                "status": "awaiting_human",
                "reason_code": "X",
                "domain_result": "Needs review.",
                "entities": {},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)
        ir = result["intent_results"][0]
        assert ir["status"] == "blocked"
        assert "cannot be processed" in ir["domain_result"].lower()


class TestEscalationHandlerOverride:
    @patch("app.graph.nodes.escalation_handler.interrupt")
    def test_override_with_custom_response(self, mock_interrupt):
        mock_interrupt.return_value = {"decisions": [
            {"index": 0, "action": "override", "custom_response": "Special exception granted."},
        ]}
        intent_results = [
            {
                "intent": "refund",
                "status": "awaiting_human",
                "reason_code": "X",
                "domain_result": "Needs review.",
                "entities": {},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)
        ir = result["intent_results"][0]
        assert ir["status"] == "fulfilled"
        assert ir["domain_result"] == "Special exception granted."


class TestEscalationHandlerStaleApproval:
    @patch("app.graph.nodes.escalation_handler.interrupt")
    @patch("app.graph.nodes.escalation_handler._revalidate_pending")
    def test_stale_approval_fails(self, mock_revalidate, mock_interrupt):
        mock_interrupt.return_value = {"decisions": [
            {"index": 0, "action": "approve"},
        ]}
        mock_revalidate.return_value = "Order can no longer be cancelled."

        intent_results = [
            {
                "intent": "cancel_order",
                "status": "awaiting_human",
                "reason_code": "REQUIRES_APPROVAL",
                "domain_result": "Needs approval.",
                "entities": {"order_id": "ORD-1"},
                "pending_tool_call": {"tool": "cancel_order", "args": {"order_id": "ORD-1"}, "reasons": ["REQUIRES_APPROVAL"]},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)

        ir = result["intent_results"][0]
        assert ir["status"] == "failed"
        assert ir["reason_code"] == "STALE_APPROVAL"
        assert ir["_newly_resolved"] is True
        assert "can no longer be processed" in ir["domain_result"]

    @patch("app.graph.nodes.escalation_handler.interrupt")
    @patch("app.graph.nodes.escalation_handler._invoke_write_tool")
    @patch("app.graph.nodes.escalation_handler._invoke_read_tool")
    def test_stale_refund_amount_changed_blocks_write(self, mock_read, mock_write, mock_interrupt):
        """When refund amount changes between approval request and resume, write must NOT execute."""
        mock_interrupt.return_value = {"decisions": [
            {"index": 0, "action": "approve"},
        ]}
        mock_read.side_effect = [
            {"order_id": "ORD-1", "total_amount": 75000},
            {"eligible": True, "amount": 75000},
        ]

        intent_results = [
            {
                "intent": "refund",
                "status": "awaiting_human",
                "reason_code": "GLOBAL_REFUND_CEILING",
                "domain_result": "Exceeds limit.",
                "entities": {"order_id": "ORD-1"},
                "pending_tool_call": {
                    "tool": "initiate_refund",
                    "args": {"order_id": "ORD-1", "amount": 60000},
                    "reasons": ["GLOBAL_REFUND_CEILING"],
                },
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)

        mock_write.assert_not_called()
        ir = result["intent_results"][0]
        assert ir["status"] == "failed"
        assert ir["reason_code"] == "STALE_APPROVAL"
        assert "changed" in ir["domain_result"]


class TestEscalationHandlerEdgeCases:
    @patch("app.graph.nodes.escalation_handler.interrupt")
    def test_index_out_of_bounds_skipped(self, mock_interrupt):
        mock_interrupt.return_value = {"decisions": [
            {"index": 99, "action": "approve"},
        ]}
        intent_results = [
            {
                "intent": "refund",
                "status": "awaiting_human",
                "reason_code": "X",
                "domain_result": "Needs review.",
                "entities": {},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)
        assert result["intent_results"][0]["status"] == "awaiting_human"

    @patch("app.graph.nodes.escalation_handler.interrupt")
    def test_missing_index_skipped(self, mock_interrupt):
        mock_interrupt.return_value = {"decisions": [
            {"action": "approve"},
        ]}
        intent_results = [
            {
                "intent": "refund",
                "status": "awaiting_human",
                "reason_code": "X",
                "domain_result": "Needs review.",
                "entities": {},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)
        assert result["intent_results"][0]["status"] == "awaiting_human"

    @patch("app.graph.nodes.escalation_handler.interrupt")
    def test_negative_index_rejected(self, mock_interrupt):
        """Regression: negative index -1 must NOT select the last result via Python list[-1]."""
        mock_interrupt.return_value = {"decisions": [
            {"index": -1, "action": "approve"},
        ]}
        intent_results = [
            {
                "intent": "refund",
                "status": "awaiting_human",
                "reason_code": "GLOBAL_REFUND_CEILING",
                "domain_result": "Exceeds limit.",
                "entities": {"order_id": "ORD-1"},
                "pending_tool_call": {"tool": "initiate_refund", "args": {"order_id": "ORD-1", "amount": 60000}, "reasons": ["GLOBAL_REFUND_CEILING"]},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)
        assert result["intent_results"][0]["status"] == "awaiting_human"

    @patch("app.graph.nodes.escalation_handler.interrupt")
    def test_negative_index_minus_two_rejected(self, mock_interrupt):
        """Regression: index -2 on a 2-item list must NOT select index 0."""
        mock_interrupt.return_value = {"decisions": [
            {"index": -2, "action": "approve"},
        ]}
        intent_results = [
            {
                "intent": "cancel_order",
                "status": "awaiting_human",
                "reason_code": "X",
                "domain_result": "Review.",
                "entities": {"order_id": "ORD-1"},
                "pending_tool_call": {"tool": "cancel_order", "args": {"order_id": "ORD-1"}, "reasons": ["X"]},
            },
            {
                "intent": "refund",
                "status": "awaiting_human",
                "reason_code": "Y",
                "domain_result": "Review.",
                "entities": {},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)
        assert result["intent_results"][0]["status"] == "awaiting_human"
        assert result["intent_results"][1]["status"] == "awaiting_human"

    @patch("app.graph.nodes.escalation_handler.interrupt")
    def test_float_index_rejected(self, mock_interrupt):
        """Non-int index (float 0.0) is safely rejected."""
        mock_interrupt.return_value = {"decisions": [
            {"index": 0.0, "action": "approve"},
        ]}
        intent_results = [
            {
                "intent": "refund",
                "status": "awaiting_human",
                "reason_code": "X",
                "domain_result": "Review.",
                "entities": {},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)
        assert result["intent_results"][0]["status"] == "awaiting_human"

    @patch("app.graph.nodes.escalation_handler.interrupt")
    def test_string_index_rejected(self, mock_interrupt):
        """String index is safely rejected."""
        mock_interrupt.return_value = {"decisions": [
            {"index": "0", "action": "approve"},
        ]}
        intent_results = [
            {
                "intent": "refund",
                "status": "awaiting_human",
                "reason_code": "X",
                "domain_result": "Review.",
                "entities": {},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)
        assert result["intent_results"][0]["status"] == "awaiting_human"

    @staticmethod
    def _two_pending_with_tool_calls() -> list[dict]:
        return [
            {
                "intent": "cancel_order",
                "status": "awaiting_human",
                "reason_code": "X",
                "domain_result": "Review cancel.",
                "entities": {"order_id": "ORD-1"},
                "pending_tool_call": {"tool": "cancel_order", "args": {"order_id": "ORD-1"}, "reasons": ["X"]},
            },
            {
                "intent": "refund",
                "status": "awaiting_human",
                "reason_code": "GLOBAL_REFUND_CEILING",
                "domain_result": "Review refund.",
                "entities": {"order_id": "ORD-2"},
                "pending_tool_call": {"tool": "initiate_refund", "args": {"order_id": "ORD-2", "amount": 60000}, "reasons": ["GLOBAL_REFUND_CEILING"]},
            },
        ]

    @patch("app.graph.nodes.escalation_handler.interrupt")
    @patch("app.graph.nodes.escalation_handler._invoke_write_tool")
    @patch("app.graph.nodes.escalation_handler._invoke_read_tool")
    def test_bool_true_index_rejected(self, mock_read, mock_write, mock_interrupt):
        """Regression: True is an int subclass (== 1) and must NOT select intent_results[1]."""
        mock_interrupt.return_value = {"decisions": [{"index": True, "action": "approve"}]}
        state = _make_escalation_state(self._two_pending_with_tool_calls())
        result = escalation_handler_node(state)

        mock_read.assert_not_called()
        mock_write.assert_not_called()
        for ir in result["intent_results"]:
            assert ir["status"] == "awaiting_human"
            assert "_newly_resolved" not in ir

    @patch("app.graph.nodes.escalation_handler.interrupt")
    @patch("app.graph.nodes.escalation_handler._invoke_write_tool")
    @patch("app.graph.nodes.escalation_handler._invoke_read_tool")
    def test_bool_false_index_rejected(self, mock_read, mock_write, mock_interrupt):
        """Regression: False is an int subclass (== 0) and must NOT select intent_results[0]."""
        mock_interrupt.return_value = {"decisions": [{"index": False, "action": "approve"}]}
        state = _make_escalation_state(self._two_pending_with_tool_calls())
        result = escalation_handler_node(state)

        mock_read.assert_not_called()
        mock_write.assert_not_called()
        for ir in result["intent_results"]:
            assert ir["status"] == "awaiting_human"
            assert "_newly_resolved" not in ir

    @patch("app.graph.nodes.escalation_handler.interrupt")
    def test_list_format_decisions(self, mock_interrupt):
        """interrupt can return a list directly instead of {"decisions": [...]}."""
        mock_interrupt.return_value = [
            {"index": 0, "action": "deny", "reason": "No."},
        ]
        intent_results = [
            {
                "intent": "refund",
                "status": "awaiting_human",
                "reason_code": "X",
                "domain_result": "Review.",
                "entities": {},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)
        assert result["intent_results"][0]["status"] == "blocked"

    @patch("app.graph.nodes.escalation_handler.interrupt")
    @patch("app.graph.nodes.escalation_handler._invoke_write_tool")
    @patch("app.graph.nodes.escalation_handler._invoke_read_tool")
    def test_mixed_decisions(self, mock_read, mock_write, mock_interrupt):
        mock_interrupt.return_value = {"decisions": [
            {"index": 0, "action": "approve"},
            {"index": 1, "action": "deny", "reason": "Not warranted."},
        ]}
        mock_read.return_value = {"order_status": "pending"}
        mock_write.return_value = {"success": True, "message": "Done."}

        intent_results = [
            {
                "intent": "cancel_order",
                "status": "awaiting_human",
                "reason_code": "X",
                "domain_result": "Review cancel.",
                "entities": {"order_id": "ORD-1"},
                "pending_tool_call": {"tool": "cancel_order", "args": {"order_id": "ORD-1"}, "reasons": ["X"]},
            },
            {
                "intent": "refund",
                "status": "awaiting_human",
                "reason_code": "Y",
                "domain_result": "Review refund.",
                "entities": {"order_id": "ORD-2"},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)
        assert result["intent_results"][0]["status"] == "fulfilled"
        assert result["intent_results"][0]["_newly_resolved"] is True
        assert result["intent_results"][0]["tool_calls"][-1]["tool"] == "cancel_order"
        assert result["intent_results"][0]["tool_calls"][-1]["result"]["success"] is True

        assert result["intent_results"][1]["status"] == "blocked"
        assert result["intent_results"][1]["reason_code"] == "HUMAN_DENIED"
        assert result["intent_results"][1]["domain_result"] == "Not warranted."
        assert result["intent_results"][1]["_newly_resolved"] is True

        mock_write.assert_called_once()
        mock_read.assert_called_once()
