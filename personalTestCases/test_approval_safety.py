"""Cross-cutting approval safety tests.

Exercises edge cases around the approval pipeline that span
write_executor → escalation_handler → action_check:
  - Conditional action that also needs approval
  - Approval of one intent doesn't leak to another
  - Revalidation failures for every cancellable status
  - Tool execution exception during approved write
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.graph.nodes.escalation_handler import (
    _revalidate_pending,
    escalation_handler_node,
)
from app.graph.nodes.write_executor import (
    _execute_refund_workflow,
    _needs_approval,
)
from app.graph.policy_gate import GLOBAL_REFUND_CEILING

from .conftest import CUSTOMER_ALICE, TENANT_MEGAMART, make_base_state


def _make_escalation_state(intent_results: list[dict]) -> dict:
    state = make_base_state()
    state["intent_results"] = intent_results
    return state


# ===================================================================
# Conditional + approval interaction
# ===================================================================

class TestConditionalNeedsApproval:
    @patch("app.graph.nodes.write_executor._invoke_write_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_refund_above_ceiling_with_conditional_still_needs_approval(self, mock_read, mock_write):
        """Even when a write comes through the conditional path, the ceiling check
        must still trigger approval after the condition is revalidated."""
        mock_read.side_effect = [
            {"order_id": "ORD-1", "total_amount": GLOBAL_REFUND_CEILING + 1},
            {"eligible": True, "amount": GLOBAL_REFUND_CEILING + 1, "refund_type": "original_method"},
            {"order_status": "pending"},
        ]
        result = _execute_refund_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-1"},
            {"refund_type": "original_method"},
            pre_condition={
                "condition": {
                    "field": "order_status",
                    "operator": "not_equals",
                    "value": "shipped",
                },
                "evaluation": {"status": "satisfied"},
            },
        )
        assert result["status"] == "awaiting_human"
        assert result["pending_tool_call"]["tool"] == "initiate_refund"
        assert result["pending_tool_call"]["args"]["amount"] == GLOBAL_REFUND_CEILING + 1
        assert [call.args[0] for call in mock_read.call_args_list] == [
            "get_order_details", "get_return_eligibility", "get_order_details",
        ]
        mock_write.assert_not_called()


# ===================================================================
# Approval isolation between intents
# ===================================================================

class TestApprovalIsolation:
    @patch("app.graph.nodes.escalation_handler.interrupt")
    @patch("app.graph.nodes.escalation_handler._invoke_write_tool")
    @patch("app.graph.nodes.escalation_handler._invoke_read_tool")
    def test_approving_one_does_not_resolve_another(self, mock_read, mock_write, mock_interrupt):
        """Approving index 0 must leave index 1 as awaiting_human if no decision targets it."""
        mock_interrupt.return_value = {"decisions": [
            {"index": 0, "action": "approve"},
        ]}
        mock_read.return_value = {"order_status": "pending"}
        mock_write.return_value = {"success": True, "message": "Cancelled."}

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
                "reason_code": "GLOBAL_REFUND_CEILING",
                "domain_result": "Exceeds limit.",
                "entities": {"order_id": "ORD-2"},
                "pending_tool_call": {
                    "tool": "initiate_refund",
                    "args": {"order_id": "ORD-2", "amount": 60000},
                    "reasons": ["GLOBAL_REFUND_CEILING"],
                },
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)

        assert result["intent_results"][0]["status"] == "fulfilled"
        assert result["intent_results"][0]["_newly_resolved"] is True

        assert result["intent_results"][1]["status"] == "awaiting_human"
        assert "_newly_resolved" not in result["intent_results"][1]

        mock_write.assert_called_once()


# ===================================================================
# Revalidation across all cancellable statuses
# ===================================================================

class TestRevalidationCancellableStatuses:
    """Verify _revalidate_pending correctly gates cancellation for every status."""

    @pytest.mark.parametrize("status", ["pending", "processing", "confirmed"])
    @patch("app.graph.nodes.escalation_handler._invoke_read_tool")
    def test_cancellable_statuses_pass(self, mock_read, status):
        mock_read.return_value = {"order_status": status}
        result = _revalidate_pending(
            {"tool": "cancel_order", "args": {"order_id": "ORD-1"}},
            TENANT_MEGAMART, CUSTOMER_ALICE,
        )
        assert result is None

    @pytest.mark.parametrize("status", ["shipped", "delivered", "returned", "cancelled"])
    @patch("app.graph.nodes.escalation_handler._invoke_read_tool")
    def test_non_cancellable_statuses_blocked(self, mock_read, status):
        mock_read.return_value = {"order_status": status}
        result = _revalidate_pending(
            {"tool": "cancel_order", "args": {"order_id": "ORD-1"}},
            TENANT_MEGAMART, CUSTOMER_ALICE,
        )
        assert result is not None
        assert status in result

    @patch("app.graph.nodes.escalation_handler._invoke_read_tool")
    def test_refund_no_longer_eligible_blocked(self, mock_read):
        mock_read.side_effect = [
            {"order_id": "ORD-1", "total_amount": 1000},
            {"eligible": False, "reason": "Return window expired"},
        ]
        result = _revalidate_pending(
            {"tool": "initiate_refund", "args": {"order_id": "ORD-1", "amount": 1000}},
            TENANT_MEGAMART, CUSTOMER_ALICE,
        )
        assert result is not None
        assert "expired" in result.lower() or "eligible" in result.lower()


# ===================================================================
# Tool execution exception during approved write
# ===================================================================

class TestApprovedWriteException:
    @patch("app.graph.nodes.escalation_handler.interrupt")
    @patch("app.graph.nodes.escalation_handler._invoke_write_tool")
    @patch("app.graph.nodes.escalation_handler._invoke_read_tool")
    def test_write_exception_produces_failed(self, mock_read, mock_write, mock_interrupt):
        """If _execute_pending_tool raises, the intent goes to 'failed'."""
        mock_interrupt.return_value = {"decisions": [
            {"index": 0, "action": "approve"},
        ]}
        mock_read.return_value = {"order_status": "pending"}
        mock_write.side_effect = RuntimeError("DB connection lost")

        intent_results = [
            {
                "intent": "cancel_order",
                "status": "awaiting_human",
                "reason_code": "X",
                "domain_result": "Review.",
                "entities": {"order_id": "ORD-1"},
                "pending_tool_call": {"tool": "cancel_order", "args": {"order_id": "ORD-1"}, "reasons": ["X"]},
            },
        ]
        state = _make_escalation_state(intent_results)
        result = escalation_handler_node(state)

        ir = result["intent_results"][0]
        assert ir["status"] == "failed"
        assert "execution failed" in ir["domain_result"].lower() or "failed" in ir["domain_result"].lower()


# ===================================================================
# _needs_approval boundary
# ===================================================================

class TestNeedsApprovalBoundary:
    def test_refund_exactly_at_ceiling_no_approval(self):
        """Refund at exactly the ceiling should NOT need approval (uses > not >=)."""
        result = _needs_approval(
            "initiate_refund",
            {"amount": GLOBAL_REFUND_CEILING},
            {},
        )
        assert result is None
