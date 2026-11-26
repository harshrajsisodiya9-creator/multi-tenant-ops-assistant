"""Tests for write_executor_node and its deterministic workflows.

Covers: refund workflow (happy path, ineligible, constraint violation,
approval required, tool failure), cancel workflow (happy path, constraint block,
non-cancellable), non-write intent rejection, missing workflow, exception handling.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.graph.nodes.write_executor import (
    _execute_cancel_workflow,
    _execute_refund_workflow,
    _hard_constraint_check,
    _invoke_tool,
    _invoke_write_tool,
    _needs_approval,
    write_executor_node,
)
from app.graph.policy_gate import GLOBAL_REFUND_CEILING

from .conftest import CUSTOMER_ALICE, TENANT_MEGAMART, make_base_state, make_intent_result


# ===================================================================
# _hard_constraint_check
# ===================================================================

class TestHardConstraintCheck:
    def test_refund_type_mismatch(self):
        result = _hard_constraint_check(
            "initiate_refund",
            {"refund_type": "store_credit"},
            {"refund_type": "original_method"},
        )
        assert result is not None
        assert "original_method" in result

    def test_refund_type_match(self):
        result = _hard_constraint_check(
            "initiate_refund",
            {"refund_type": "original_method"},
            {"refund_type": "original_method"},
        )
        assert result is None

    def test_cancel_blocked(self):
        result = _hard_constraint_check(
            "cancel_order",
            {"order_id": "ORD-1"},
            {"cancel_blocked": True},
        )
        assert result is not None

    def test_cancel_not_blocked(self):
        result = _hard_constraint_check(
            "cancel_order",
            {"order_id": "ORD-1"},
            {},
        )
        assert result is None

    def test_unknown_tool_no_constraint(self):
        result = _hard_constraint_check(
            "get_order_details",
            {},
            {},
        )
        assert result is None


# ===================================================================
# _needs_approval
# ===================================================================

class TestNeedsApproval:
    def test_explicit_needs_approval(self):
        result = _needs_approval(
            "initiate_refund",
            {"amount": 100},
            {"needs_approval": True, "approval_reasons": ["TENANT_REFUND_REVIEW_REQUIRED"]},
        )
        assert result is not None
        assert "TENANT_REFUND_REVIEW_REQUIRED" in result

    def test_refund_over_global_ceiling(self):
        result = _needs_approval(
            "initiate_refund",
            {"amount": GLOBAL_REFUND_CEILING + 1},
            {},
        )
        assert result is not None
        assert "GLOBAL_REFUND_CEILING" in result

    def test_refund_under_ceiling_no_approval(self):
        result = _needs_approval(
            "initiate_refund",
            {"amount": GLOBAL_REFUND_CEILING - 1},
            {},
        )
        assert result is None

    def test_non_refund_no_approval(self):
        result = _needs_approval("cancel_order", {"order_id": "ORD-1"}, {})
        assert result is None


# ===================================================================
# Refund workflow
# ===================================================================

class TestRefundWorkflow:
    @patch("app.graph.nodes.write_executor._invoke_write_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_happy_path(self, mock_read, mock_write):
        mock_read.side_effect = [
            {"order_id": "ORD-1", "total_amount": 1000},
            {"eligible": True, "amount": 1000, "refund_type": "original_method"},
        ]
        mock_write.return_value = {"success": True, "message": "Refund processed."}

        result = _execute_refund_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-1"},
            {"refund_type": "original_method"},
        )
        assert result["status"] == "fulfilled"
        assert result["domain_result"] == "Refund processed."

        read_calls = mock_read.call_args_list
        assert read_calls[0][0] == ("get_order_details", TENANT_MEGAMART, CUSTOMER_ALICE)
        assert read_calls[0][1] == {"order_id": "ORD-1"}
        assert read_calls[1][0] == ("get_return_eligibility", TENANT_MEGAMART, CUSTOMER_ALICE)

        mock_write.assert_called_once_with(
            "initiate_refund", TENANT_MEGAMART, CUSTOMER_ALICE,
            order_id="ORD-1", amount=1000, refund_type="original_method",
        )

        assert len(result["tool_calls"]) == 3
        assert result["tool_calls"][0]["tool"] == "get_order_details"
        assert result["tool_calls"][1]["tool"] == "get_return_eligibility"
        assert result["tool_calls"][2]["tool"] == "initiate_refund"
        assert result["tool_calls"][2]["result"]["success"] is True

    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_order_error(self, mock_read):
        mock_read.return_value = {"error": "Order not found"}
        result = _execute_refund_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-FAKE"}, {},
        )
        assert result["status"] == "failed"
        assert "not found" in result["domain_result"].lower() or "Could not" in result["domain_result"]

    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_not_eligible(self, mock_read):
        mock_read.side_effect = [
            {"order_id": "ORD-1", "total_amount": 1000},
            {"eligible": False, "reason": "Return window expired"},
        ]
        result = _execute_refund_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-1"}, {},
        )
        assert result["status"] == "blocked"
        assert "expired" in result["domain_result"].lower() or "Not eligible" in result["domain_result"]

    @patch("app.graph.nodes.write_executor._invoke_write_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_constraints_refund_type_overrides_eligibility(self, mock_read, mock_write):
        """Policy constraint refund_type takes precedence over eligibility default."""
        mock_read.side_effect = [
            {"order_id": "ORD-1", "total_amount": 1000},
            {"eligible": True, "amount": 1000, "refund_type": "original_method"},
        ]
        mock_write.return_value = {"success": True, "message": "Refunded as store credit."}
        result = _execute_refund_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-1"},
            {"refund_type": "store_credit"},
        )
        assert result["status"] == "fulfilled"
        mock_write.assert_called_once_with(
            "initiate_refund", TENANT_MEGAMART, CUSTOMER_ALICE,
            order_id="ORD-1", amount=1000, refund_type="store_credit",
        )

    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_needs_approval(self, mock_read):
        mock_read.side_effect = [
            {"order_id": "ORD-1", "total_amount": 60000},
            {"eligible": True, "amount": 60000, "refund_type": "original_method"},
        ]
        result = _execute_refund_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-1"},
            {"refund_type": "original_method"},
        )
        assert result["status"] == "awaiting_human"
        assert result.get("pending_tool_call") is not None

    @patch("app.graph.nodes.write_executor._invoke_write_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_write_failure(self, mock_read, mock_write):
        mock_read.side_effect = [
            {"order_id": "ORD-1", "total_amount": 1000},
            {"eligible": True, "amount": 1000, "refund_type": "original_method"},
        ]
        mock_write.return_value = {"success": False, "error": "DB error"}

        result = _execute_refund_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-1"},
            {"refund_type": "original_method"},
        )
        assert result["status"] == "failed"
        assert result["domain_result"] == "DB error"
        assert len(result["tool_calls"]) == 3
        assert result["tool_calls"][2]["result"]["success"] is False


# ===================================================================
# Cancel workflow
# ===================================================================

class TestCancelWorkflow:
    @patch("app.graph.nodes.write_executor._invoke_write_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_happy_path(self, mock_read, mock_write):
        mock_read.return_value = {"order_id": "ORD-1", "order_status": "pending"}
        mock_write.return_value = {"success": True, "message": "Order cancelled."}

        result = _execute_cancel_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-1"}, {},
        )
        assert result["status"] == "fulfilled"
        assert result["domain_result"] == "Order cancelled."

        mock_read.assert_called_once_with(
            "get_order_details", TENANT_MEGAMART, CUSTOMER_ALICE, order_id="ORD-1",
        )
        mock_write.assert_called_once_with(
            "cancel_order", TENANT_MEGAMART, CUSTOMER_ALICE, order_id="ORD-1",
        )
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["tool"] == "get_order_details"
        assert result["tool_calls"][1]["tool"] == "cancel_order"
        assert result["tool_calls"][1]["result"]["success"] is True

    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_order_error(self, mock_read):
        mock_read.return_value = {"error": "Order not found"}
        result = _execute_cancel_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-FAKE"}, {},
        )
        assert result["status"] == "failed"

    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_cancel_blocked_by_constraint(self, mock_read):
        mock_read.return_value = {"order_id": "ORD-1", "order_status": "pending"}
        result = _execute_cancel_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-1"},
            {"cancel_blocked": True},
        )
        assert result["status"] == "blocked"

    @patch("app.graph.nodes.write_executor._invoke_write_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_cancel_needs_approval(self, mock_read, mock_write):
        mock_read.return_value = {"order_id": "ORD-1", "order_status": "pending"}
        result = _execute_cancel_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-1"},
            {"needs_approval": True, "approval_reasons": ["HIGH_VALUE_ORDER"]},
        )
        assert result["status"] == "awaiting_human"
        assert result["pending_tool_call"]["tool"] == "cancel_order"
        assert result["reason_code_override"] == "HIGH_VALUE_ORDER"
        mock_write.assert_not_called()

    @patch("app.graph.nodes.write_executor._invoke_write_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_cancel_write_failure(self, mock_read, mock_write):
        mock_read.return_value = {"order_id": "ORD-1", "order_status": "pending"}
        mock_write.return_value = {"success": False, "error": "DB error"}
        result = _execute_cancel_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-1"}, {},
        )
        assert result["status"] == "failed"
        assert result["domain_result"] == "DB error"


# ===================================================================
# write_executor_node
# ===================================================================

class TestWriteExecutorNode:
    @patch("app.graph.nodes.write_executor._invoke_write_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_refund_intent_dispatched(self, mock_read, mock_write):
        mock_read.side_effect = [
            {"order_id": "ORD-1", "total_amount": 1000},
            {"eligible": True, "amount": 1000, "refund_type": "original_method"},
        ]
        mock_write.return_value = {"success": True, "message": "Done."}

        ir = make_intent_result(
            intent="refund", policy_action="allow",
            reason_code="REFUND_ALLOWED",
            entities={"order_id": "ORD-1"},
            constraints={"refund_type": "original_method"},
        )
        state = make_base_state()
        state["current_intent"] = ir
        result = write_executor_node(state)
        assert result["processed_results"][0]["status"] == "fulfilled"

    def test_non_write_intent_blocked(self):
        ir = make_intent_result(
            intent="faq", policy_action="allow", reason_code="FAQ_ALLOWED",
        )
        state = make_base_state()
        state["current_intent"] = ir
        result = write_executor_node(state)
        assert result["processed_results"][0]["status"] == "blocked"

    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_exception_in_workflow_produces_failed(self, mock_read):
        mock_read.side_effect = RuntimeError("boom")
        ir = make_intent_result(
            intent="refund", policy_action="allow",
            reason_code="REFUND_ALLOWED",
            entities={"order_id": "ORD-1"},
        )
        state = make_base_state()
        state["current_intent"] = ir
        result = write_executor_node(state)
        assert result["processed_results"][0]["status"] == "failed"
        assert "unexpected error" in result["processed_results"][0]["domain_result"].lower()

    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_pending_tool_call_propagated(self, mock_read):
        """When approval is needed, pending_tool_call must appear in the result."""
        mock_read.side_effect = [
            {"order_id": "ORD-1", "total_amount": 60000},
            {"eligible": True, "amount": 60000, "refund_type": "original_method"},
        ]
        ir = make_intent_result(
            intent="refund", policy_action="allow",
            reason_code="REFUND_ALLOWED",
            entities={"order_id": "ORD-1"},
            constraints={"refund_type": "original_method"},
        )
        state = make_base_state()
        state["current_intent"] = ir
        result = write_executor_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "awaiting_human"
        assert "pending_tool_call" in pr
        ptc = pr["pending_tool_call"]
        assert ptc["tool"] == "initiate_refund"
        assert ptc["args"]["order_id"] == "ORD-1"
        assert ptc["args"]["amount"] == 60000
        assert "GLOBAL_REFUND_CEILING" in ptc["reasons"]

    def test_missing_workflow_for_write_intent_blocked(self):
        ir = make_intent_result(
            intent="refund", policy_action="allow",
            reason_code="REFUND_ALLOWED",
            entities={"order_id": "ORD-1"},
        )
        state = make_base_state()
        state["current_intent"] = ir
        with patch.dict("app.graph.nodes.write_executor._WRITE_WORKFLOWS", {}, clear=True):
            result = write_executor_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "blocked"
        assert "cannot be processed" in pr["domain_result"].lower()


class TestToolInvocationGuards:
    def test_invoke_write_tool_rejects_read_tool(self):
        res = _invoke_write_tool("get_order_details", TENANT_MEGAMART, CUSTOMER_ALICE, order_id="ORD-1")
        assert res["success"] is False
        assert "not a write tool" in res["error"].lower()

    def test_invoke_tool_unknown_tool_returns_error(self):
        res = _invoke_tool("nonexistent_tool", TENANT_MEGAMART, CUSTOMER_ALICE)
        assert res["success"] is False
        assert "unknown tool" in res["error"].lower()
