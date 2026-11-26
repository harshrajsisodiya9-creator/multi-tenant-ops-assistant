"""Tests for condition revalidation inside write workflows.

When the conditional executor delegates to a write workflow, the write
workflow re-reads evidence and re-evaluates the condition before writing.
These tests verify that stale conditions are caught.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.graph.nodes.write_executor import (
    _execute_cancel_workflow,
    _execute_refund_workflow,
    _revalidate_pre_condition,
)

from .conftest import CUSTOMER_ALICE, TENANT_MEGAMART


# ===================================================================
# _revalidate_pre_condition direct tests
# ===================================================================

class TestRevalidatePreCondition:
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_condition_still_satisfied_returns_none(self, mock_read):
        mock_read.return_value = {"order_status": "pending"}
        pre_condition = {
            "condition": {"field": "order_status", "operator": "not_equals", "value": "shipped"},
            "evaluation": {"status": "satisfied"},
        }
        result = _revalidate_pre_condition(
            pre_condition, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1",
        )
        assert result is None

    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_condition_no_longer_satisfied_returns_failure(self, mock_read):
        mock_read.return_value = {"order_status": "shipped"}
        pre_condition = {
            "condition": {"field": "order_status", "operator": "not_equals", "value": "shipped"},
            "evaluation": {"status": "satisfied"},
        }
        result = _revalidate_pre_condition(
            pre_condition, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1",
        )
        assert result is not None
        assert result["status"] == "blocked"
        assert "no longer met" in result["domain_result"]

    def test_none_pre_condition_returns_none(self):
        result = _revalidate_pre_condition(None, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1")
        assert result is None

    def test_empty_condition_returns_none(self):
        result = _revalidate_pre_condition(
            {"condition": {}, "evaluation": {}},
            TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1",
        )
        assert result is None

    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_tool_error_produces_unknown_and_blocks(self, mock_read):
        mock_read.return_value = {"error": "Order lookup failed"}
        pre_condition = {
            "condition": {"field": "order_status", "operator": "not_equals", "value": "shipped"},
            "evaluation": {"status": "satisfied"},
        }
        result = _revalidate_pre_condition(
            pre_condition, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1",
        )
        assert result is not None
        assert result["status"] == "blocked"
        assert "no longer met" in result["domain_result"]


# ===================================================================
# Condition revalidation inside cancel workflow
# ===================================================================

class TestCancelWorkflowRevalidation:
    @patch("app.graph.nodes.write_executor._invoke_write_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_cancel_blocked_when_condition_stale(self, mock_read, mock_write):
        """Order shipped between condition check and write attempt -> blocked."""
        mock_read.side_effect = [
            {"order_id": "ORD-1", "order_status": "shipped"},
            {"order_status": "shipped"},
        ]
        pre_condition = {
            "condition": {"field": "order_status", "operator": "not_equals", "value": "shipped"},
            "evaluation": {"status": "satisfied"},
        }
        result = _execute_cancel_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-1"}, {},
            pre_condition=pre_condition,
        )
        assert result["status"] == "blocked"
        assert "no longer met" in result["domain_result"]
        mock_write.assert_not_called()

    @patch("app.graph.nodes.write_executor._invoke_write_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_cancel_proceeds_when_condition_still_valid(self, mock_read, mock_write):
        mock_read.side_effect = [
            {"order_id": "ORD-1", "order_status": "pending"},
            {"order_status": "pending"},
        ]
        mock_write.return_value = {"success": True, "message": "Cancelled."}

        pre_condition = {
            "condition": {"field": "order_status", "operator": "not_equals", "value": "shipped"},
            "evaluation": {"status": "satisfied"},
        }
        result = _execute_cancel_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-1"}, {},
            pre_condition=pre_condition,
        )
        assert result["status"] == "fulfilled"
        mock_write.assert_called_once()


# ===================================================================
# Condition revalidation inside refund workflow
# ===================================================================

class TestRefundWorkflowRevalidation:
    @patch("app.graph.nodes.write_executor._invoke_write_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_refund_blocked_when_condition_stale(self, mock_read, mock_write):
        """Condition flips before refund write -> blocked."""
        mock_read.side_effect = [
            {"order_id": "ORD-1", "total_amount": 1000},
            {"eligible": True, "amount": 1000, "refund_type": "original_method"},
            {"order_status": "shipped"},
        ]
        pre_condition = {
            "condition": {"field": "order_status", "operator": "not_equals", "value": "shipped"},
            "evaluation": {"status": "satisfied"},
        }
        result = _execute_refund_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-1"},
            {"refund_type": "original_method"},
            pre_condition=pre_condition,
        )
        assert result["status"] == "blocked"
        assert "no longer met" in result["domain_result"]
        mock_write.assert_not_called()

    @patch("app.graph.nodes.write_executor._invoke_write_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_refund_proceeds_when_condition_still_valid(self, mock_read, mock_write):
        mock_read.side_effect = [
            {"order_id": "ORD-1", "total_amount": 1000},
            {"eligible": True, "amount": 1000, "refund_type": "original_method"},
            {"order_status": "pending"},
        ]
        mock_write.return_value = {"success": True, "message": "Refunded."}

        pre_condition = {
            "condition": {"field": "order_status", "operator": "not_equals", "value": "shipped"},
            "evaluation": {"status": "satisfied"},
        }
        result = _execute_refund_workflow(
            TENANT_MEGAMART, CUSTOMER_ALICE,
            {"order_id": "ORD-1"},
            {"refund_type": "original_method"},
            pre_condition=pre_condition,
        )
        assert result["status"] == "fulfilled"
        mock_write.assert_called_once()
