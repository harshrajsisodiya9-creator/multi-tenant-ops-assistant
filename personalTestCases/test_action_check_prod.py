"""Tests for action_check_node production-mode paths.

Covers: RECENT_DELIVERY_REFUND, CANCEL_AFTER_SHIP audit flags,
verification via VERIFIERS, verification failure setting status to failed.
All DB access is mocked — no real database connections.
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.graph.nodes.action_check import (
    NEAR_CEILING_THRESHOLD,
    RECENT_DELIVERY_DAYS,
    _audit_tool_calls,
    _verify_cancel,
    _verify_refund,
    action_check_node,
)
from app.graph.policy_gate import GLOBAL_REFUND_CEILING

from .conftest import CUSTOMER_ALICE, TENANT_MEGAMART, make_base_state


# ===================================================================
# RECENT_DELIVERY_REFUND flag
# ===================================================================

class TestRecentDeliveryRefundFlag:
    @patch("app.graph.nodes.action_check._get_delivery_date")
    def test_recent_delivery_flagged(self, mock_delivery):
        mock_delivery.return_value = date.today()
        tool_calls = [
            {"tool": "initiate_refund", "args": {"order_id": "ORD-1", "amount": 100}, "result": {"success": True}},
        ]
        flags = _audit_tool_calls(tool_calls, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1")
        flag_types = [f["flag"] for f in flags]
        assert "RECENT_DELIVERY_REFUND" in flag_types

    @patch("app.graph.nodes.action_check._get_delivery_date")
    def test_old_delivery_not_flagged(self, mock_delivery):
        mock_delivery.return_value = date.today() - timedelta(days=RECENT_DELIVERY_DAYS + 1)
        tool_calls = [
            {"tool": "initiate_refund", "args": {"order_id": "ORD-1", "amount": 100}, "result": {"success": True}},
        ]
        flags = _audit_tool_calls(tool_calls, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1")
        flag_types = [f["flag"] for f in flags]
        assert "RECENT_DELIVERY_REFUND" not in flag_types

    @patch("app.graph.nodes.action_check._get_delivery_date")
    def test_no_delivery_date_not_flagged(self, mock_delivery):
        mock_delivery.return_value = None
        tool_calls = [
            {"tool": "initiate_refund", "args": {"order_id": "ORD-1", "amount": 100}, "result": {"success": True}},
        ]
        flags = _audit_tool_calls(tool_calls, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1")
        flag_types = [f["flag"] for f in flags]
        assert "RECENT_DELIVERY_REFUND" not in flag_types

    @patch("app.graph.nodes.action_check._get_delivery_date")
    def test_exact_boundary_day_not_flagged(self, mock_delivery):
        """Delivery exactly RECENT_DELIVERY_DAYS ago is NOT recent (uses strict <)."""
        mock_delivery.return_value = date.today() - timedelta(days=RECENT_DELIVERY_DAYS)
        tool_calls = [
            {"tool": "initiate_refund", "args": {"order_id": "ORD-1", "amount": 100}, "result": {"success": True}},
        ]
        flags = _audit_tool_calls(tool_calls, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1")
        flag_types = [f["flag"] for f in flags]
        assert "RECENT_DELIVERY_REFUND" not in flag_types


# ===================================================================
# CANCEL_AFTER_SHIP flag
# ===================================================================

class TestCancelAfterShipFlag:
    @patch("app.graph.nodes.action_check._get_shipping_status")
    def test_shipped_flagged(self, mock_shipping):
        mock_shipping.return_value = "shipped"
        tool_calls = [
            {"tool": "cancel_order", "args": {"order_id": "ORD-1"}, "result": {"success": True}},
        ]
        flags = _audit_tool_calls(tool_calls, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1")
        flag_types = [f["flag"] for f in flags]
        assert "CANCEL_AFTER_SHIP" in flag_types

    @patch("app.graph.nodes.action_check._get_shipping_status")
    def test_in_transit_flagged(self, mock_shipping):
        mock_shipping.return_value = "in_transit"
        tool_calls = [
            {"tool": "cancel_order", "args": {"order_id": "ORD-1"}, "result": {"success": True}},
        ]
        flags = _audit_tool_calls(tool_calls, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1")
        flag_types = [f["flag"] for f in flags]
        assert "CANCEL_AFTER_SHIP" in flag_types

    @patch("app.graph.nodes.action_check._get_shipping_status")
    def test_pending_not_flagged(self, mock_shipping):
        mock_shipping.return_value = "pending"
        tool_calls = [
            {"tool": "cancel_order", "args": {"order_id": "ORD-1"}, "result": {"success": True}},
        ]
        flags = _audit_tool_calls(tool_calls, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1")
        flag_types = [f["flag"] for f in flags]
        assert "CANCEL_AFTER_SHIP" not in flag_types

    @patch("app.graph.nodes.action_check._get_shipping_status")
    def test_none_status_not_flagged(self, mock_shipping):
        mock_shipping.return_value = None
        tool_calls = [
            {"tool": "cancel_order", "args": {"order_id": "ORD-1"}, "result": {"success": True}},
        ]
        flags = _audit_tool_calls(tool_calls, TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-1")
        flag_types = [f["flag"] for f in flags]
        assert "CANCEL_AFTER_SHIP" not in flag_types


# ===================================================================
# Prod-mode verification paths
# ===================================================================

class TestProdVerification:
    @patch("app.graph.nodes.action_check._get_shipping_status", return_value=None)
    @patch("app.graph.nodes.action_check._get_delivery_date", return_value=None)
    @patch("app.graph.nodes.action_check._verify_refund")
    def test_verification_success(self, mock_verify, mock_delivery, mock_shipping):
        mock_verify.return_value = {"verified": True, "amount": 1000, "date": "2026-10-01"}

        state = make_base_state()
        state["intent_results"] = [
            {
                "intent": "refund",
                "status": "fulfilled",
                "domain_result": "Refund processed.",
                "entities": {"order_id": "ORD-1"},
                "tool_calls": [
                    {"tool": "initiate_refund", "args": {"order_id": "ORD-1", "amount": 1000}, "result": {"success": True}},
                ],
            },
        ]

        with patch("app.config.setting") as mock_setting:
            mock_setting.agent_mode = "prod"
            result = action_check_node(state)

        tc = result["intent_results"][0]["tool_calls"][0]
        assert tc["verification"]["verified"] is True
        assert result["intent_results"][0]["status"] == "fulfilled"

    @patch("app.graph.nodes.action_check._get_shipping_status", return_value=None)
    @patch("app.graph.nodes.action_check._get_delivery_date", return_value=None)
    @patch("app.graph.nodes.action_check._verify_refund")
    def test_verification_failure_sets_failed(self, mock_verify, mock_delivery, mock_shipping):
        mock_verify.return_value = {"verified": False, "reason": "no RefundHistory record found"}

        state = make_base_state()
        state["intent_results"] = [
            {
                "intent": "refund",
                "status": "fulfilled",
                "domain_result": "Refund processed.",
                "entities": {"order_id": "ORD-1"},
                "tool_calls": [
                    {"tool": "initiate_refund", "args": {"order_id": "ORD-1", "amount": 1000}, "result": {"success": True}},
                ],
            },
        ]

        with patch("app.config.setting") as mock_setting:
            mock_setting.agent_mode = "prod"
            result = action_check_node(state)

        ir = result["intent_results"][0]
        assert ir["status"] == "failed"
        assert "didn't complete" in ir["domain_result"]
        assert ir["failure_reason"] == "no RefundHistory record found"

    @patch("app.graph.nodes.action_check._get_shipping_status", return_value=None)
    @patch("app.graph.nodes.action_check._get_delivery_date", return_value=None)
    @patch("app.graph.nodes.action_check._verify_cancel")
    def test_cancel_verification_success(self, mock_verify, mock_delivery, mock_shipping):
        mock_verify.return_value = {"verified": True, "status": "cancelled"}

        state = make_base_state()
        state["intent_results"] = [
            {
                "intent": "cancel_order",
                "status": "fulfilled",
                "domain_result": "Order cancelled.",
                "entities": {"order_id": "ORD-1"},
                "tool_calls": [
                    {"tool": "cancel_order", "args": {"order_id": "ORD-1"}, "result": {"success": True}},
                ],
            },
        ]

        with patch("app.config.setting") as mock_setting:
            mock_setting.agent_mode = "prod"
            result = action_check_node(state)

        assert result["intent_results"][0]["status"] == "fulfilled"
        assert result["intent_results"][0]["tool_calls"][0]["verification"]["verified"] is True

    @patch("app.graph.nodes.action_check._get_shipping_status", return_value=None)
    @patch("app.graph.nodes.action_check._get_delivery_date", return_value=None)
    @patch("app.graph.nodes.action_check._verify_cancel")
    def test_cancel_verification_failure_sets_failed(self, mock_verify, mock_delivery, mock_shipping):
        """When cancel verification returns not verified, intent should be failed."""
        mock_verify.return_value = {"verified": False, "reason": "order status is 'pending', expected 'cancelled'"}

        state = make_base_state()
        state["intent_results"] = [
            {
                "intent": "cancel_order",
                "status": "fulfilled",
                "domain_result": "Order cancelled.",
                "entities": {"order_id": "ORD-1"},
                "tool_calls": [
                    {"tool": "cancel_order", "args": {"order_id": "ORD-1"}, "result": {"success": True}},
                ],
            },
        ]

        with patch("app.config.setting") as mock_setting:
            mock_setting.agent_mode = "prod"
            result = action_check_node(state)

        ir = result["intent_results"][0]
        assert ir["status"] == "failed"
        assert "didn't complete" in ir["domain_result"]
        assert ir["failure_reason"] == "order status is 'pending', expected 'cancelled'"


class TestDbVerifiers:
    def test_verify_refund_found_and_not_found(self):
        tid = str(uuid4())
        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = MagicMock(
            amount=1500, refund_date=date(2026, 10, 1),
        )
        with patch("app.database.SessionLocal", return_value=mock_db):
            res = _verify_refund(tid, CUSTOMER_ALICE, "ORD-1")
        assert res == {"verified": True, "amount": 1500, "date": "2026-10-01"}
        mock_db.close.assert_called_once()

        mock_db.reset_mock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = None
        with patch("app.database.SessionLocal", return_value=mock_db):
            res = _verify_refund(tid, CUSTOMER_ALICE, "ORD-1")
        assert res == {"verified": False, "reason": "no RefundHistory record found"}
        mock_db.close.assert_called_once()

    def test_verify_cancel_statuses(self):
        tid = str(uuid4())
        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = MagicMock(status="cancelled")
        with patch("app.database.SessionLocal", return_value=mock_db):
            assert _verify_cancel(tid, CUSTOMER_ALICE, "ORD-1") == {"verified": True, "status": "cancelled"}
        mock_db.close.assert_called_once()

        mock_db.reset_mock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = MagicMock(status="pending")
        with patch("app.database.SessionLocal", return_value=mock_db):
            res = _verify_cancel(tid, CUSTOMER_ALICE, "ORD-1")
        assert res["verified"] is False
        assert "pending" in res["reason"]
        mock_db.close.assert_called_once()

        mock_db.reset_mock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = None
        with patch("app.database.SessionLocal", return_value=mock_db):
            assert _verify_cancel(tid, CUSTOMER_ALICE, "ORD-1") == {"verified": False, "reason": "order not found"}
        mock_db.close.assert_called_once()
