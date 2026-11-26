"""Tests for pre_check_node.

Covers: inactive tenant blocking, low-confidence clarification,
out-of-scope blocking, happy-path passthrough, edge-case confidence values.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.graph.nodes.pre_check import CONVERSATIONAL_INTENTS, pre_check_node

from .conftest import (
    CUSTOMER_ALICE,
    TENANT_INACTIVE,
    TENANT_MEGAMART,
    make_base_state,
    make_router_output,
)


class TestPreCheckBailConditions:
    """Pre-check should bail (pre_check_bail=True) for certain conditions."""

    @patch("app.graph.nodes.pre_check.get_tenant_info")
    def test_inactive_tenant_blocks(self, mock_tenant):
        mock_tenant.return_value = {"active": False}
        state = make_base_state(tenant_id=TENANT_INACTIVE)
        state["router_output"] = make_router_output(intent="order_status", confidence=0.95)

        result = pre_check_node(state)

        assert result["pre_check_bail"] is True
        assert result["policy_decision"]["action"] == "block"
        assert result["policy_decision"]["reason_code"] == "TENANT_INACTIVE"
        assert len(result["intent_results"]) == 1
        assert result["intent_results"][0]["policy_action"] == "block"

    @patch("app.graph.nodes.pre_check.get_tenant_info")
    def test_low_confidence_clarifies(self, mock_tenant):
        mock_tenant.return_value = {"active": True}
        state = make_base_state()
        state["router_output"] = make_router_output(intent="refund", confidence=0.3)

        result = pre_check_node(state)

        assert result["pre_check_bail"] is True
        assert result["policy_decision"]["action"] == "clarify"
        assert result["policy_decision"]["reason_code"] == "LOW_CONFIDENCE"

    @patch("app.graph.nodes.pre_check.get_tenant_info")
    def test_out_of_scope_blocks(self, mock_tenant):
        mock_tenant.return_value = {"active": True}
        state = make_base_state()
        state["router_output"] = make_router_output(intent="out_of_scope", confidence=0.95)

        result = pre_check_node(state)

        assert result["pre_check_bail"] is True
        assert result["policy_decision"]["action"] == "block"
        assert result["policy_decision"]["reason_code"] == "OUT_OF_SCOPE"


class TestPreCheckPassthrough:
    """Pre-check should pass through (pre_check_bail=False) for valid requests."""

    @patch("app.graph.nodes.pre_check.get_tenant_info")
    def test_valid_intent_passes(self, mock_tenant):
        mock_tenant.return_value = {"active": True}
        state = make_base_state()
        state["router_output"] = make_router_output(intent="order_status", confidence=0.95)

        result = pre_check_node(state)
        assert result["pre_check_bail"] is False

    @patch("app.graph.nodes.pre_check.get_tenant_info")
    def test_conversational_intent_passes(self, mock_tenant):
        mock_tenant.return_value = {"active": True}
        for intent in CONVERSATIONAL_INTENTS:
            state = make_base_state()
            state["router_output"] = make_router_output(intent=intent, confidence=0.95)
            result = pre_check_node(state)
            assert result["pre_check_bail"] is False, f"Failed for intent={intent}"


class TestPreCheckEdgeCases:
    """Confidence boundary and missing tenant info."""

    @patch("app.graph.nodes.pre_check.get_tenant_info")
    def test_confidence_exactly_0_6_passes(self, mock_tenant):
        mock_tenant.return_value = {"active": True}
        state = make_base_state()
        state["router_output"] = make_router_output(intent="refund", confidence=0.6)
        result = pre_check_node(state)
        assert result["pre_check_bail"] is False

    @patch("app.graph.nodes.pre_check.get_tenant_info")
    def test_confidence_just_below_0_6_bails(self, mock_tenant):
        mock_tenant.return_value = {"active": True}
        state = make_base_state()
        state["router_output"] = make_router_output(intent="refund", confidence=0.59)
        result = pre_check_node(state)
        assert result["pre_check_bail"] is True

    @patch("app.graph.nodes.pre_check.get_tenant_info")
    def test_missing_tenant_info_defaults_active(self, mock_tenant):
        """get_tenant_info returning None should default to active."""
        mock_tenant.return_value = None
        state = make_base_state()
        state["router_output"] = make_router_output(intent="order_status", confidence=0.95)
        result = pre_check_node(state)
        assert result["pre_check_bail"] is False

    @patch("app.graph.nodes.pre_check.get_tenant_info")
    def test_inactive_tenant_takes_priority_over_low_confidence(self, mock_tenant):
        """Tenant inactive check runs before confidence check."""
        mock_tenant.return_value = {"active": False}
        state = make_base_state(tenant_id=TENANT_INACTIVE)
        state["router_output"] = make_router_output(intent="refund", confidence=0.3)
        result = pre_check_node(state)
        assert result["policy_decision"]["reason_code"] == "TENANT_INACTIVE"
