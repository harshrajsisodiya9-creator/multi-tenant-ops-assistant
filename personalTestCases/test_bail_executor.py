"""Tests for bail_executor_node.

Covers: block action produces blocked status, clarify produces needs_clarification,
escalate produces awaiting_human, unknown action defaults to blocked,
multiple results processed, response_phase set to complete, has_tool_calls=False.
"""
from __future__ import annotations

import pytest

from app.graph.nodes.intent_executor import bail_executor_node

from .conftest import make_base_state, make_intent_result


class TestBailExecutorBlock:
    def test_tenant_inactive_block(self):
        state = make_base_state()
        state["intent_results"] = [
            make_intent_result(
                intent="order_status",
                policy_action="block",
                reason_code="TENANT_INACTIVE",
                status="pending",
            ),
        ]
        result = bail_executor_node(state)
        ir = result["intent_results"][0]
        assert ir["status"] == "blocked"
        assert "unavailable" in ir["domain_result"].lower() or "TENANT_INACTIVE" in ir["domain_result"]
        assert result["response_phase"] == "complete"
        assert result["has_tool_calls"] is False

    def test_out_of_scope_block(self):
        state = make_base_state()
        state["intent_results"] = [
            make_intent_result(
                intent="out_of_scope",
                policy_action="block",
                reason_code="OUT_OF_SCOPE",
                status="pending",
            ),
        ]
        result = bail_executor_node(state)
        assert result["intent_results"][0]["status"] == "blocked"
        assert "scope" in result["intent_results"][0]["domain_result"].lower()


class TestBailExecutorClarify:
    def test_low_confidence_clarify(self):
        state = make_base_state()
        state["intent_results"] = [
            make_intent_result(
                intent="refund",
                policy_action="clarify",
                reason_code="LOW_CONFIDENCE",
                status="pending",
            ),
        ]
        result = bail_executor_node(state)
        ir = result["intent_results"][0]
        assert ir["status"] == "needs_clarification"
        assert "rephrase" in ir["domain_result"].lower()


class TestBailExecutorEscalate:
    def test_escalate_produces_awaiting_human(self):
        state = make_base_state()
        state["intent_results"] = [
            make_intent_result(
                intent="complaint",
                policy_action="escalate",
                reason_code="ESCALATION_KEYWORDS_DETECTED",
                status="pending",
            ),
        ]
        result = bail_executor_node(state)
        ir = result["intent_results"][0]
        assert ir["status"] == "awaiting_human"


class TestBailExecutorEdgeCases:
    def test_unknown_action_defaults_to_blocked(self):
        state = make_base_state()
        state["intent_results"] = [
            make_intent_result(
                intent="refund",
                policy_action="allow",
                reason_code="UNKNOWN",
                status="pending",
            ),
        ]
        result = bail_executor_node(state)
        ir = result["intent_results"][0]
        assert ir["status"] == "blocked"
        assert ir["domain_result"] != ""

    def test_already_non_pending_skipped(self):
        state = make_base_state()
        state["intent_results"] = [
            make_intent_result(
                intent="order_status",
                policy_action="allow",
                reason_code="ORDER_STATUS_ALLOWED",
                status="fulfilled",
                domain_result="Already done.",
            ),
        ]
        result = bail_executor_node(state)
        ir = result["intent_results"][0]
        assert ir["status"] == "fulfilled"
        assert ir["domain_result"] == "Already done."

    def test_multiple_intents_all_processed(self):
        state = make_base_state()
        state["intent_results"] = [
            make_intent_result(
                intent="refund", policy_action="block",
                reason_code="TENANT_INACTIVE", status="pending",
            ),
            make_intent_result(
                intent="cancel_order", policy_action="clarify",
                reason_code="LOW_CONFIDENCE", status="pending",
            ),
        ]
        result = bail_executor_node(state)
        assert result["intent_results"][0]["status"] == "blocked"
        assert result["intent_results"][1]["status"] == "needs_clarification"

    def test_empty_intent_results(self):
        state = make_base_state()
        state["intent_results"] = []
        result = bail_executor_node(state)
        assert result["intent_results"] == []
        assert result["response_phase"] == "complete"

    def test_missing_intent_results_defaults_empty(self):
        state = make_base_state()
        result = bail_executor_node(state)
        assert result["intent_results"] == []
