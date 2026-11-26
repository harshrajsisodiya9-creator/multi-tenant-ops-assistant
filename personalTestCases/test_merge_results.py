"""Tests for merge_results_node.

Covers: empty processed_results, single fulfilled result, multiple results,
tool_calls flag, escalation detection -> partial response_phase,
mixed statuses.
"""
from __future__ import annotations

import pytest

from app.graph.nodes.intent_executor import merge_results_node

from .conftest import make_base_state


class TestMergeResultsNode:
    def test_empty_processed_results(self):
        state = make_base_state()
        state["processed_results"] = []
        result = merge_results_node(state)
        assert result["intent_results"] == []
        assert result["has_tool_calls"] is False
        assert result["response_phase"] == "complete"

    def test_missing_processed_results_key(self):
        state = make_base_state()
        result = merge_results_node(state)
        assert result["intent_results"] == []

    def test_single_fulfilled_no_tools(self):
        state = make_base_state()
        state["processed_results"] = [
            {"intent": "greeting", "status": "fulfilled", "domain_result": "Hello!"},
        ]
        result = merge_results_node(state)
        assert len(result["intent_results"]) == 1
        assert result["has_tool_calls"] is False
        assert result["response_phase"] == "complete"

    def test_has_tool_calls_flag(self):
        state = make_base_state()
        state["processed_results"] = [
            {
                "intent": "refund",
                "status": "fulfilled",
                "tool_calls": [{"tool": "initiate_refund", "args": {}, "result": {"success": True}}],
            },
        ]
        result = merge_results_node(state)
        assert result["has_tool_calls"] is True

    def test_escalation_sets_partial(self):
        state = make_base_state()
        state["processed_results"] = [
            {"intent": "order_status", "status": "fulfilled", "domain_result": "Order shipped."},
            {"intent": "refund", "status": "awaiting_human", "domain_result": "Needs approval."},
        ]
        result = merge_results_node(state)
        assert result["response_phase"] == "partial"

    def test_no_escalation_complete(self):
        state = make_base_state()
        state["processed_results"] = [
            {"intent": "refund", "status": "fulfilled", "domain_result": "Done."},
            {"intent": "cancel_order", "status": "blocked", "domain_result": "Not allowed."},
        ]
        result = merge_results_node(state)
        assert result["response_phase"] == "complete"

    def test_multiple_results_all_included(self):
        state = make_base_state()
        state["processed_results"] = [
            {"intent": "order_status", "status": "fulfilled"},
            {"intent": "refund", "status": "fulfilled"},
            {"intent": "cancel_order", "status": "blocked"},
        ]
        result = merge_results_node(state)
        assert len(result["intent_results"]) == 3

    def test_tool_calls_empty_list_not_truthy(self):
        state = make_base_state()
        state["processed_results"] = [
            {"intent": "refund", "status": "fulfilled", "tool_calls": []},
        ]
        result = merge_results_node(state)
        assert result["has_tool_calls"] is False
