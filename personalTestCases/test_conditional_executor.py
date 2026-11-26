"""Tests for conditional_executor_node.

Covers: conditional_action satisfied -> write, not satisfied -> no action,
conditional_question answered, unknown condition, missing condition,
condition satisfied but non-write intent, exception handling.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.graph.condition_evaluator import ConditionResult
from app.graph.nodes.conditional_executor import conditional_executor_node

from .conftest import CUSTOMER_ALICE, TENANT_MEGAMART, make_base_state, make_intent_result


def _make_conditional_state(intent_result: dict) -> dict:
    state = make_base_state(user_message="If it hasn't shipped, cancel it")
    state["current_intent"] = intent_result
    return state


class TestConditionalActionSatisfied:
    """When condition is satisfied and request_type is conditional_action."""

    @patch("app.graph.nodes.conditional_executor.evaluate_condition")
    @patch("app.graph.nodes.conditional_executor._gather_evidence")
    def test_delegates_to_write_workflow(self, mock_evidence, mock_eval):
        mock_evidence.return_value = (
            {"get_order_details": {"order_status": "pending"}},
            [{"tool": "get_order_details", "args": {"order_id": "ORD-1"}, "result": {"order_status": "pending"}}],
        )
        mock_eval.return_value = {
            "status": ConditionResult.SATISFIED,
            "reason": None,
            "actual_value": "pending",
        }
        fake_workflow = MagicMock(return_value={
            "domain_result": "Order cancelled.",
            "status": "fulfilled",
            "tool_calls": [{"tool": "cancel_order", "args": {"order_id": "ORD-1"}, "result": {"success": True}}],
        })

        ir = make_intent_result(
            intent="cancel_order", policy_action="allow",
            reason_code="CANCEL_ALLOWED",
            entities={"order_id": "ORD-1"},
            conditional={
                "request_type": "conditional_action",
                "condition": {"field": "order_status", "operator": "not_equals", "value": "shipped"},
            },
        )
        state = _make_conditional_state(ir)
        with patch.dict("app.graph.nodes.conditional_executor._WRITE_WORKFLOWS", {"cancel_order": fake_workflow}):
            result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert len(pr["tool_calls"]) >= 1

        assert pr["conditional"]["evaluation"]["status"] == ConditionResult.SATISFIED
        assert pr["conditional"]["evaluation"]["actual_value"] == "pending"
        assert pr["conditional"]["request_type"] == "conditional_action"

        mock_evidence.assert_called_once()
        mock_eval.assert_called_once()
        fake_workflow.assert_called_once()
        pre_condition = fake_workflow.call_args.kwargs["pre_condition"]
        assert pre_condition["condition"]["field"] == "order_status"
        assert pre_condition["evaluation"]["status"] == ConditionResult.SATISFIED


class TestConditionalActionNotSatisfied:
    @patch("app.graph.nodes.conditional_executor.evaluate_condition")
    @patch("app.graph.nodes.conditional_executor._gather_evidence")
    def test_no_action_taken(self, mock_evidence, mock_eval):
        mock_evidence.return_value = (
            {"get_order_details": {"order_status": "shipped"}},
            [{"tool": "get_order_details", "args": {"order_id": "ORD-1"}, "result": {"order_status": "shipped"}}],
        )
        mock_eval.return_value = {
            "status": ConditionResult.NOT_SATISFIED,
            "reason": "The order status is 'shipped', not 'pending'.",
            "actual_value": "shipped",
        }

        ir = make_intent_result(
            intent="cancel_order", policy_action="allow",
            reason_code="CANCEL_ALLOWED",
            entities={"order_id": "ORD-1"},
            conditional={
                "request_type": "conditional_action",
                "condition": {"field": "order_status", "operator": "not_equals", "value": "shipped"},
            },
        )
        state = _make_conditional_state(ir)
        result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "not met" in pr["domain_result"].lower()

        assert pr["conditional"]["evaluation"]["status"] == ConditionResult.NOT_SATISFIED
        assert pr["conditional"]["evaluation"]["actual_value"] == "shipped"
        mock_evidence.assert_called_once()
        mock_eval.assert_called_once()


class TestConditionalQuestion:
    @patch("app.graph.nodes.conditional_executor.evaluate_condition")
    @patch("app.graph.nodes.conditional_executor._gather_evidence")
    def test_satisfied_question(self, mock_evidence, mock_eval):
        mock_evidence.return_value = (
            {"get_order_details": {"order_status": "shipped"}},
            [{"tool": "get_order_details", "args": {}, "result": {"order_status": "shipped"}}],
        )
        mock_eval.return_value = {
            "status": ConditionResult.SATISFIED,
            "reason": None,
            "actual_value": "shipped",
        }

        ir = make_intent_result(
            intent="order_status", policy_action="allow",
            reason_code="ORDER_STATUS_ALLOWED",
            entities={"order_id": "ORD-1"},
            conditional={
                "request_type": "conditional_question",
                "condition": {"field": "order_status", "operator": "equals", "value": "shipped"},
            },
        )
        state = _make_conditional_state(ir)
        result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "yes" in pr["domain_result"].lower()

    @patch("app.graph.nodes.conditional_executor.evaluate_condition")
    @patch("app.graph.nodes.conditional_executor._gather_evidence")
    def test_not_satisfied_question(self, mock_evidence, mock_eval):
        mock_evidence.return_value = (
            {"get_order_details": {"order_status": "pending"}},
            [],
        )
        mock_eval.return_value = {
            "status": ConditionResult.NOT_SATISFIED,
            "reason": "The order status is 'pending', not 'shipped'.",
            "actual_value": "pending",
        }

        ir = make_intent_result(
            intent="order_status", policy_action="allow",
            reason_code="ORDER_STATUS_ALLOWED",
            entities={"order_id": "ORD-1"},
            conditional={
                "request_type": "conditional_question",
                "condition": {"field": "order_status", "operator": "equals", "value": "shipped"},
            },
        )
        state = _make_conditional_state(ir)
        result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "no" in pr["domain_result"].lower()


class TestConditionalUnknown:
    @patch("app.graph.nodes.conditional_executor.evaluate_condition")
    @patch("app.graph.nodes.conditional_executor._gather_evidence")
    def test_unknown_returns_clarification_prompt(self, mock_evidence, mock_eval):
        """When condition evaluation is UNKNOWN, node asks for more details.

        Note: conditional_executor_node currently sets status='fulfilled' (existing quirk;
        ideally this would be 'needs_clarification').
        """
        mock_evidence.return_value = ({}, [])
        mock_eval.return_value = {
            "status": ConditionResult.UNKNOWN,
            "reason": "Required data not available.",
            "actual_value": None,
        }

        ir = make_intent_result(
            intent="cancel_order", policy_action="allow",
            reason_code="CANCEL_ALLOWED",
            entities={"order_id": "ORD-1"},
            conditional={
                "request_type": "conditional_action",
                "condition": {"field": "order_status", "operator": "equals", "value": "pending"},
            },
        )
        state = _make_conditional_state(ir)
        result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "couldn't determine" in pr["domain_result"].lower()
        assert "more details" in pr["domain_result"].lower()


class TestConditionalEdgeCases:
    def test_missing_condition_fails_closed(self):
        ir = make_intent_result(
            intent="cancel_order", policy_action="allow",
            reason_code="CANCEL_ALLOWED",
            entities={"order_id": "ORD-1"},
            conditional={"request_type": "conditional_action", "condition": {}},
        )
        state = _make_conditional_state(ir)
        result = conditional_executor_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "could not evaluate" in pr["domain_result"].lower()

    def test_no_conditional_field_at_all(self):
        ir = make_intent_result(
            intent="cancel_order", policy_action="allow",
            reason_code="CANCEL_ALLOWED",
            entities={"order_id": "ORD-1"},
            conditional=None,
        )
        state = _make_conditional_state(ir)
        result = conditional_executor_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "could not evaluate" in pr["domain_result"].lower()
        assert pr.get("tool_calls", []) == []

    @patch("app.graph.nodes.conditional_executor.evaluate_condition")
    @patch("app.graph.nodes.conditional_executor._gather_evidence")
    def test_satisfied_non_write_intent_blocked(self, mock_evidence, mock_eval):
        """Condition satisfied but intent is not a write intent -> blocked."""
        mock_evidence.return_value = ({}, [])
        mock_eval.return_value = {
            "status": ConditionResult.SATISFIED,
            "reason": None,
            "actual_value": "pending",
        }

        ir = make_intent_result(
            intent="order_status", policy_action="allow",
            reason_code="ORDER_STATUS_ALLOWED",
            entities={"order_id": "ORD-1"},
            conditional={
                "request_type": "conditional_action",
                "condition": {"field": "order_status", "operator": "equals", "value": "pending"},
            },
        )
        state = _make_conditional_state(ir)
        result = conditional_executor_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "blocked"

    @patch("app.graph.nodes.conditional_executor._gather_evidence")
    def test_condition_satisfied_with_real_evaluator(self, mock_evidence):
        """Use the REAL evaluate_condition (not mocked) to test integration."""
        mock_evidence.return_value = (
            {"get_order_details": {"order_status": "pending"}},
            [{"tool": "get_order_details", "args": {"order_id": "ORD-1"}, "result": {"order_status": "pending"}}],
        )

        ir = make_intent_result(
            intent="order_status", policy_action="allow",
            reason_code="ORDER_STATUS_ALLOWED",
            entities={"order_id": "ORD-1"},
            conditional={
                "request_type": "conditional_question",
                "condition": {"field": "order_status", "operator": "not_equals", "value": "shipped"},
            },
        )
        state = _make_conditional_state(ir)
        result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "yes" in pr["domain_result"].lower()
        assert pr["conditional"]["evaluation"]["status"] == ConditionResult.SATISFIED

    @patch("app.graph.nodes.conditional_executor._gather_evidence")
    def test_condition_not_satisfied_with_real_evaluator(self, mock_evidence):
        """Real evaluate_condition + NOT_SATISFIED path."""
        mock_evidence.return_value = (
            {"get_order_details": {"order_status": "shipped"}},
            [{"tool": "get_order_details", "args": {"order_id": "ORD-1"}, "result": {"order_status": "shipped"}}],
        )

        ir = make_intent_result(
            intent="order_status", policy_action="allow",
            reason_code="ORDER_STATUS_ALLOWED",
            entities={"order_id": "ORD-1"},
            conditional={
                "request_type": "conditional_question",
                "condition": {"field": "order_status", "operator": "not_equals", "value": "shipped"},
            },
        )
        state = _make_conditional_state(ir)
        result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "no" in pr["domain_result"].lower()
        assert pr["conditional"]["evaluation"]["status"] == ConditionResult.NOT_SATISFIED

    @patch("app.graph.nodes.conditional_executor._gather_evidence")
    def test_exception_produces_failed(self, mock_evidence):
        mock_evidence.side_effect = RuntimeError("boom")
        ir = make_intent_result(
            intent="cancel_order", policy_action="allow",
            reason_code="CANCEL_ALLOWED",
            entities={"order_id": "ORD-1"},
            conditional={
                "request_type": "conditional_action",
                "condition": {"field": "order_status", "operator": "equals", "value": "pending"},
            },
        )
        state = _make_conditional_state(ir)
        result = conditional_executor_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "failed"
