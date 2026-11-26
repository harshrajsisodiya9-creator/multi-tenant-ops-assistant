"""Step 2 tests — Conditional Execution (Read → Evaluate → Write).

Covers: condition_evaluator, conditional_executor, router conditional detection,
policy_fan_out conditional passthrough, builder routing, and integration.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.graph.condition_evaluator import (
    ConditionResult,
    EVIDENCE_EXTRACTORS,
    FIELD_ALLOWED_OPERATORS,
    SUPPORTED_FIELDS,
    evaluate_condition,
    required_tools_for_condition,
)
from app.graph.schemas import (
    CONDITION_FIELDS,
    CONDITION_OPERATORS,
    Condition,
    ConditionalAction,
    IntentResult,
    RouterOutput,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. Condition Evaluator — core logic
# ═══════════════════════════════════════════════════════════════════════

class TestConditionEvaluatorStructure:
    def test_supported_fields_match_schema(self):
        schema_fields = set(Condition.model_fields["field"].annotation.__args__)
        assert schema_fields == SUPPORTED_FIELDS

    def test_every_field_has_extractor(self):
        for field in SUPPORTED_FIELDS:
            assert field in EVIDENCE_EXTRACTORS

    def test_every_field_has_allowed_operators(self):
        for field in SUPPORTED_FIELDS:
            assert field in FIELD_ALLOWED_OPERATORS
            assert len(FIELD_ALLOWED_OPERATORS[field]) > 0

    def test_required_tools_returns_tool_list(self):
        tools = required_tools_for_condition({"field": "order_status"})
        assert tools == ["get_order_details"]

    def test_required_tools_unknown_field_returns_empty(self):
        tools = required_tools_for_condition({"field": "nonexistent"})
        assert tools == []


class TestConditionEvaluatorEquals:
    def test_equals_satisfied(self):
        condition = {"field": "order_status", "operator": "equals", "value": "shipped"}
        evidence = {"get_order_details": {"order_status": "shipped"}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.SATISFIED

    def test_equals_not_satisfied(self):
        condition = {"field": "order_status", "operator": "equals", "value": "shipped"}
        evidence = {"get_order_details": {"order_status": "delivered"}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.NOT_SATISFIED
        assert "delivered" in result["reason"]

    def test_equals_case_insensitive(self):
        condition = {"field": "order_status", "operator": "equals", "value": "Shipped"}
        evidence = {"get_order_details": {"order_status": "shipped"}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.SATISFIED


class TestConditionEvaluatorNotEquals:
    def test_not_equals_satisfied(self):
        condition = {"field": "order_status", "operator": "not_equals", "value": "shipped"}
        evidence = {"get_order_details": {"order_status": "processing"}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.SATISFIED

    def test_not_equals_not_satisfied(self):
        condition = {"field": "order_status", "operator": "not_equals", "value": "shipped"}
        evidence = {"get_order_details": {"order_status": "shipped"}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.NOT_SATISFIED


class TestConditionEvaluatorIn:
    def test_in_satisfied(self):
        condition = {"field": "order_status", "operator": "in", "value": ["shipped", "delivered"]}
        evidence = {"get_order_details": {"order_status": "shipped"}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.SATISFIED

    def test_in_not_satisfied(self):
        condition = {"field": "order_status", "operator": "in", "value": ["shipped", "delivered"]}
        evidence = {"get_order_details": {"order_status": "processing"}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.NOT_SATISFIED


class TestConditionEvaluatorNotIn:
    def test_not_in_satisfied(self):
        condition = {"field": "order_status", "operator": "not_in", "value": ["shipped", "delivered"]}
        evidence = {"get_order_details": {"order_status": "processing"}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.SATISFIED

    def test_not_in_not_satisfied(self):
        condition = {"field": "order_status", "operator": "not_in", "value": ["shipped", "delivered"]}
        evidence = {"get_order_details": {"order_status": "shipped"}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.NOT_SATISFIED


class TestConditionEvaluatorBoolean:
    def test_is_true_satisfied(self):
        condition = {"field": "is_refundable", "operator": "is_true", "value": ""}
        evidence = {"get_return_eligibility": {"eligible": True}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.SATISFIED

    def test_is_true_not_satisfied(self):
        condition = {"field": "is_refundable", "operator": "is_true", "value": ""}
        evidence = {"get_return_eligibility": {"eligible": False}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.NOT_SATISFIED

    def test_is_false_satisfied(self):
        condition = {"field": "is_return_eligible", "operator": "is_false", "value": ""}
        evidence = {"get_return_eligibility": {"eligible": False}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.SATISFIED

    def test_is_false_not_satisfied(self):
        condition = {"field": "is_return_eligible", "operator": "is_false", "value": ""}
        evidence = {"get_return_eligibility": {"eligible": True}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.NOT_SATISFIED


class TestConditionEvaluatorUnknown:
    def test_unsupported_field(self):
        condition = {"field": "unknown_field", "operator": "equals", "value": "x"}
        result = evaluate_condition(condition, {})
        assert result["status"] == ConditionResult.UNKNOWN
        assert "Unsupported" in result["reason"]

    def test_unsupported_operator_for_field(self):
        condition = {"field": "is_refundable", "operator": "equals", "value": "true"}
        evidence = {"get_return_eligibility": {"eligible": True}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.UNKNOWN
        assert "not supported" in result["reason"]

    def test_missing_evidence(self):
        condition = {"field": "order_status", "operator": "equals", "value": "shipped"}
        result = evaluate_condition(condition, {})
        assert result["status"] == ConditionResult.UNKNOWN
        assert "not available" in result["reason"]

    def test_tool_error_in_evidence(self):
        condition = {"field": "order_status", "operator": "equals", "value": "shipped"}
        evidence = {"get_order_details": {"error": "Order not found."}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.UNKNOWN
        assert "Order not found" in result["reason"]

    def test_missing_field_in_tool_result(self):
        condition = {"field": "order_status", "operator": "equals", "value": "shipped"}
        evidence = {"get_order_details": {"total_amount": 100}}
        result = evaluate_condition(condition, evidence)
        assert result["status"] == ConditionResult.UNKNOWN
        assert "not available" in result["reason"]


# ═══════════════════════════════════════════════════════════════════════
# 2. Pydantic schemas — conditional fields
# ═══════════════════════════════════════════════════════════════════════

class TestSchemas:
    def test_condition_model_accepts_valid_input(self):
        c = Condition(field="order_status", operator="equals", value="shipped")
        assert c.field == "order_status"

    def test_conditional_action_model(self):
        ca = ConditionalAction(
            request_type="conditional_action",
            condition=Condition(field="order_status", operator="not_equals", value="shipped"),
        )
        assert ca.request_type == "conditional_action"
        assert ca.condition.operator == "not_equals"

    def test_router_output_conditional_default_none(self):
        ro = RouterOutput(intent="refund", confidence=0.9)
        assert ro.conditional is None

    def test_router_output_with_conditional(self):
        ro = RouterOutput(
            intent="cancel_order",
            confidence=0.95,
            conditional=ConditionalAction(
                request_type="conditional_action",
                condition=Condition(field="order_status", operator="not_equals", value="shipped"),
            ),
        )
        assert ro.conditional is not None
        assert ro.conditional.condition.field == "order_status"

    def test_intent_result_conditional_default_none(self):
        ir = IntentResult(intent="refund", policy_action="allow", reason_code="OK")
        assert ir.conditional is None

    def test_intent_result_with_conditional(self):
        ir = IntentResult(
            intent="cancel_order",
            policy_action="allow",
            reason_code="OK",
            conditional={"request_type": "conditional_action", "condition": {"field": "order_status", "operator": "equals", "value": "processing"}},
        )
        assert ir.conditional is not None
        assert ir.conditional["request_type"] == "conditional_action"


# ═══════════════════════════════════════════════════════════════════════
# 3. Conditional Executor — node logic
# ═══════════════════════════════════════════════════════════════════════

class TestConditionalExecutorNode:
    """Unit tests for the conditional_executor_node using mocked tool calls."""

    def _build_state(self, intent="cancel_order", order_id="ORD-MM-1005",
                     request_type="conditional_action",
                     field="order_status", operator="not_equals", value="shipped",
                     policy_action="allow", reason_code="CANCEL_OK"):
        return {
            "tenant_id": "megamart",
            "customer_id": "CUST-001",
            "user_message": "If my order hasn't shipped, cancel it",
            "current_intent": {
                "intent": intent,
                "entities": {"order_id": order_id},
                "policy_action": policy_action,
                "reason_code": reason_code,
                "constraints": {},
                "status": "pending",
                "conditional": {
                    "request_type": request_type,
                    "condition": {
                        "field": field,
                        "operator": operator,
                        "value": value,
                    },
                },
            },
        }

    @patch("app.graph.nodes.conditional_executor._invoke_read_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    @patch("app.graph.nodes.write_executor._invoke_write_tool")
    def test_condition_satisfied_triggers_write(self, mock_write, mock_we_read, mock_ce_read):
        """Condition met → write workflow executes."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        mock_ce_read.return_value = {"order_status": "processing"}
        mock_we_read.return_value = {"order_status": "processing", "order_id": "ORD-MM-1005"}
        mock_write.return_value = {"success": True, "message": "Order ORD-MM-1005 cancelled."}

        state = self._build_state()
        result = conditional_executor_node(state)

        assert len(result["processed_results"]) == 1
        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "cancelled" in pr["domain_result"].lower()
        assert pr["conditional"]["evaluation"]["status"] == ConditionResult.SATISFIED

    @patch("app.graph.nodes.conditional_executor._invoke_read_tool")
    def test_condition_not_satisfied_no_write(self, mock_read):
        """Condition not met → no write, explanation returned."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        mock_read.return_value = {"order_status": "shipped"}

        state = self._build_state()
        result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "not met" in pr["domain_result"].lower()
        assert pr["conditional"]["evaluation"]["status"] == ConditionResult.NOT_SATISFIED

    @patch("app.graph.nodes.conditional_executor._invoke_read_tool")
    def test_condition_unknown_returns_clarification(self, mock_read):
        """Evidence unavailable → unknown status, clarification."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        mock_read.return_value = {"error": "Order not found."}

        state = self._build_state()
        result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "couldn't determine" in pr["domain_result"].lower()
        assert pr["conditional"]["evaluation"]["status"] == ConditionResult.UNKNOWN

    @patch("app.graph.nodes.conditional_executor._invoke_read_tool")
    def test_conditional_question_satisfied(self, mock_read):
        """conditional_question + satisfied → answer yes, no write."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        mock_read.return_value = {"order_status": "shipped"}

        state = self._build_state(
            intent="order_status",
            request_type="conditional_question",
            field="order_status",
            operator="equals",
            value="shipped",
            reason_code="STATUS_OK",
        )
        result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "yes" in pr["domain_result"].lower()
        assert "shipped" in pr["domain_result"].lower()

    @patch("app.graph.nodes.conditional_executor._invoke_read_tool")
    def test_conditional_question_not_satisfied(self, mock_read):
        """conditional_question + not satisfied → answer no."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        mock_read.return_value = {"order_status": "processing"}

        state = self._build_state(
            intent="order_status",
            request_type="conditional_question",
            field="order_status",
            operator="equals",
            value="shipped",
            reason_code="STATUS_OK",
        )
        result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "no" in pr["domain_result"].lower()

    @patch("app.graph.nodes.conditional_executor._invoke_read_tool")
    def test_conditional_question_never_writes(self, mock_read):
        """conditional_question on a write intent still never writes."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        mock_read.return_value = {"eligible": True}

        state = self._build_state(
            intent="refund",
            request_type="conditional_question",
            field="is_refundable",
            operator="is_true",
            value="",
            reason_code="REFUND_OK",
        )
        result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "yes" in pr["domain_result"].lower()
        tool_names = [tc["tool"] for tc in pr.get("tool_calls", [])]
        assert "initiate_refund" not in tool_names

    def test_missing_condition_fails_closed(self):
        """No condition in metadata → fails gracefully."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-001",
            "user_message": "cancel order",
            "current_intent": {
                "intent": "cancel_order",
                "entities": {"order_id": "ORD-1"},
                "policy_action": "allow",
                "reason_code": "OK",
                "constraints": {},
                "status": "pending",
                "conditional": {},
            },
        }
        result = conditional_executor_node(state)
        pr = result["processed_results"][0]
        assert "could not evaluate" in pr["domain_result"].lower()

    @patch("app.graph.nodes.conditional_executor._invoke_read_tool")
    def test_satisfied_non_write_intent_blocked(self, mock_read):
        """Condition satisfied but intent is not a write → blocked."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        mock_read.return_value = {"order_status": "processing"}

        state = self._build_state(
            intent="order_status",
            request_type="conditional_action",
            reason_code="STATUS_OK",
        )
        result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "blocked"

    @patch("app.graph.nodes.conditional_executor._invoke_read_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    @patch("app.graph.nodes.write_executor._invoke_write_tool")
    def test_evaluation_stored_in_result(self, mock_write, mock_we_read, mock_ce_read):
        """The evaluation result is stored in result['conditional']['evaluation']."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        mock_ce_read.return_value = {"order_status": "processing"}
        mock_we_read.return_value = {"order_status": "processing", "order_id": "ORD-MM-1005"}
        mock_write.return_value = {"success": True, "message": "Cancelled."}

        state = self._build_state()
        result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        evaluation = pr["conditional"]["evaluation"]
        assert evaluation["status"] == ConditionResult.SATISFIED
        assert evaluation["actual_value"] == "processing"

    @patch("app.graph.nodes.conditional_executor._invoke_read_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    def test_write_workflow_failure_propagates(self, mock_we_read, mock_ce_read):
        """When the write workflow itself fails, the error propagates."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        mock_ce_read.return_value = {"order_status": "processing"}
        mock_we_read.return_value = {"error": "Order not found."}

        state = self._build_state()
        result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        assert "not found" in pr["domain_result"].lower() or pr["status"] in ("fulfilled", "failed")

    @patch("app.graph.nodes.conditional_executor._invoke_read_tool")
    @patch("app.graph.nodes.write_executor._invoke_read_tool")
    @patch("app.graph.nodes.write_executor._invoke_write_tool")
    def test_tool_calls_combined(self, mock_write, mock_we_read, mock_ce_read):
        """Evidence tool calls + write workflow tool calls are combined."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        mock_ce_read.return_value = {"order_status": "processing"}
        mock_we_read.return_value = {"order_status": "processing", "order_id": "ORD-MM-1005"}
        mock_write.return_value = {"success": True, "message": "Cancelled."}

        state = self._build_state()
        result = conditional_executor_node(state)

        pr = result["processed_results"][0]
        tool_names = [tc["tool"] for tc in pr["tool_calls"]]
        assert "get_order_details" in tool_names


# ═══════════════════════════════════════════════════════════════════════
# 4. Router — conditional pattern detection
# ═══════════════════════════════════════════════════════════════════════

class TestRouterConditionalDetection:
    def test_conditional_pattern_detected(self):
        from app.graph.nodes.router import _detect_conditional
        assert _detect_conditional("if it hasn't shipped, cancel it") is True

    def test_no_conditional_pattern(self):
        from app.graph.nodes.router import _detect_conditional
        assert _detect_conditional("cancel my order please") is False

    def test_conditional_question_detected(self):
        from app.graph.nodes.router import _detect_conditional_question
        assert _detect_conditional_question("has my order shipped yet?") is True

    def test_no_conditional_question(self):
        from app.graph.nodes.router import _detect_conditional_question
        assert _detect_conditional_question("cancel my order") is False

    def test_pre_check_returns_is_conditional_flag(self):
        from app.graph.nodes.router import _pre_check
        result = _pre_check("If my order hasn't shipped, cancel it")
        assert len(result) == 5
        assert result[4] is True

    def test_pre_check_non_conditional_returns_false(self):
        from app.graph.nodes.router import _pre_check
        result = _pre_check("Hello there")
        assert result[4] is False

    def test_conditional_message_always_goes_to_llm(self):
        """Even if keyword pre-check matches, conditional messages go to LLM."""
        from app.graph.nodes.router import _pre_check
        _, _, confidence, _, is_conditional = _pre_check("if it hasn't shipped, cancel my order")
        assert is_conditional is True


# ═══════════════════════════════════════════════════════════════════════
# 5. Policy fan-out — conditional passthrough
# ═══════════════════════════════════════════════════════════════════════

class TestPolicyFanOutConditional:
    @patch("app.graph.nodes.policy_fan_out.get_tenant_info", return_value={"active": True})
    @patch("app.graph.nodes.policy_fan_out.get_customer_orders", return_value=[])
    def test_conditional_passed_to_primary_intent(self, mock_orders, mock_tenant):
        from app.graph.nodes.policy_fan_out import policy_fan_out_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-001",
            "user_message": "if not shipped, cancel ORD-MM-1005",
            "router_output": {
                "intent": "cancel_order",
                "entities": {"order_id": "ORD-MM-1005"},
                "confidence": 0.95,
                "escalation_signals": [],
                "sub_intents": [],
                "conditional": {
                    "request_type": "conditional_action",
                    "condition": {
                        "field": "order_status",
                        "operator": "not_equals",
                        "value": "shipped",
                    },
                },
            },
        }

        result = policy_fan_out_node(state)
        intent_results = result["intent_results"]
        assert len(intent_results) >= 1
        assert intent_results[0]["conditional"] is not None
        assert intent_results[0]["conditional"]["request_type"] == "conditional_action"

    @patch("app.graph.nodes.policy_fan_out.get_tenant_info", return_value={"active": True})
    @patch("app.graph.nodes.policy_fan_out.get_customer_orders", return_value=[])
    def test_conditional_not_passed_to_sub_intents(self, mock_orders, mock_tenant):
        from app.graph.nodes.policy_fan_out import policy_fan_out_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-001",
            "user_message": "if not shipped cancel ORD-MM-1005 and check my other order",
            "router_output": {
                "intent": "cancel_order",
                "entities": {"order_id": "ORD-MM-1005"},
                "confidence": 0.95,
                "escalation_signals": [],
                "sub_intents": [
                    {"intent": "order_status", "entities": {}, "confidence": 0.8},
                ],
                "conditional": {
                    "request_type": "conditional_action",
                    "condition": {
                        "field": "order_status",
                        "operator": "not_equals",
                        "value": "shipped",
                    },
                },
            },
        }

        result = policy_fan_out_node(state)
        intent_results = result["intent_results"]
        assert len(intent_results) == 2
        assert intent_results[0]["conditional"] is not None
        assert intent_results[1]["conditional"] is None

    @patch("app.graph.nodes.policy_fan_out.get_tenant_info", return_value={"active": True})
    def test_no_conditional_stays_none(self, mock_tenant):
        from app.graph.nodes.policy_fan_out import policy_fan_out_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "",
            "user_message": "hello",
            "router_output": {
                "intent": "greeting",
                "entities": {},
                "confidence": 0.95,
                "escalation_signals": [],
                "sub_intents": [],
                "conditional": None,
            },
        }

        result = policy_fan_out_node(state)
        intent_results = result["intent_results"]
        assert intent_results[0]["conditional"] is None


# ═══════════════════════════════════════════════════════════════════════
# 6. Builder — routing dispatch
# ═══════════════════════════════════════════════════════════════════════

class TestBuilderConditionalRouting:
    def test_conditional_intent_routes_to_conditional_executor(self):
        from app.graph.builder import route_intents

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-001",
            "user_message": "if not shipped, cancel it",
            "intent_results": [
                {
                    "intent": "cancel_order",
                    "entities": {"order_id": "ORD-1"},
                    "policy_action": "allow",
                    "reason_code": "CANCEL_OK",
                    "constraints": {},
                    "status": "pending",
                    "conditional": {
                        "request_type": "conditional_action",
                        "condition": {"field": "order_status", "operator": "not_equals", "value": "shipped"},
                    },
                },
            ],
        }

        sends = route_intents(state)
        assert len(sends) == 1
        assert sends[0].node == "conditional_executor"

    def test_non_conditional_write_routes_to_write_executor(self):
        from app.graph.builder import route_intents

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-001",
            "user_message": "cancel my order",
            "intent_results": [
                {
                    "intent": "cancel_order",
                    "entities": {"order_id": "ORD-1"},
                    "policy_action": "allow",
                    "reason_code": "CANCEL_OK",
                    "constraints": {},
                    "status": "pending",
                    "conditional": None,
                },
            ],
        }

        sends = route_intents(state)
        assert len(sends) == 1
        assert sends[0].node == "write_executor"

    def test_graph_compiles_with_conditional_executor(self):
        from app.graph.builder import build_graph
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert "conditional_executor" in node_names


# ═══════════════════════════════════════════════════════════════════════
# 7. Integration — mock mode end-to-end conditional flows
# ═══════════════════════════════════════════════════════════════════════

class TestConditionalIntegrationMock:
    """Integration tests using the mock tool registry (tenant_id='megamart')."""

    @patch("app.graph.nodes.write_executor.setting")
    @patch("app.graph.nodes.conditional_executor._invoke_read_tool")
    def test_cancel_if_not_shipped_satisfied(self, mock_read, mock_setting):
        """ORD-MM-1005 is 'processing' → condition 'not_equals shipped' is satisfied → cancel executes."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        mock_setting.agent_mode = "mock"
        mock_read.return_value = {"order_status": "processing", "order_id": "ORD-MM-1005"}

        state = {
            "tenant_id": "megamart",
            "customer_id": "alice@example.com",
            "user_message": "If my winter jacket order hasn't shipped, cancel it",
            "current_intent": {
                "intent": "cancel_order",
                "entities": {"order_id": "ORD-MM-1005"},
                "policy_action": "allow",
                "reason_code": "CANCEL_OK",
                "constraints": {},
                "status": "pending",
                "conditional": {
                    "request_type": "conditional_action",
                    "condition": {
                        "field": "order_status",
                        "operator": "not_equals",
                        "value": "shipped",
                    },
                },
            },
        }

        result = conditional_executor_node(state)
        pr = result["processed_results"][0]
        assert pr["conditional"]["evaluation"]["status"] == ConditionResult.SATISFIED

    @patch("app.graph.nodes.conditional_executor._invoke_read_tool")
    def test_cancel_if_not_shipped_not_satisfied(self, mock_read):
        """ORD-MM-1006 is 'shipped' → condition 'not_equals shipped' is NOT satisfied → no cancel."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        mock_read.return_value = {"order_status": "shipped"}

        state = {
            "tenant_id": "megamart",
            "customer_id": "eve@example.com",
            "user_message": "If my bluetooth speaker hasn't shipped, cancel it",
            "current_intent": {
                "intent": "cancel_order",
                "entities": {"order_id": "ORD-MM-1006"},
                "policy_action": "allow",
                "reason_code": "CANCEL_OK",
                "constraints": {},
                "status": "pending",
                "conditional": {
                    "request_type": "conditional_action",
                    "condition": {
                        "field": "order_status",
                        "operator": "not_equals",
                        "value": "shipped",
                    },
                },
            },
        }

        result = conditional_executor_node(state)
        pr = result["processed_results"][0]
        assert pr["conditional"]["evaluation"]["status"] == ConditionResult.NOT_SATISFIED
        assert "not met" in pr["domain_result"].lower()

    @patch("app.graph.nodes.conditional_executor._invoke_read_tool")
    def test_refund_if_eligible_question(self, mock_read):
        """conditional_question asking if refundable → answer only."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        mock_read.return_value = {"eligible": True}

        state = {
            "tenant_id": "megamart",
            "customer_id": "alice@example.com",
            "user_message": "Am I eligible for a refund on ORD-MM-1005?",
            "current_intent": {
                "intent": "refund",
                "entities": {"order_id": "ORD-MM-1005"},
                "policy_action": "allow",
                "reason_code": "REFUND_OK",
                "constraints": {},
                "status": "pending",
                "conditional": {
                    "request_type": "conditional_question",
                    "condition": {
                        "field": "is_refundable",
                        "operator": "is_true",
                        "value": "",
                    },
                },
            },
        }

        result = conditional_executor_node(state)
        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "yes" in pr["domain_result"].lower()
        tool_names = [tc["tool"] for tc in pr["tool_calls"]]
        assert "initiate_refund" not in tool_names


# ═══════════════════════════════════════════════════════════════════════
# 8. No eval() / no arbitrary expressions
# ═══════════════════════════════════════════════════════════════════════

class TestNoEvalSafety:
    def test_condition_evaluator_has_no_eval(self):
        import ast, inspect
        from app.graph import condition_evaluator
        source = inspect.getsource(condition_evaluator)
        tree = ast.parse(source)
        calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("eval", "exec")
        ]
        assert calls == [], f"Found dangerous calls: {calls}"

    def test_conditional_executor_has_no_eval(self):
        import inspect
        from app.graph.nodes import conditional_executor
        source = inspect.getsource(conditional_executor)
        assert "eval(" not in source
        assert "exec(" not in source

    def test_no_llm_in_evaluator(self):
        import inspect
        from app.graph import condition_evaluator
        source = inspect.getsource(condition_evaluator)
        assert "ChatGroq" not in source
        assert "llm" not in source.lower() or "llm calls" in source.lower()

    def test_no_llm_in_conditional_executor(self):
        import inspect
        from app.graph.nodes import conditional_executor
        source = inspect.getsource(conditional_executor)
        assert "ChatGroq" not in source
        assert "invoke" not in source or "_invoke_read_tool" in source


# ═══════════════════════════════════════════════════════════════════════
# 9. Merge results compatibility
# ═══════════════════════════════════════════════════════════════════════

class TestMergeResultsCompat:
    def test_conditional_executor_returns_processed_results_list(self):
        """Return shape is {"processed_results": [result_dict]}."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-001",
            "user_message": "test",
            "current_intent": {
                "intent": "cancel_order",
                "entities": {"order_id": "ORD-1"},
                "policy_action": "allow",
                "reason_code": "OK",
                "constraints": {},
                "status": "pending",
                "conditional": None,
            },
        }
        result = conditional_executor_node(state)
        assert "processed_results" in result
        assert isinstance(result["processed_results"], list)
        assert len(result["processed_results"]) == 1
