"""Tests for condition_evaluator module.

Covers: all supported operators, field normalizers, evidence extraction,
missing evidence, tool errors, unsupported fields and operators.
"""
from __future__ import annotations

import pytest

from app.graph.condition_evaluator import (
    EVIDENCE_EXTRACTORS,
    FIELD_ALLOWED_OPERATORS,
    SUPPORTED_FIELDS,
    ConditionResult,
    _normalize,
    _normalize_list,
    evaluate_condition,
    required_tools_for_condition,
)


# ===================================================================
# _normalize
# ===================================================================

class TestNormalize:
    def test_none_returns_none(self):
        assert _normalize(None) is None

    def test_bool_true_returns_true(self):
        assert _normalize(True) is True

    def test_bool_false_returns_false(self):
        assert _normalize(False) is False

    def test_string_lowercased_and_stripped(self):
        assert _normalize("  Shipped  ") == "shipped"

    def test_int_converted_to_string(self):
        assert _normalize(42) == "42"


# ===================================================================
# _normalize_list
# ===================================================================

class TestNormalizeList:
    def test_string_becomes_single_item_list(self):
        assert _normalize_list("Shipped") == ["shipped"]

    def test_list_normalized(self):
        assert _normalize_list(["Pending", " SHIPPED "]) == ["pending", "shipped"]

    def test_integers_in_list(self):
        assert _normalize_list([1, 2]) == ["1", "2"]


# ===================================================================
# required_tools_for_condition
# ===================================================================

class TestRequiredToolsForCondition:
    def test_order_status_requires_get_order_details(self):
        tools = required_tools_for_condition({"field": "order_status"})
        assert tools == ["get_order_details"]

    def test_is_refundable_requires_get_return_eligibility(self):
        tools = required_tools_for_condition({"field": "is_refundable"})
        assert tools == ["get_return_eligibility"]

    def test_unsupported_field_returns_empty(self):
        tools = required_tools_for_condition({"field": "nonexistent_field"})
        assert tools == []

    def test_empty_field_returns_empty(self):
        tools = required_tools_for_condition({})
        assert tools == []


# ===================================================================
# evaluate_condition — operator coverage
# ===================================================================

class TestEvaluateConditionOperators:
    def test_equals_satisfied(self):
        result = evaluate_condition(
            {"field": "order_status", "operator": "equals", "value": "shipped"},
            {"get_order_details": {"order_status": "shipped"}},
        )
        assert result["status"] == ConditionResult.SATISFIED
        assert result["actual_value"] == "shipped"
        assert result["reason"] is None

    def test_equals_not_satisfied(self):
        result = evaluate_condition(
            {"field": "order_status", "operator": "equals", "value": "shipped"},
            {"get_order_details": {"order_status": "pending"}},
        )
        assert result["status"] == ConditionResult.NOT_SATISFIED
        assert result["actual_value"] == "pending"
        assert "pending" in result["reason"]

    def test_equals_case_insensitive(self):
        result = evaluate_condition(
            {"field": "order_status", "operator": "equals", "value": "SHIPPED"},
            {"get_order_details": {"order_status": "shipped"}},
        )
        assert result["status"] == ConditionResult.SATISFIED

    def test_not_equals_satisfied(self):
        result = evaluate_condition(
            {"field": "order_status", "operator": "not_equals", "value": "shipped"},
            {"get_order_details": {"order_status": "pending"}},
        )
        assert result["status"] == ConditionResult.SATISFIED

    def test_not_equals_not_satisfied(self):
        result = evaluate_condition(
            {"field": "order_status", "operator": "not_equals", "value": "shipped"},
            {"get_order_details": {"order_status": "shipped"}},
        )
        assert result["status"] == ConditionResult.NOT_SATISFIED

    def test_in_operator_satisfied(self):
        result = evaluate_condition(
            {"field": "order_status", "operator": "in", "value": ["pending", "processing"]},
            {"get_order_details": {"order_status": "pending"}},
        )
        assert result["status"] == ConditionResult.SATISFIED

    def test_in_operator_not_satisfied(self):
        result = evaluate_condition(
            {"field": "order_status", "operator": "in", "value": ["pending", "processing"]},
            {"get_order_details": {"order_status": "shipped"}},
        )
        assert result["status"] == ConditionResult.NOT_SATISFIED

    def test_not_in_operator_satisfied(self):
        result = evaluate_condition(
            {"field": "order_status", "operator": "not_in", "value": ["shipped", "delivered"]},
            {"get_order_details": {"order_status": "pending"}},
        )
        assert result["status"] == ConditionResult.SATISFIED

    def test_not_in_operator_not_satisfied(self):
        result = evaluate_condition(
            {"field": "order_status", "operator": "not_in", "value": ["shipped", "delivered"]},
            {"get_order_details": {"order_status": "shipped"}},
        )
        assert result["status"] == ConditionResult.NOT_SATISFIED

    def test_is_true_satisfied(self):
        result = evaluate_condition(
            {"field": "is_refundable", "operator": "is_true", "value": ""},
            {"get_return_eligibility": {"eligible": True}},
        )
        assert result["status"] == ConditionResult.SATISFIED

    def test_is_true_not_satisfied(self):
        result = evaluate_condition(
            {"field": "is_refundable", "operator": "is_true", "value": ""},
            {"get_return_eligibility": {"eligible": False}},
        )
        assert result["status"] == ConditionResult.NOT_SATISFIED

    def test_is_false_satisfied(self):
        result = evaluate_condition(
            {"field": "is_refundable", "operator": "is_false", "value": ""},
            {"get_return_eligibility": {"eligible": False}},
        )
        assert result["status"] == ConditionResult.SATISFIED

    def test_is_false_not_satisfied(self):
        result = evaluate_condition(
            {"field": "is_refundable", "operator": "is_false", "value": ""},
            {"get_return_eligibility": {"eligible": True}},
        )
        assert result["status"] == ConditionResult.NOT_SATISFIED

    def test_in_with_string_value(self):
        """in operator with a bare string wraps it in a list."""
        result = evaluate_condition(
            {"field": "order_status", "operator": "in", "value": "pending"},
            {"get_order_details": {"order_status": "pending"}},
        )
        assert result["status"] == ConditionResult.SATISFIED


# ===================================================================
# evaluate_condition — error/unknown paths
# ===================================================================

class TestEvaluateConditionUnknown:
    def test_unsupported_field(self):
        result = evaluate_condition(
            {"field": "bogus_field", "operator": "equals", "value": "x"},
            {},
        )
        assert result["status"] == ConditionResult.UNKNOWN
        assert "Unsupported" in result["reason"]

    def test_unsupported_operator_for_field(self):
        result = evaluate_condition(
            {"field": "order_status", "operator": "is_true", "value": ""},
            {"get_order_details": {"order_status": "shipped"}},
        )
        assert result["status"] == ConditionResult.UNKNOWN
        assert "not supported" in result["reason"]

    def test_missing_evidence_tool(self):
        result = evaluate_condition(
            {"field": "order_status", "operator": "equals", "value": "shipped"},
            {},
        )
        assert result["status"] == ConditionResult.UNKNOWN
        assert "not available" in result["reason"]

    def test_tool_returned_error(self):
        result = evaluate_condition(
            {"field": "order_status", "operator": "equals", "value": "shipped"},
            {"get_order_details": {"error": "Order not found"}},
        )
        assert result["status"] == ConditionResult.UNKNOWN
        assert "Order not found" in result["reason"]

    def test_result_key_missing(self):
        result = evaluate_condition(
            {"field": "order_status", "operator": "equals", "value": "shipped"},
            {"get_order_details": {"some_other_key": "value"}},
        )
        assert result["status"] == ConditionResult.UNKNOWN
        assert "not available" in result["reason"]

    def test_empty_field(self):
        result = evaluate_condition(
            {"field": "", "operator": "equals", "value": "x"},
            {},
        )
        assert result["status"] == ConditionResult.UNKNOWN


# ===================================================================
# EVIDENCE_EXTRACTORS / FIELD_ALLOWED_OPERATORS consistency
# ===================================================================

class TestExtractorConsistency:
    def test_all_supported_fields_have_extractors(self):
        for field in SUPPORTED_FIELDS:
            assert field in EVIDENCE_EXTRACTORS

    def test_all_supported_fields_have_allowed_operators(self):
        for field in SUPPORTED_FIELDS:
            assert field in FIELD_ALLOWED_OPERATORS
            assert len(FIELD_ALLOWED_OPERATORS[field]) > 0

    def test_fulfillment_status_aliases_order_status(self):
        """fulfillment_status uses the same tool and key as order_status."""
        assert EVIDENCE_EXTRACTORS["fulfillment_status"]["tool"] == EVIDENCE_EXTRACTORS["order_status"]["tool"]
        assert EVIDENCE_EXTRACTORS["fulfillment_status"]["result_key"] == EVIDENCE_EXTRACTORS["order_status"]["result_key"]
