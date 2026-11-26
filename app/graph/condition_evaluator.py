"""Deterministic condition evaluator.

Evaluates structured conditions against authoritative read evidence.
Never uses eval(), arbitrary expressions, or LLM calls.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Supported fields and how to extract them from tool results
# ---------------------------------------------------------------------------

EVIDENCE_EXTRACTORS: dict[str, dict] = {
    "order_status": {
        "tool": "get_order_details",
        "result_key": "order_status",
    },
    "fulfillment_status": {
        "tool": "get_order_details",
        "result_key": "order_status",
    },
    "is_refundable": {
        "tool": "get_return_eligibility",
        "result_key": "eligible",
    },
    "is_return_eligible": {
        "tool": "get_return_eligibility",
        "result_key": "eligible",
    },
}

FIELD_ALLOWED_OPERATORS: dict[str, frozenset[str]] = {
    "order_status": frozenset({"equals", "not_equals", "in", "not_in"}),
    "fulfillment_status": frozenset({"equals", "not_equals", "in", "not_in"}),
    "is_refundable": frozenset({"is_true", "is_false"}),
    "is_return_eligible": frozenset({"is_true", "is_false"}),
}

SUPPORTED_FIELDS: frozenset[str] = frozenset(EVIDENCE_EXTRACTORS.keys())


# ---------------------------------------------------------------------------
# Which read tools each condition field requires
# ---------------------------------------------------------------------------

def required_tools_for_condition(condition: dict) -> list[str]:
    field = condition.get("field", "")
    extractor = EVIDENCE_EXTRACTORS.get(field)
    if not extractor:
        return []
    return [extractor["tool"]]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize(val: Any) -> str | bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    return str(val).strip().lower()


def _normalize_list(vals: str | list[str]) -> list[str]:
    if isinstance(vals, str):
        return [vals.strip().lower()]
    return [str(v).strip().lower() for v in vals]


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

class ConditionResult:
    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"
    UNKNOWN = "unknown"


def evaluate_condition(condition: dict, evidence: dict[str, dict]) -> dict:
    """Evaluate a structured condition against tool-result evidence.

    Args:
        condition: {"field": ..., "operator": ..., "value": ...}
        evidence: {"tool_name": tool_result_dict, ...}

    Returns:
        {"status": "satisfied"|"not_satisfied"|"unknown", "reason": str|None, "actual_value": ...}
    """
    field = condition.get("field", "")
    operator = condition.get("operator", "")
    value = condition.get("value", "")

    if field not in SUPPORTED_FIELDS:
        return {
            "status": ConditionResult.UNKNOWN,
            "reason": f"Unsupported condition field: '{field}'.",
            "actual_value": None,
        }

    allowed_ops = FIELD_ALLOWED_OPERATORS.get(field, frozenset())
    if operator not in allowed_ops:
        return {
            "status": ConditionResult.UNKNOWN,
            "reason": f"Operator '{operator}' is not supported for field '{field}'.",
            "actual_value": None,
        }

    extractor = EVIDENCE_EXTRACTORS[field]
    tool_name = extractor["tool"]
    result_key = extractor["result_key"]

    tool_result = evidence.get(tool_name)
    if tool_result is None:
        return {
            "status": ConditionResult.UNKNOWN,
            "reason": f"Required data from '{tool_name}' is not available.",
            "actual_value": None,
        }

    if tool_result.get("error"):
        return {
            "status": ConditionResult.UNKNOWN,
            "reason": f"Could not retrieve data: {tool_result['error']}",
            "actual_value": None,
        }

    actual_value = tool_result.get(result_key)
    if actual_value is None:
        return {
            "status": ConditionResult.UNKNOWN,
            "reason": f"Field '{result_key}' is not available in the data.",
            "actual_value": None,
        }

    return _apply_operator(operator, actual_value, value, field)


def _apply_operator(operator: str, actual: Any, expected: Any, field: str) -> dict:
    actual_norm = _normalize(actual)

    if operator == "equals":
        expected_norm = _normalize(expected)
        satisfied = actual_norm == expected_norm
    elif operator == "not_equals":
        expected_norm = _normalize(expected)
        satisfied = actual_norm != expected_norm
    elif operator == "in":
        expected_list = _normalize_list(expected)
        satisfied = actual_norm in expected_list
    elif operator == "not_in":
        expected_list = _normalize_list(expected)
        satisfied = actual_norm not in expected_list
    elif operator == "is_true":
        satisfied = bool(actual)
    elif operator == "is_false":
        satisfied = not bool(actual)
    else:
        return {
            "status": ConditionResult.UNKNOWN,
            "reason": f"Unsupported operator: '{operator}'.",
            "actual_value": actual,
        }

    if satisfied:
        return {
            "status": ConditionResult.SATISFIED,
            "reason": None,
            "actual_value": actual,
        }
    else:
        return {
            "status": ConditionResult.NOT_SATISFIED,
            "reason": _build_not_satisfied_reason(field, operator, expected, actual),
            "actual_value": actual,
        }


def _build_not_satisfied_reason(field: str, operator: str, expected: Any, actual: Any) -> str:
    field_label = field.replace("_", " ")
    if operator == "equals":
        return f"The {field_label} is '{actual}', not '{expected}'."
    elif operator == "not_equals":
        return f"The {field_label} is '{actual}'."
    elif operator == "in":
        return f"The {field_label} is '{actual}', which is not in the expected values."
    elif operator == "not_in":
        return f"The {field_label} is '{actual}', which matches the excluded values."
    elif operator == "is_true":
        return f"The {field_label} check returned false."
    elif operator == "is_false":
        return f"The {field_label} check returned true."
    return f"Condition on {field_label} was not satisfied (actual: {actual})."
