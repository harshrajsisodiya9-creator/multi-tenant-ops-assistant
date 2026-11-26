"""Regression tests for write execution, HITL reliability, and error handling fixes.

Covers:
- Cancellation failure status
- Missing order / DB failure status
- Ineligible refund status
- Cross-customer verification scoping
- HITL approval with stale order state
- HITL modified operation parameters
- Mock/production tool selection in HITL
- HITL write verification routing
- Error isolation across parallel executors
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


# ===================================================================
# 1. Cancellation failure returns "failed", not "fulfilled"
# ===================================================================

class TestCancelFailureStatus:
    def test_cancel_non_cancellable_returns_failed(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "cancel ORD-MM-1001",
            "current_intent": {
                "intent": "cancel_order",
                "entities": {"order_id": "ORD-MM-1001"},
                "policy_action": "allow",
                "reason_code": "CANCEL_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        r = result["processed_results"][0]
        assert r["status"] == "failed", (
            f"Non-cancellable order should produce 'failed', got '{r['status']}'"
        )
        assert "cannot be cancelled" in r["domain_result"].lower()

    def test_cancel_success_still_fulfilled(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-006",
            "user_message": "cancel ORD-MM-1007",
            "current_intent": {
                "intent": "cancel_order",
                "entities": {"order_id": "ORD-MM-1007"},
                "policy_action": "allow",
                "reason_code": "CANCEL_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        r = result["processed_results"][0]
        assert r["status"] == "fulfilled"


# ===================================================================
# 2. Missing order and database failure statuses
# ===================================================================

class TestOrderLookupFailure:
    def test_refund_order_not_found_returns_failed(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "refund ORD-FAKE",
            "current_intent": {
                "intent": "refund",
                "entities": {"order_id": "ORD-FAKE"},
                "policy_action": "allow",
                "reason_code": "REFUND_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        r = result["processed_results"][0]
        assert r["status"] == "failed"
        assert "not found" in r["domain_result"].lower()

    def test_cancel_order_not_found_returns_failed(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "cancel ORD-FAKE",
            "current_intent": {
                "intent": "cancel_order",
                "entities": {"order_id": "ORD-FAKE"},
                "policy_action": "allow",
                "reason_code": "CANCEL_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        r = result["processed_results"][0]
        assert r["status"] == "failed"
        assert "not found" in r["domain_result"].lower()

    def test_refund_not_eligible_returns_blocked(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-003",
            "user_message": "refund ORD-MM-1003",
            "current_intent": {
                "intent": "refund",
                "entities": {"order_id": "ORD-MM-1003"},
                "policy_action": "allow",
                "reason_code": "REFUND_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        result = write_executor_node(state)
        r = result["processed_results"][0]
        assert r["status"] == "blocked"

    def test_database_exception_returns_failed(self):
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "refund ORD-MM-1001",
            "current_intent": {
                "intent": "refund",
                "entities": {"order_id": "ORD-MM-1001"},
                "policy_action": "allow",
                "reason_code": "REFUND_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        with patch(
            "app.graph.nodes.write_executor._invoke_read_tool",
            side_effect=RuntimeError("DB connection lost"),
        ):
            result = write_executor_node(state)
            r = result["processed_results"][0]
            assert r["status"] == "failed"
            assert "unexpected error" in r["domain_result"].lower()


# ===================================================================
# 3. Cross-customer verification scoping
# ===================================================================

class TestVerificationScoping:
    def test_verify_cancel_requires_customer_id(self):
        """_verify_cancel signature now requires customer_id."""
        from app.graph.nodes.action_check import _verify_cancel
        import inspect

        sig = inspect.signature(_verify_cancel)
        params = list(sig.parameters.keys())
        assert "customer_id" in params

    def test_get_delivery_date_requires_customer_id(self):
        from app.graph.nodes.action_check import _get_delivery_date
        import inspect

        sig = inspect.signature(_get_delivery_date)
        params = list(sig.parameters.keys())
        assert "customer_id" in params

    def test_get_shipping_status_requires_customer_id(self):
        from app.graph.nodes.action_check import _get_shipping_status
        import inspect

        sig = inspect.signature(_get_shipping_status)
        params = list(sig.parameters.keys())
        assert "customer_id" in params

    def test_audit_tool_calls_requires_customer_id(self):
        from app.graph.nodes.action_check import _audit_tool_calls
        import inspect

        sig = inspect.signature(_audit_tool_calls)
        params = list(sig.parameters.keys())
        assert "customer_id" in params


# ===================================================================
# 4. HITL approval with stale order state
# ===================================================================

class TestHITLStaleState:
    def test_approve_stale_cancel(self):
        """An order that is no longer cancellable should be rejected on approval."""
        from app.graph.nodes.escalation_handler import escalation_handler_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "cancel_order",
                    "entities": {"order_id": "ORD-MM-1001"},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_CANCEL_REVIEW_REQUIRED",
                    "domain_result": "Cancel requires review.",
                    "pending_tool_call": {
                        "tool": "cancel_order",
                        "args": {"order_id": "ORD-MM-1001"},
                        "reasons": ["TENANT_CANCEL_REVIEW_REQUIRED"],
                    },
                },
            ],
        }
        human_response = {"decisions": [{"index": 0, "action": "approve"}]}

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        r = result["intent_results"][0]
        assert r["status"] == "failed"
        assert r["reason_code"] == "STALE_APPROVAL"
        assert "can no longer be cancelled" in r["domain_result"].lower()

    def test_approve_missing_order(self):
        """An order that no longer exists should be rejected on approval."""
        from app.graph.nodes.escalation_handler import escalation_handler_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-FAKE"},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "domain_result": "Refund requires review.",
                    "pending_tool_call": {
                        "tool": "initiate_refund",
                        "args": {"order_id": "ORD-FAKE", "amount": 1000, "refund_type": "original_method"},
                        "reasons": ["TENANT_REFUND_REVIEW_REQUIRED"],
                    },
                },
            ],
        }
        human_response = {"decisions": [{"index": 0, "action": "approve"}]}

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        r = result["intent_results"][0]
        assert r["status"] == "failed"
        assert r["reason_code"] == "STALE_APPROVAL"
        assert "no longer accessible" in r["domain_result"].lower()


# ===================================================================
# 5. Modified operation parameters after approval
# ===================================================================

class TestHITLModifiedParameters:
    def test_approve_refund_amount_changed(self):
        """If the refund amount no longer matches, approval should be rejected."""
        from app.graph.nodes.escalation_handler import escalation_handler_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-MM-1005"},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "domain_result": "Refund requires review.",
                    "pending_tool_call": {
                        "tool": "initiate_refund",
                        "args": {"order_id": "ORD-MM-1005", "amount": 99999, "refund_type": "original_method"},
                        "reasons": ["TENANT_REFUND_REVIEW_REQUIRED"],
                    },
                },
            ],
        }
        human_response = {"decisions": [{"index": 0, "action": "approve"}]}

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        r = result["intent_results"][0]
        assert r["status"] == "failed"
        assert r["reason_code"] == "STALE_APPROVAL"
        assert "amount has changed" in r["domain_result"].lower()


# ===================================================================
# 6. Mock/production tool selection in HITL
# ===================================================================

class TestHITLToolSelection:
    def test_execute_pending_tool_uses_mode_aware_factory(self):
        """_execute_pending_tool should NOT hardcode prod imports."""
        import inspect
        from app.graph.nodes.escalation_handler import _execute_pending_tool

        source = inspect.getsource(_execute_pending_tool)
        assert "agent_prod.tools" not in source, (
            "_execute_pending_tool still hardcodes agent_prod imports"
        )

    def test_execute_pending_tool_uses_invoke_write_tool(self):
        """_execute_pending_tool should delegate to _invoke_write_tool."""
        import inspect
        from app.graph.nodes.escalation_handler import _execute_pending_tool

        source = inspect.getsource(_execute_pending_tool)
        assert "_invoke_write_tool" in source

    def test_approve_with_pending_tool_in_mock_mode(self):
        """Approval should work in mock mode (default for tests)."""
        from app.graph.nodes.escalation_handler import escalation_handler_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-006",
            "intent_results": [
                {
                    "intent": "cancel_order",
                    "entities": {"order_id": "ORD-MM-1007"},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_CANCEL_REVIEW_REQUIRED",
                    "domain_result": "Cancel requires review.",
                    "pending_tool_call": {
                        "tool": "cancel_order",
                        "args": {"order_id": "ORD-MM-1007"},
                        "reasons": ["TENANT_CANCEL_REVIEW_REQUIRED"],
                    },
                },
            ],
        }
        human_response = {"decisions": [{"index": 0, "action": "approve"}]}

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        r = result["intent_results"][0]
        assert r["status"] == "fulfilled"
        assert r["_newly_resolved"] is True
        assert any(tc["tool"] == "cancel_order" for tc in r.get("tool_calls", []))


# ===================================================================
# 7. HITL write verification routing
# ===================================================================

class TestHITLActionCheckRouting:
    def test_after_escalation_routes_to_hitl_action_check(self):
        """When HITL produces tool_calls, graph should route through hitl_action_check."""
        from app.graph.builder import after_escalation

        state = {
            "response_phase": "follow_up",
            "intent_results": [
                {
                    "intent": "cancel_order",
                    "status": "fulfilled",
                    "_newly_resolved": True,
                    "tool_calls": [{"tool": "cancel_order", "args": {"order_id": "ORD-1"}, "result": {}}],
                },
            ],
        }
        assert after_escalation(state) == "hitl_action_check"

    def test_after_escalation_skips_action_check_when_no_tool_calls(self):
        """HITL without tool_calls (annotation-only approve) goes to follow_up."""
        from app.graph.builder import after_escalation

        state = {
            "response_phase": "follow_up",
            "intent_results": [
                {
                    "intent": "refund",
                    "status": "fulfilled",
                    "_newly_resolved": True,
                },
            ],
        }
        assert after_escalation(state) == "follow_up"

    def test_hitl_action_check_node_exists_in_graph(self):
        from app.graph.builder import build_graph

        graph = build_graph()
        node_names = set(graph.get_graph().nodes)
        assert "hitl_action_check" in node_names


# ===================================================================
# 8. Error isolation across parallel executors
# ===================================================================

class TestErrorIsolation:
    def test_write_executor_exception_produces_failed_result(self):
        """An exception in the workflow should produce a structured failed result."""
        from app.graph.nodes.write_executor import write_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "refund ORD-MM-1001",
            "current_intent": {
                "intent": "refund",
                "entities": {"order_id": "ORD-MM-1001"},
                "policy_action": "allow",
                "reason_code": "REFUND_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
            },
        }

        with patch(
            "app.graph.nodes.write_executor._invoke_read_tool",
            side_effect=Exception("unexpected crash"),
        ):
            result = write_executor_node(state)
            r = result["processed_results"][0]
            assert r["status"] == "failed"
            assert "processed_results" in result

    def test_conditional_executor_exception_produces_failed_result(self):
        """An exception in the conditional executor should produce a structured failed result."""
        from app.graph.nodes.conditional_executor import conditional_executor_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "user_message": "if order is pending, cancel it",
            "current_intent": {
                "intent": "cancel_order",
                "entities": {"order_id": "ORD-MM-1001"},
                "policy_action": "allow",
                "reason_code": "CANCEL_ALLOWED",
                "constraints": {},
                "domain_result": "",
                "status": "pending",
                "conditional": {
                    "request_type": "conditional_action",
                    "condition": {
                        "field": "order_status",
                        "operator": "equals",
                        "value": "pending",
                    },
                },
            },
        }

        with patch(
            "app.graph.nodes.conditional_executor._gather_evidence",
            side_effect=RuntimeError("evidence gathering failed"),
        ):
            result = conditional_executor_node(state)
            r = result["processed_results"][0]
            assert r["status"] == "failed"
            assert "unexpected error" in r["domain_result"].lower()

    def test_action_check_exception_does_not_crash(self):
        """An exception during verification should not crash action_check_node."""
        from app.graph.nodes.action_check import action_check_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "refund",
                    "status": "fulfilled",
                    "domain_result": "Refund processed.",
                    "entities": {"order_id": "ORD-MM-1001"},
                    "tool_calls": [
                        {
                            "tool": "initiate_refund",
                            "args": {"order_id": "ORD-MM-1001", "amount": 4500},
                        }
                    ],
                },
                {
                    "intent": "order_status",
                    "status": "fulfilled",
                    "domain_result": "Shipped.",
                    "entities": {"order_id": "ORD-MM-1002"},
                    "tool_calls": [],
                },
            ],
        }

        with patch(
            "app.graph.nodes.action_check._audit_tool_calls",
            side_effect=RuntimeError("audit crash"),
        ):
            result = action_check_node(state)
            assert len(result["intent_results"]) == 2
            assert result["intent_results"][1]["status"] == "fulfilled"


# ===================================================================
# 9. HITL revalidation helper
# ===================================================================

class TestRevalidatePending:
    def test_revalidate_cancel_non_cancellable(self):
        from app.graph.nodes.escalation_handler import _revalidate_pending

        pending = {"tool": "cancel_order", "args": {"order_id": "ORD-MM-1001"}}
        error = _revalidate_pending(pending, "megamart", "CUST-MM-001")
        assert error is not None
        assert "can no longer be cancelled" in error.lower()

    def test_revalidate_cancel_cancellable(self):
        from app.graph.nodes.escalation_handler import _revalidate_pending

        pending = {"tool": "cancel_order", "args": {"order_id": "ORD-MM-1007"}}
        error = _revalidate_pending(pending, "megamart", "CUST-MM-006")
        assert error is None

    def test_revalidate_refund_missing_order(self):
        from app.graph.nodes.escalation_handler import _revalidate_pending

        pending = {
            "tool": "initiate_refund",
            "args": {"order_id": "ORD-FAKE", "amount": 1000, "refund_type": "original_method"},
        }
        error = _revalidate_pending(pending, "megamart", "CUST-MM-001")
        assert error is not None
        assert "no longer accessible" in error.lower()

    def test_revalidate_no_order_id_passes(self):
        from app.graph.nodes.escalation_handler import _revalidate_pending

        pending = {"tool": "cancel_order", "args": {}}
        error = _revalidate_pending(pending, "megamart", "CUST-MM-001")
        assert error is None


# ===================================================================
# 10. Existing HITL behaviors preserved
# ===================================================================

class TestHITLPreserved:
    def test_deny_still_works(self):
        from app.graph.nodes.escalation_handler import escalation_handler_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-MM-1001"},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "domain_result": "Refund requires review.",
                },
            ],
        }
        human_response = {
            "decisions": [{"index": 0, "action": "deny", "reason": "Not eligible"}]
        }

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        assert result["intent_results"][0]["status"] == "blocked"
        assert result["intent_results"][0]["reason_code"] == "HUMAN_DENIED"

    def test_override_still_works(self):
        from app.graph.nodes.escalation_handler import escalation_handler_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-MM-1001"},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "domain_result": "Refund requires review.",
                },
            ],
        }
        human_response = {
            "decisions": [
                {"index": 0, "action": "override", "custom_response": "Special credit applied."}
            ]
        }

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        assert result["intent_results"][0]["status"] == "fulfilled"
        assert "Special credit applied" in result["intent_results"][0]["domain_result"]

    def test_approve_without_pending_tool_annotates(self):
        from app.graph.nodes.escalation_handler import escalation_handler_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-MM-1001"},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "domain_result": "Refund requires review.",
                },
            ],
        }
        human_response = {"decisions": [{"index": 0, "action": "approve"}]}

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        assert result["intent_results"][0]["status"] == "fulfilled"
        assert "Approved by support team" in result["intent_results"][0]["domain_result"]

    def test_no_escalations_passthrough(self):
        from app.graph.nodes.escalation_handler import escalation_handler_node

        state = {
            "intent_results": [
                {"intent": "faq", "status": "fulfilled", "domain_result": "answer"},
            ],
        }
        result = escalation_handler_node(state)
        assert result == {}

    def test_response_phase_follow_up(self):
        from app.graph.nodes.escalation_handler import escalation_handler_node

        state = {
            "tenant_id": "megamart",
            "customer_id": "CUST-MM-001",
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {},
                    "status": "awaiting_human",
                    "reason_code": "TENANT_REFUND_REVIEW_REQUIRED",
                    "domain_result": "Review needed.",
                },
            ],
        }
        human_response = {"decisions": [{"index": 0, "action": "deny", "reason": "No"}]}

        with patch(
            "app.graph.nodes.escalation_handler.interrupt",
            return_value=human_response,
        ):
            result = escalation_handler_node(state)

        assert result["response_phase"] == "follow_up"
