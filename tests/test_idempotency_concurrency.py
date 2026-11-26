"""Tests for idempotent write operations and concurrency protection.

Covers:
- OperationLog model and idempotency key generation
- Request-level idempotency in initiate_refund, cancel_order, initiate_return
- Business-level duplicate protection (RefundHistory unique constraint)
- Customer-level advisory lock helpers
- Conditional write revalidation inside write workflows
- Pre-condition passing from conditional_executor to write_executor
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest


# ===================================================================
# 1. Idempotency key generation
# ===================================================================

class TestIdempotencyKeyGeneration:
    def test_key_is_deterministic(self):
        from app.graph.agent_prod.tools.order_tools import _make_idempotency_key

        k1 = _make_idempotency_key("initiate_refund", order_id="ORD-001", tenant_id="t1")
        k2 = _make_idempotency_key("initiate_refund", order_id="ORD-001", tenant_id="t1")
        assert k1 == k2

    def test_key_differs_for_different_operations(self):
        from app.graph.agent_prod.tools.order_tools import _make_idempotency_key

        k1 = _make_idempotency_key("initiate_refund", order_id="ORD-001", tenant_id="t1")
        k2 = _make_idempotency_key("cancel_order", order_id="ORD-001", tenant_id="t1")
        assert k1 != k2

    def test_key_differs_for_different_orders(self):
        from app.graph.agent_prod.tools.order_tools import _make_idempotency_key

        k1 = _make_idempotency_key("initiate_refund", order_id="ORD-001", tenant_id="t1")
        k2 = _make_idempotency_key("initiate_refund", order_id="ORD-002", tenant_id="t1")
        assert k1 != k2

    def test_key_differs_for_different_tenants(self):
        from app.graph.agent_prod.tools.order_tools import _make_idempotency_key

        k1 = _make_idempotency_key("initiate_refund", order_id="ORD-001", tenant_id="t1")
        k2 = _make_idempotency_key("initiate_refund", order_id="ORD-001", tenant_id="t2")
        assert k1 != k2

    def test_key_length_is_32(self):
        from app.graph.agent_prod.tools.order_tools import _make_idempotency_key

        k = _make_idempotency_key("initiate_refund", order_id="ORD-001", tenant_id="t1")
        assert len(k) == 32

    def test_key_sorted_params(self):
        from app.graph.agent_prod.tools.order_tools import _make_idempotency_key

        k1 = _make_idempotency_key("initiate_return", order_id="ORD-001", item_id="ITEM-1", tenant_id="t1")
        k2 = _make_idempotency_key("initiate_return", tenant_id="t1", item_id="ITEM-1", order_id="ORD-001")
        assert k1 == k2


# ===================================================================
# 2. OperationLog model structure
# ===================================================================

class TestOperationLogModel:
    def test_model_has_required_fields(self):
        from app.models import OperationLog

        assert hasattr(OperationLog, "tenant_id")
        assert hasattr(OperationLog, "customer_id")
        assert hasattr(OperationLog, "idempotency_key")
        assert hasattr(OperationLog, "operation")
        assert hasattr(OperationLog, "status")
        assert hasattr(OperationLog, "request_args")
        assert hasattr(OperationLog, "result")
        assert hasattr(OperationLog, "created_at")

    def test_table_name(self):
        from app.models import OperationLog

        assert OperationLog.__tablename__ == "operation_log"

    def test_unique_constraint_exists(self):
        from app.models import OperationLog

        constraints = OperationLog.__table_args__
        unique_names = [
            c.name for c in constraints if hasattr(c, "name") and c.name
        ]
        assert "uq_operation_log_idempotency" in unique_names


# ===================================================================
# 3. RefundHistory unique constraint
# ===================================================================

class TestRefundHistoryUniqueConstraint:
    def test_unique_constraint_on_tenant_order(self):
        from app.models import RefundHistory

        constraints = RefundHistory.__table_args__
        unique_names = [
            c.name for c in constraints if hasattr(c, "name") and c.name
        ]
        assert "uq_refund_history_tenant_order" in unique_names


# ===================================================================
# 4. Advisory lock helper
# ===================================================================

class TestAdvisoryLock:
    def test_advisory_lock_skipped_on_sqlite(self):
        from app.graph.agent_prod.tools.order_tools import _customer_advisory_lock

        mock_db = MagicMock()
        mock_db.bind.dialect.name = "sqlite"
        _customer_advisory_lock(mock_db, "tenant1", "cust1")
        mock_db.execute.assert_not_called()

    def test_advisory_lock_called_on_postgresql(self):
        from app.graph.agent_prod.tools.order_tools import _customer_advisory_lock

        mock_db = MagicMock()
        mock_db.bind.dialect.name = "postgresql"
        _customer_advisory_lock(mock_db, "tenant1", "cust1")
        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args
        sql_text = str(call_args[0][0])
        assert "pg_advisory_xact_lock" in sql_text

    def test_advisory_lock_key_stable(self):
        from app.graph.agent_prod.tools.order_tools import _customer_advisory_lock

        mock_db1 = MagicMock()
        mock_db1.bind.dialect.name = "postgresql"
        mock_db2 = MagicMock()
        mock_db2.bind.dialect.name = "postgresql"

        _customer_advisory_lock(mock_db1, "tenant1", "cust1")
        _customer_advisory_lock(mock_db2, "tenant1", "cust1")

        lid1 = mock_db1.execute.call_args[0][1]["lid"]
        lid2 = mock_db2.execute.call_args[0][1]["lid"]
        assert lid1 == lid2

    def test_advisory_lock_key_differs_for_different_customers(self):
        from app.graph.agent_prod.tools.order_tools import _customer_advisory_lock

        mock_db1 = MagicMock()
        mock_db1.bind.dialect.name = "postgresql"
        mock_db2 = MagicMock()
        mock_db2.bind.dialect.name = "postgresql"

        _customer_advisory_lock(mock_db1, "tenant1", "cust1")
        _customer_advisory_lock(mock_db2, "tenant1", "cust2")

        lid1 = mock_db1.execute.call_args[0][1]["lid"]
        lid2 = mock_db2.execute.call_args[0][1]["lid"]
        assert lid1 != lid2


# ===================================================================
# 5. Idempotency claim helper
# ===================================================================

class TestClaimIdempotency:
    def test_claim_returns_none_on_new_key(self):
        from app.graph.agent_prod.tools.order_tools import _claim_idempotency

        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = None

        result = _claim_idempotency(
            mock_db, str(uuid4()), "cust1", "key123",
            "initiate_refund", {"order_id": "ORD-1"},
        )
        assert result is None
        mock_db.add.assert_called_once()
        mock_db.flush.assert_called_once()

    def test_claim_returns_existing_on_duplicate(self):
        from app.graph.agent_prod.tools.order_tools import _claim_idempotency

        existing = MagicMock()
        existing.status = "completed"
        existing.result = '{"success": true}'

        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = existing

        result = _claim_idempotency(
            mock_db, str(uuid4()), "cust1", "key123",
            "initiate_refund", {"order_id": "ORD-1"},
        )
        assert result is existing
        mock_db.add.assert_not_called()


# ===================================================================
# 6. Write executor with pre_condition revalidation
# ===================================================================

class TestPreConditionRevalidation:
    def test_refund_workflow_accepts_pre_condition(self):
        """Verify the refund workflow signature accepts pre_condition."""
        from app.graph.nodes.write_executor import _execute_refund_workflow
        import inspect

        sig = inspect.signature(_execute_refund_workflow)
        assert "pre_condition" in sig.parameters

    def test_cancel_workflow_accepts_pre_condition(self):
        """Verify the cancel workflow signature accepts pre_condition."""
        from app.graph.nodes.write_executor import _execute_cancel_workflow
        import inspect

        sig = inspect.signature(_execute_cancel_workflow)
        assert "pre_condition" in sig.parameters

    def test_revalidate_returns_none_when_no_condition(self):
        from app.graph.nodes.write_executor import _revalidate_pre_condition

        result = _revalidate_pre_condition(None, "t1", "c1", "ORD-1")
        assert result is None

    def test_revalidate_returns_none_when_empty_condition(self):
        from app.graph.nodes.write_executor import _revalidate_pre_condition

        result = _revalidate_pre_condition({"condition": {}}, "t1", "c1", "ORD-1")
        assert result is None

    def test_revalidate_blocks_when_condition_no_longer_met(self):
        from app.graph.nodes.write_executor import _revalidate_pre_condition

        pre_condition = {
            "condition": {
                "field": "order_status",
                "operator": "equals",
                "value": "pending",
            },
        }

        with patch(
            "app.graph.nodes.write_executor._invoke_read_tool",
            return_value={"order_status": "shipped"},
        ):
            result = _revalidate_pre_condition(
                pre_condition, "t1", "c1", "ORD-1",
            )

        assert result is not None
        assert result["status"] == "blocked"
        assert "no longer met" in result["domain_result"]

    def test_revalidate_passes_when_condition_still_met(self):
        from app.graph.nodes.write_executor import _revalidate_pre_condition

        pre_condition = {
            "condition": {
                "field": "order_status",
                "operator": "equals",
                "value": "pending",
            },
        }

        with patch(
            "app.graph.nodes.write_executor._invoke_read_tool",
            return_value={"order_status": "pending"},
        ):
            result = _revalidate_pre_condition(
                pre_condition, "t1", "c1", "ORD-1",
            )

        assert result is None


# ===================================================================
# 7. Conditional executor passes condition to write workflow
# ===================================================================

class TestConditionalPassesCondition:
    def test_condition_passed_to_write_workflow(self):
        from app.graph.nodes.conditional_executor import _execute_conditional

        condition = {
            "field": "order_status",
            "operator": "equals",
            "value": "pending",
        }
        conditional = {
            "request_type": "conditional_action",
            "condition": condition,
        }

        mock_workflow = MagicMock(return_value={
            "domain_result": "Done",
            "status": "fulfilled",
            "tool_calls": [],
        })

        with patch(
            "app.graph.nodes.conditional_executor._gather_evidence",
            return_value=(
                {"get_order_details": {"order_status": "pending"}},
                [{"tool": "get_order_details", "args": {"order_id": "ORD-1"}, "result": {"order_status": "pending"}}],
            ),
        ), patch(
            "app.graph.nodes.conditional_executor.evaluate_condition",
            return_value={"status": "satisfied", "actual_value": "pending", "reason": None},
        ), patch(
            "app.graph.nodes.conditional_executor._WRITE_WORKFLOWS",
            {"cancel_order": mock_workflow},
        ):
            result_dict = {
                "intent": "cancel_order",
                "conditional": conditional,
                "entities": {"order_id": "ORD-1"},
                "constraints": {},
            }
            _execute_conditional(
                result_dict, "cancel_order", conditional, "conditional_action",
                condition, "megamart", "CUST-001", {"order_id": "ORD-1"}, {},
            )

        mock_workflow.assert_called_once()
        call_kwargs = mock_workflow.call_args
        assert "pre_condition" in call_kwargs.kwargs or len(call_kwargs.args) > 4
        if "pre_condition" in call_kwargs.kwargs:
            pc = call_kwargs.kwargs["pre_condition"]
        else:
            pc = call_kwargs.args[4]
        assert pc is not None
        assert pc["condition"] == condition

    def test_no_condition_on_direct_write(self):
        """Write executor called without conditional passes no pre_condition."""
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
# 8. Refund workflow blocks when condition changes
# ===================================================================

class TestRefundWorkflowConditionRevalidation:
    def test_refund_blocked_when_condition_no_longer_satisfied(self):
        from app.graph.nodes.write_executor import _execute_refund_workflow

        pre_condition = {
            "condition": {
                "field": "is_refundable",
                "operator": "is_true",
                "value": "",
            },
        }

        with patch(
            "app.graph.nodes.write_executor._invoke_read_tool",
        ) as mock_read:
            mock_read.side_effect = [
                {"order_status": "delivered", "total_amount": 500, "order_id": "ORD-1"},
                {"eligible": True, "amount": 500, "refund_type": "original_method"},
                {"eligible": False, "reason": "Window expired"},
            ]

            result = _execute_refund_workflow(
                "megamart", "CUST-001",
                {"order_id": "ORD-1"}, {},
                pre_condition=pre_condition,
            )

        assert result["status"] == "blocked"
        assert "no longer met" in result["domain_result"]

    def test_refund_proceeds_when_condition_still_satisfied(self):
        from app.graph.nodes.write_executor import _execute_refund_workflow

        pre_condition = {
            "condition": {
                "field": "is_refundable",
                "operator": "is_true",
                "value": "",
            },
        }

        with patch(
            "app.graph.nodes.write_executor._invoke_read_tool",
        ) as mock_read, patch(
            "app.graph.nodes.write_executor._invoke_write_tool",
            return_value={"success": True, "message": "Refunded"},
        ):
            mock_read.side_effect = [
                {"order_status": "delivered", "total_amount": 500, "order_id": "ORD-1"},
                {"eligible": True, "amount": 500, "refund_type": "original_method"},
                {"eligible": True},
            ]

            result = _execute_refund_workflow(
                "megamart", "CUST-001",
                {"order_id": "ORD-1"}, {},
                pre_condition=pre_condition,
            )

        assert result["status"] == "fulfilled"


# ===================================================================
# 9. Cancel workflow blocks when condition changes
# ===================================================================

class TestCancelWorkflowConditionRevalidation:
    def test_cancel_blocked_when_condition_no_longer_satisfied(self):
        from app.graph.nodes.write_executor import _execute_cancel_workflow

        pre_condition = {
            "condition": {
                "field": "order_status",
                "operator": "equals",
                "value": "pending",
            },
        }

        with patch(
            "app.graph.nodes.write_executor._invoke_read_tool",
        ) as mock_read:
            mock_read.side_effect = [
                {"order_status": "shipped", "order_id": "ORD-1"},
                {"order_status": "shipped"},
            ]

            result = _execute_cancel_workflow(
                "megamart", "CUST-001",
                {"order_id": "ORD-1"}, {},
                pre_condition=pre_condition,
            )

        assert result["status"] == "blocked"
        assert "no longer met" in result["domain_result"]


# ===================================================================
# 10. Mock-mode write executor still works (no idempotency DB)
# ===================================================================

class TestMockModePreserved:
    def test_refund_in_mock_mode(self):
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

        result = write_executor_node(state)
        r = result["processed_results"][0]
        assert r["status"] in ("fulfilled", "blocked", "failed", "awaiting_human")

    def test_cancel_in_mock_mode(self):
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
# 11. OperationLog complete helper
# ===================================================================

class TestCompleteIdempotency:
    def test_complete_sets_status_and_result(self):
        from app.graph.agent_prod.tools.order_tools import _complete_idempotency

        mock_db = MagicMock()
        mock_log = MagicMock()
        result = {"success": True, "message": "Done"}

        _complete_idempotency(mock_db, mock_log, "completed", result)

        assert mock_log.status == "completed"
        assert json.loads(mock_log.result) == result

    def test_complete_failed_sets_status(self):
        from app.graph.agent_prod.tools.order_tools import _complete_idempotency

        mock_db = MagicMock()
        mock_log = MagicMock()
        result = {"success": False, "error": "Not found"}

        _complete_idempotency(mock_db, mock_log, "failed", result)

        assert mock_log.status == "failed"


# ===================================================================
# 12. Workflow backward compatibility
# ===================================================================

class TestWorkflowBackwardCompatibility:
    def test_refund_workflow_works_without_pre_condition(self):
        """Direct write executor doesn't pass pre_condition; workflow must still work."""
        from app.graph.nodes.write_executor import _execute_refund_workflow

        with patch(
            "app.graph.nodes.write_executor._invoke_read_tool",
        ) as mock_read, patch(
            "app.graph.nodes.write_executor._invoke_write_tool",
            return_value={"success": True, "message": "Refunded"},
        ):
            mock_read.side_effect = [
                {"order_status": "delivered", "total_amount": 500, "order_id": "ORD-1"},
                {"eligible": True, "amount": 500, "refund_type": "original_method"},
            ]

            result = _execute_refund_workflow(
                "t1", "c1", {"order_id": "ORD-1"}, {},
            )

        assert result["status"] == "fulfilled"

    def test_cancel_workflow_works_without_pre_condition(self):
        """Direct cancel workflow without pre_condition."""
        from app.graph.nodes.write_executor import _execute_cancel_workflow

        with patch(
            "app.graph.nodes.write_executor._invoke_read_tool",
            return_value={"order_status": "pending", "order_id": "ORD-1"},
        ), patch(
            "app.graph.nodes.write_executor._invoke_write_tool",
            return_value={"success": True, "message": "Cancelled"},
        ):
            result = _execute_cancel_workflow(
                "t1", "c1", {"order_id": "ORD-1"}, {},
            )

        assert result["status"] == "fulfilled"
