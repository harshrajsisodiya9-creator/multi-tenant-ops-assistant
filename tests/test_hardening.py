"""Step 6 hardening tests — 8 tests covering order lock, allowlist,
double refund, cancel shipped, FOR UPDATE, state change policy re-run,
write tool check registry, and factory guard."""
from __future__ import annotations

import importlib
from datetime import date
from unittest.mock import MagicMock, patch
from uuid import uuid4, UUID as _UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from langchain_core.messages import AIMessage

from app.database import Base
from app.graph.agent_prod.react_agent import (
    INTENT_TOOLS,
    WRITE_TOOLS,
    _hard_constraint_check,
    _needs_approval,
    run_react_agent,
)
from app.graph.agent_prod.tool_context import ToolContext
from app.models import Customer, Order, OrderItem, RefundHistory, Tenant

TENANT_ID = str(uuid4())
CUSTOMER_A = "CUST-H-001"
CUSTOMER_B = "CUST-H-002"


@pytest.fixture()
def prod_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    tables = [
        Tenant.__table__, Customer.__table__, Order.__table__,
        OrderItem.__table__, RefundHistory.__table__,
    ]
    Base.metadata.create_all(engine, tables=tables)
    DBSession = sessionmaker(bind=engine)

    db = DBSession()
    db.add(Tenant(id=_UUID(TENANT_ID), name="megamart", api_key_hash="x"))
    db.add(Customer(customer_id=CUSTOMER_A, tenant_id=_UUID(TENANT_ID), email="a@test.com", name="A"))
    db.add(Customer(customer_id=CUSTOMER_B, tenant_id=_UUID(TENANT_ID), email="b@test.com", name="B"))
    db.flush()
    db.add(Order(
        order_id="ORD-H-001", tenant_id=_UUID(TENANT_ID),
        customer_id=CUSTOMER_A, customer_email="a@test.com",
        product="Widget", amount=2000, total_amount=2000,
        status="delivered", category="electronics",
        delivery_date=date.today(),
    ))
    db.add(Order(
        order_id="ORD-H-002", tenant_id=_UUID(TENANT_ID),
        customer_id=CUSTOMER_A, customer_email="a@test.com",
        product="Gadget", amount=500, total_amount=500,
        status="shipped", category="electronics",
        delivery_date=None,
    ))
    db.flush()
    db.add(OrderItem(
        item_id="ITEM-H-001-1", order_id="ORD-H-001",
        product="Widget", category="electronics", amount=2000, status="delivered",
    ))
    db.commit()
    db.close()

    yield DBSession


# ── 1. Order lock ────────────────────────────────────────────────────

class TestOrderLock:
    def test_tool_call_different_order_blocked(self):
        """LLM calls a tool targeting ORD-999 but agent was assigned ORD-001."""
        ctx = ToolContext(tenant_id=TENANT_ID, customer_id=CUSTOMER_A)
        mock_llm = MagicMock()
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "get_order_details", "args": {"order_id": "ORD-999"}, "id": "tc1"}],
        )
        final_msg = AIMessage(content="Blocked.")
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm.invoke = MagicMock(side_effect=[ai_msg, final_msg])

        result = run_react_agent(
            ctx=ctx, order_id="ORD-H-001",
            user_message="check order", intent="refund", llm=mock_llm,
        )
        blocked = [tc for tc in result["tool_calls"] if tc.get("blocked")]
        assert len(blocked) == 1
        assert "order lock" in blocked[0]["reason"]


# ── 2. Allowlist ─────────────────────────────────────────────────────

class TestAllowlist:
    def test_tool_not_in_intent_allowlist_blocked(self):
        """cancel_order is not in the refund allowlist."""
        ctx = ToolContext(tenant_id=TENANT_ID, customer_id=CUSTOMER_A)
        mock_llm = MagicMock()
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "cancel_order", "args": {"order_id": "ORD-H-001"}, "id": "tc1"}],
        )
        final_msg = AIMessage(content="Cannot cancel.")
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm.invoke = MagicMock(side_effect=[ai_msg, final_msg])

        result = run_react_agent(
            ctx=ctx, order_id="ORD-H-001",
            user_message="cancel", intent="refund", llm=mock_llm,
        )
        blocked = [tc for tc in result["tool_calls"] if tc.get("blocked")]
        assert len(blocked) == 1
        assert "not in allowlist" in blocked[0]["reason"]


# ── 3. Double refund ─────────────────────────────────────────────────

class TestDoubleRefund:
    def test_second_refund_same_order_rejected(self, prod_db):
        from app.graph.agent_prod.tools.order_tools import initiate_refund

        ctx = ToolContext(tenant_id=TENANT_ID, customer_id=CUSTOMER_A)
        with patch("app.graph.agent_prod.tools.order_tools._get_db", return_value=prod_db()):
            first = initiate_refund(ctx, "ORD-H-001", 2000)
        assert first["success"] is True

        with patch("app.graph.agent_prod.tools.order_tools._get_db", return_value=prod_db()):
            second = initiate_refund(ctx, "ORD-H-001", 2000)
        assert second["success"] is False
        assert "already exists" in second["error"].lower()


# ── 4. Cancel shipped ───────────────────────────────────────────────

class TestCancelShipped:
    def test_cancel_shipped_order_rejected(self, prod_db):
        from app.graph.agent_prod.tools.order_tools import cancel_order

        ctx = ToolContext(tenant_id=TENANT_ID, customer_id=CUSTOMER_A)
        with patch("app.graph.agent_prod.tools.order_tools._get_db", return_value=prod_db()):
            result = cancel_order(ctx, "ORD-H-002")
        assert result["success"] is False
        assert "cannot be cancelled" in result["error"].lower()


# ── 5. FOR UPDATE applied on write tools ─────────────────────────────

class TestForUpdate:
    def test_find_order_for_update_flag(self, prod_db):
        """Verify _find_order accepts for_update and returns the order."""
        from app.graph.agent_prod.tools.order_tools import _find_order

        db = prod_db()
        ctx = ToolContext(tenant_id=TENANT_ID, customer_id=CUSTOMER_A)
        order = _find_order(db, ctx, "ORD-H-001", for_update=True)
        assert order is not None
        assert order.order_id == "ORD-H-001"
        db.close()


# ── 6. State change — policy re-run catches stale gate decision ──────

class TestPolicyReRun:
    def test_refund_blocked_by_policy_recheck(self, prod_db):
        """Gate allowed refund for electronics, but between gate and write
        the category changed to perishables (returns not allowed).
        The write tool's policy re-run catches the change."""
        from app.graph.agent_prod.tools.order_tools import initiate_refund
        from app.graph.policy_models import PolicyRule, ReturnPolicy

        db = prod_db()
        order = db.query(Order).filter(Order.order_id == "ORD-H-001").one()
        order.category = "perishables"
        db.commit()
        db.close()

        blocked_policy = PolicyRule(
            return_policy=ReturnPolicy(allowed=False),
        )

        ctx = ToolContext(tenant_id=TENANT_ID, customer_id=CUSTOMER_A)
        with (
            patch("app.graph.agent_prod.tools.order_tools._get_db", return_value=prod_db()),
            patch("app.graph.agent_prod.tools.order_tools.resolve_policy", return_value=blocked_policy),
        ):
            result = initiate_refund(ctx, "ORD-H-001", 2000)
        assert result["success"] is False
        assert "not allowed" in result["error"].lower()


# ── 7. Write tool check registry ────────────────────────────────────

class TestWriteToolRegistry:
    def test_every_write_tool_has_action_check_or_policy_rerun(self):
        """Every tool in WRITE_TOOLS must pass through _hard_constraint_check/_needs_approval
        in react_agent AND/OR have _check_return_policy in its body."""
        import inspect
        from app.graph.agent_prod.tools import order_tools

        for tool_name in WRITE_TOOLS:
            fn = getattr(order_tools, tool_name, None)
            assert fn is not None, f"Write tool '{tool_name}' not found in order_tools"
            source = inspect.getsource(fn)
            has_policy_check = "_check_return_policy" in source
            has_for_update = "for_update=True" in source
            has_duplicate_check = "already exists" in source
            has_status_check = "cancellable" in source or "cannot be cancelled" in source
            assert has_for_update, f"Write tool '{tool_name}' missing FOR UPDATE lock"
            assert has_policy_check or has_duplicate_check or has_status_check, (
                f"Write tool '{tool_name}' has no policy re-check, duplicate check, or status check"
            )


# ── 8. Factory guard ────────────────────────────────────────────────

class TestFactoryGuard:
    def test_prod_env_mock_mode_raises(self):
        """The guard raises RuntimeError when app_env=prod but agent_mode!=prod."""
        from app.config import Setting, _check_factory_guard

        bad = Setting.model_construct(app_env="prod", agent_mode="mock")
        with pytest.raises(RuntimeError, match="APP_ENV=prod requires AGENT_MODE=prod"):
            _check_factory_guard(bad)

    def test_prod_env_prod_mode_ok(self):
        """No error when both are prod."""
        from app.config import Setting, _check_factory_guard

        ok = Setting.model_construct(app_env="prod", agent_mode="prod")
        _check_factory_guard(ok)
