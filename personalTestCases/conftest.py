"""Shared fixtures for personalTestCases.

Provides deterministic state builders, mock helpers, and the auto-patching
of AGENT_MODE that the existing test suite uses.
"""
from __future__ import annotations

import os
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.graph.conversation_store import conversation_store
from app.graph.schemas import IntentResult, PolicyDecision, RouterOutput


# ---------------------------------------------------------------------------
# Auto-patch agent mode to "mock" for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_agent_mode():
    with patch.dict(os.environ, {"AGENT_MODE": "mock"}):
        import app.config
        original = app.config.setting.agent_mode
        app.config.setting.agent_mode = "mock"
        yield
        app.config.setting.agent_mode = original


# ---------------------------------------------------------------------------
# Tenant / customer constants
# ---------------------------------------------------------------------------

TENANT_MEGAMART = "megamart"
TENANT_TINY_CANDLES = "tiny-candles"
TENANT_INACTIVE = "inactive-co"
CUSTOMER_ALICE = "CUST-MM-001"
CUSTOMER_BOB = "CUST-MM-002"
CUSTOMER_GRACE = "CUST-TC-001"


# ---------------------------------------------------------------------------
# State builder helpers
# ---------------------------------------------------------------------------

def make_base_state(
    *,
    tenant_id: str = TENANT_MEGAMART,
    customer_id: str = CUSTOMER_ALICE,
    user_message: str = "Hello",
    conversation_id: str = "conv-test-001",
    conversation_history: list[dict] | None = None,
    customer_context: dict | None = None,
    orders_context: list[dict] | None = None,
    turn_summaries: list[dict] | None = None,
) -> dict:
    return {
        "tenant_id": tenant_id,
        "customer_id": customer_id,
        "user_message": user_message,
        "conversation_id": conversation_id,
        "conversation_history": conversation_history or [],
        "customer_context": customer_context or {},
        "orders_context": orders_context or [],
        "turn_summaries": turn_summaries or [],
    }


def make_router_output(
    *,
    intent: str = "order_status",
    entities: dict | None = None,
    confidence: float = 0.95,
    escalation_signals: list[str] | None = None,
    sub_intents: list[dict] | None = None,
    conditional: dict | None = None,
) -> dict:
    ro = RouterOutput(
        intent=intent,
        entities=entities or {},
        confidence=confidence,
        escalation_signals=escalation_signals or [],
        sub_intents=sub_intents or [],
        conditional=conditional,
    )
    return ro.model_dump()


def make_intent_result(
    *,
    intent: str = "order_status",
    entities: dict | None = None,
    policy_action: str = "allow",
    reason_code: str = "ORDER_STATUS_ALLOWED",
    constraints: dict | None = None,
    domain_result: str = "",
    status: str = "pending",
    tool_calls: list[dict] | None = None,
    conditional: dict | None = None,
) -> dict:
    ir = IntentResult(
        intent=intent,
        entities=entities or {},
        policy_action=policy_action,
        reason_code=reason_code,
        constraints=constraints or {},
        domain_result=domain_result,
        status=status,
        tool_calls=tool_calls or [],
        conditional=conditional,
    )
    return ir.model_dump()


def make_policy_decision(
    *,
    action: str = "allow",
    reason_code: str = "ORDER_STATUS_ALLOWED",
    constraints: dict | None = None,
) -> dict:
    pd = PolicyDecision(
        action=action,
        reason_code=reason_code,
        constraints=constraints or {},
    )
    return pd.model_dump()


# ---------------------------------------------------------------------------
# Shared compiled-graph environment fixture
# ---------------------------------------------------------------------------

INITIAL_RESPONSE = "INITIAL RESPONSE"
FOLLOW_UP_RESPONSE = "FOLLOW-UP RESPONSE"
FAQ_ANSWER = "You can return items within 7 days of delivery."

GRAPH_TEST_ORDERS = {
    "ORD-MM-1001": {"order_id": "ORD-MM-1001", "product": "Wireless Headphones", "status": "delivered", "delivery_date": None},
    "ORD-MM-1002": {"order_id": "ORD-MM-1002", "product": "Winter Jacket", "status": "processing", "delivery_date": None},
}

REFUND_AMOUNT = 60_000  # above GLOBAL_REFUND_CEILING (50_000) -> requires human approval


def _fake_policy(ctx):
    intent = ctx.router_output.intent
    if intent == "order_status":
        return PolicyDecision(action="allow", reason_code="ORDER_STATUS_ALLOWED")
    if intent == "cancel_order":
        return PolicyDecision(action="allow", reason_code="CANCEL_ALLOWED")
    if intent == "refund":
        return PolicyDecision(
            action="allow", reason_code="REFUND_ALLOWED",
            constraints={"refund_type": "original_method"},
        )
    raise AssertionError(f"unexpected intent reached the policy gate: {intent}")


def _fake_read_tool(tool_name, tenant_id, customer_id, **kwargs):
    assert (tenant_id, customer_id) == (TENANT_MEGAMART, CUSTOMER_ALICE)
    order_id = kwargs["order_id"]
    if tool_name == "get_order_details":
        return {"order_id": order_id, "order_status": "delivered", "total_amount": REFUND_AMOUNT}
    if tool_name == "get_return_eligibility":
        return {"eligible": True, "amount": REFUND_AMOUNT, "refund_type": "original_method"}
    raise AssertionError(f"unexpected read tool: {tool_name}")


def _fake_response_llm(messages):
    content = FOLLOW_UP_RESPONSE if "FOLLOW-UP" in messages[0]["content"] else INITIAL_RESPONSE
    return MagicMock(content=content)


@pytest.fixture()
def env():
    """Patch every external boundary of the compiled graph."""
    conversation_store.reset()
    with ExitStack() as stack:
        def p(target, **kwargs):
            return stack.enter_context(patch(target, **kwargs))

        router_cls = p("app.graph.nodes.router.ChatGroq")
        response_cls = p("app.graph.nodes.response_agent.ChatGroq")
        response_cls.return_value.invoke.side_effect = _fake_response_llm

        ns = SimpleNamespace(
            router_llm=router_cls.return_value.with_structured_output.return_value,
            response_llm=response_cls.return_value,
            tenant_pre_check=p("app.graph.nodes.pre_check.get_tenant_info", return_value={"active": True}),
            tenant_fan_out=p("app.graph.nodes.policy_fan_out.get_tenant_info", return_value={"active": True}),
            gate_order=p("app.graph.nodes.policy_fan_out.get_order_scoped", side_effect=lambda t, c, oid: GRAPH_TEST_ORDERS.get(oid)),
            customer_orders=p("app.graph.nodes.policy_fan_out.get_customer_orders", return_value=list(GRAPH_TEST_ORDERS.values())),
            refund_count=p("app.graph.nodes.policy_fan_out.get_refund_count_30d", return_value=0),
            policy=p("app.graph.nodes.policy_fan_out.evaluate_policy", side_effect=_fake_policy),
            executor_order=p("app.graph.nodes.intent_executor.get_order_scoped", side_effect=lambda t, c, oid: GRAPH_TEST_ORDERS.get(oid)),
            faq=p("app.graph.nodes.intent_executor._execute_faq", return_value=FAQ_ANSWER),
            condition_read=p("app.graph.nodes.conditional_executor._invoke_read_tool"),
            wx_read=p("app.graph.nodes.write_executor._invoke_read_tool", side_effect=_fake_read_tool),
            wx_write=p("app.graph.nodes.write_executor._invoke_write_tool"),
            esc_read=p("app.graph.nodes.escalation_handler._invoke_read_tool", side_effect=_fake_read_tool),
            esc_write=p("app.graph.nodes.escalation_handler._invoke_write_tool"),
        )
        ns.esc_write.return_value = {"success": True, "message": f"Refund of {REFUND_AMOUNT} processed."}
        yield ns
    conversation_store.reset()
