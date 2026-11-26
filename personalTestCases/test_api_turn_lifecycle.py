"""API-level turn lifecycle and pending-HITL protection tests for /agent and /agent/resume.

Runs FastAPI's TestClient against the real compiled graph. Authentication is
replaced by overriding get_authenticated_customer, so get_reviewer's role check
still runs for real. Graph boundaries are mocked by the `env` fixture shared
with test_graph_integration.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.auth import TokenClaims, get_authenticated_customer
from app.graph.builder import build_graph
from app.graph.conversation_store import conversation_store
from app.graph.schemas import RouterOutput

from .conftest import (
    CUSTOMER_ALICE,
    CUSTOMER_BOB,
    FOLLOW_UP_RESPONSE,
    INITIAL_RESPONSE,
    REFUND_AMOUNT,
    TENANT_MEGAMART,
    TENANT_TINY_CANDLES,
)

ALICE = TokenClaims(tenant_id=TENANT_MEGAMART, customer_id=CUSTOMER_ALICE, email="alice@example.com")
BOB = TokenClaims(tenant_id=TENANT_MEGAMART, customer_id=CUSTOMER_BOB, email="bob@example.com")
REVIEWER = TokenClaims(
    tenant_id=TENANT_MEGAMART, customer_id="REVIEWER-001", email="reviewer@megamart.com", role="reviewer",
)
OTHER_TENANT_REVIEWER = TokenClaims(
    tenant_id=TENANT_TINY_CANDLES, customer_id="REVIEWER-TC", email="reviewer@tc.com", role="reviewer",
)

REFUND_MESSAGE = "I want a refund for ORD-MM-1001"
PENDING_REFUND_ARGS = {"order_id": "ORD-MM-1001", "amount": REFUND_AMOUNT, "refund_type": "original_method"}


@pytest.fixture()
def api(env, monkeypatch):
    graph = build_graph()
    monkeypatch.setattr(main, "agent_graph", graph)
    monkeypatch.setattr(main, "_THREAD_OWNERS", {})
    monkeypatch.setattr(main, "_THREAD_LOCKS", {})
    caller = {"claims": ALICE}
    main.app.dependency_overrides[get_authenticated_customer] = lambda: caller["claims"]

    def as_caller(claims: TokenClaims) -> None:
        caller["claims"] = claims

    def post_agent(message: str, thread_id: str | None = None, *, claims: TokenClaims = ALICE):
        as_caller(claims)
        body = {"message": message}
        if thread_id is not None:
            body["thread_id"] = thread_id
        return client.post("/agent", json=body)

    def resume(thread_id: str, decisions: list[dict], *, claims: TokenClaims = REVIEWER):
        as_caller(claims)
        return client.post("/agent/resume", json={"thread_id": thread_id, "decisions": decisions})

    def state(thread_id: str):
        return graph.get_state({"configurable": {"thread_id": thread_id}})

    def store_snapshot(thread_id: str, customer_id: str = CUSTOMER_ALICE) -> dict:
        return conversation_store.get_snapshot(TENANT_MEGAMART, customer_id, thread_id)

    def start_refund_escalation() -> str:
        env.router_llm.invoke.return_value = RouterOutput(
            intent="refund", entities={"order_id": "ORD-MM-1001"}, confidence=0.9,
        )
        r = post_agent(REFUND_MESSAGE)
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "interrupted"
        return r.json()["thread_id"]

    client = TestClient(main.app)
    yield SimpleNamespace(
        client=client, env=env, graph=graph, post_agent=post_agent, resume=resume,
        state=state, store_snapshot=store_snapshot, start_refund_escalation=start_refund_escalation,
    )
    main.app.dependency_overrides.pop(get_authenticated_customer, None)


def _pending_interrupts(snapshot) -> list:
    return [i for task in snapshot.tasks for i in task.interrupts]


# ===================================================================
# 1-4: per-turn reset, fan-out, persistent memory
# ===================================================================

class TestTurnReset:
    def test_consecutive_requests_do_not_accumulate_processed_results(self, api):
        first = api.post_agent("Track my order ORD-MM-1001")
        assert first.status_code == 200
        thread_id = first.json()["thread_id"]

        second = api.post_agent("Track my order ORD-MM-1002", thread_id)
        assert second.status_code == 200
        assert second.json()["thread_id"] == thread_id

        values = api.state(thread_id).values
        assert len(values["processed_results"]) == 1
        assert values["processed_results"][0]["entities"]["order_id"] == "ORD-MM-1002"
        assert [r["entities"]["order_id"] for r in values["intent_results"]] == ["ORD-MM-1002"]

    def test_parallel_send_branches_still_aggregate_within_a_turn(self, api):
        first = api.post_agent("Track my order ORD-MM-1001")
        thread_id = first.json()["thread_id"]

        r = api.post_agent("Track my order ORD-MM-1001, ORD-MM-1002", thread_id)
        assert r.status_code == 200

        values = api.state(thread_id).values
        assert len(values["processed_results"]) == 2
        assert {p["entities"]["order_id"] for p in values["processed_results"]} == {"ORD-MM-1001", "ORD-MM-1002"}
        assert len(values["intent_results"]) == 2

    def test_transient_fields_do_not_leak_after_a_hitl_turn(self, api):
        thread_id = api.start_refund_escalation()
        assert api.resume(thread_id, [{"index": 0, "action": "approve"}]).status_code == 200
        before = api.state(thread_id).values
        assert before["response_phase"] == "follow_up"
        assert before["has_tool_calls"] is True
        assert before["router_output"]["intent"] == "refund"

        r = api.post_agent("Track my order ORD-MM-1002", thread_id)
        assert r.status_code == 200
        body = r.json()
        assert body["router_output"]["intent"] == "order_status"
        assert body["policy_decision"]["reason_code"] == "ORDER_STATUS_ALLOWED"
        assert body["response"] == INITIAL_RESPONSE
        assert "status" not in body and "escalations" not in body

        values = api.state(thread_id).values
        assert values["response_phase"] == "complete"
        assert values["has_tool_calls"] is False
        assert values["pre_check_bail"] is False
        assert len(values["processed_results"]) == 1
        assert [r["intent"] for r in values["intent_results"]] == ["order_status"]
        assert not any(r.get("_newly_resolved") or r.get("pending_tool_call") for r in values["intent_results"])

    def test_conversation_memory_survives_new_turns(self, api):
        first = api.post_agent("Track my order ORD-MM-1001")
        thread_id = first.json()["thread_id"]
        known_orders = [{"order_id": "ORD-MM-1001", "product": "Wireless Headphones", "status": "delivered", "amount": 4999}]
        conversation_store.update_orders_context(TENANT_MEGAMART, CUSTOMER_ALICE, thread_id, known_orders)
        conversation_store.update_customer_context(TENANT_MEGAMART, CUSTOMER_ALICE, thread_id, {"name": "Alice"})

        api.post_agent("Track my order ORD-MM-1002", thread_id)

        values = api.state(thread_id).values
        assert values["conversation_history"] == [
            {"role": "human", "content": "Track my order ORD-MM-1001"},
            {"role": "assistant", "content": INITIAL_RESPONSE},
        ]
        assert values["orders_context"] == known_orders
        assert values["customer_context"] == {"name": "Alice"}
        assert [s["order_id"] for s in values["turn_summaries"]] == ["ORD-MM-1001"]
        assert values["conversation_id"] == thread_id
        assert [m["content"] for m in api.store_snapshot(thread_id)["messages"]] == [
            "Track my order ORD-MM-1001", INITIAL_RESPONSE,
            "Track my order ORD-MM-1002", INITIAL_RESPONSE,
        ]


# ===================================================================
# 5-6: 409 on a paused thread, with no side effects
# ===================================================================

class TestPausedThreadRejection:
    def test_new_message_on_paused_thread_returns_409(self, api):
        thread_id = api.start_refund_escalation()

        r = api.post_agent("Actually, where is ORD-MM-1002?", thread_id)

        assert r.status_code == 409
        assert r.json() == {"detail": {
            "code": "THREAD_AWAITING_APPROVAL",
            "message": "This conversation has a pending human approval. Please wait for it to be resolved.",
        }}

    def test_rejected_request_changes_neither_memory_nor_checkpoint(self, api):
        thread_id = api.start_refund_escalation()
        snap_before = api.state(thread_id)
        store_before = api.store_snapshot(thread_id)
        router_calls_before = api.env.router_llm.invoke.call_count

        r = api.post_agent("Cancel everything", thread_id)
        assert r.status_code == 409

        snap_after = api.state(thread_id)
        assert snap_after.config["configurable"]["checkpoint_id"] == snap_before.config["configurable"]["checkpoint_id"]
        assert snap_after.values == snap_before.values
        assert snap_after.next == ("escalation_handler",)
        assert [i.value for i in _pending_interrupts(snap_after)] == [i.value for i in _pending_interrupts(snap_before)]
        assert api.store_snapshot(thread_id) == store_before
        assert api.env.router_llm.invoke.call_count == router_calls_before
        api.env.wx_write.assert_not_called()
        api.env.esc_write.assert_not_called()

    def test_pending_state_detected_from_checkpoint_not_stale_fields(self, api):
        """A completed thread whose stale fields look 'partial' is not treated as paused."""
        first = api.post_agent("Track my order ORD-MM-1001")
        thread_id = first.json()["thread_id"]
        api.graph.update_state(
            {"configurable": {"thread_id": thread_id}},
            {"response_phase": "partial", "intent_results": [{"intent": "refund", "status": "awaiting_human"}]},
            as_node="follow_up_response",
        )
        assert api.state(thread_id).next == ()

        r = api.post_agent("Track my order ORD-MM-1002", thread_id)
        assert r.status_code == 200
        assert api.state(thread_id).values["intent_results"][0]["entities"]["order_id"] == "ORD-MM-1002"


# ===================================================================
# 7-8: resume continues the original workflow without a reset
# ===================================================================

class TestResume:
    def test_resume_completes_the_original_interrupted_workflow(self, api):
        thread_id = api.start_refund_escalation()

        r = api.resume(thread_id, [{"index": 0, "action": "approve"}])

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["response"] == FOLLOW_UP_RESPONSE
        assert body["intent_results"][0]["status"] == "fulfilled"
        assert body["intent_results"][0]["intent"] == "refund"
        api.env.esc_write.assert_called_once_with(
            "initiate_refund", TENANT_MEGAMART, CUSTOMER_ALICE, **PENDING_REFUND_ARGS,
        )
        api.env.wx_write.assert_not_called()
        assert api.state(thread_id).next == ()

    def test_resume_does_not_run_the_new_turn_reset(self, api):
        thread_id = api.start_refund_escalation()
        processed_before = api.state(thread_id).values["processed_results"]
        assert len(processed_before) == 1

        with patch.object(
            conversation_store, "add_human_message", wraps=conversation_store.add_human_message,
        ) as spy:
            r = api.resume(thread_id, [{"index": 0, "action": "approve"}])
        assert r.status_code == 200

        spy.assert_not_called()  # before_router_node did not run
        values = api.state(thread_id).values
        assert values["processed_results"] == processed_before
        assert values["user_message"] == REFUND_MESSAGE
        assert values["router_output"]["intent"] == "refund"
        assert [m["content"] for m in api.store_snapshot(thread_id)["messages"]] == [
            REFUND_MESSAGE, INITIAL_RESPONSE, FOLLOW_UP_RESPONSE,
        ]

    def test_resume_on_idle_thread_returns_409_without_invoking_graph(self, api):
        first = api.post_agent("Track my order ORD-MM-1001")
        thread_id = first.json()["thread_id"]
        values_before = api.state(thread_id).values

        with patch.object(api.graph, "invoke", wraps=api.graph.invoke) as invoke_spy:
            r = api.resume(thread_id, [{"index": 0, "action": "approve"}])

        assert r.status_code == 409
        assert r.json() == {"detail": main.THREAD_NOT_AWAITING_APPROVAL}
        invoke_spy.assert_not_called()
        assert api.state(thread_id).values == values_before
        api.env.esc_read.assert_not_called()
        api.env.esc_write.assert_not_called()

    def test_resume_unknown_thread_is_rejected(self, api):
        r = api.resume("no-such-thread", [{"index": 0, "action": "approve"}])
        assert r.status_code == 403
        assert r.json() == {"detail": main.THREAD_NOT_ACCESSIBLE}


# ===================================================================
# 9-11: after resolution the thread accepts new turns; nothing re-pends
# ===================================================================

class TestAfterResolution:
    @pytest.mark.parametrize("decision", [
        {"index": 0, "action": "approve"},
        {"index": 0, "action": "deny", "reason": "Not warranted."},
    ], ids=["approved", "denied"])
    def test_new_request_accepted_after_resolution(self, api, decision):
        thread_id = api.start_refund_escalation()
        assert api.resume(thread_id, [decision]).status_code == 200

        r = api.post_agent("Track my order ORD-MM-1002", thread_id)

        assert r.status_code == 200
        assert r.json()["response"] == INITIAL_RESPONSE
        assert api.state(thread_id).values["intent_results"][0]["entities"]["order_id"] == "ORD-MM-1002"

    def test_resolved_approval_never_reappears_as_pending(self, api):
        thread_id = api.start_refund_escalation()
        assert api.resume(thread_id, [{"index": 0, "action": "approve"}]).status_code == 200

        for message in ("Track my order ORD-MM-1002", "Track my order ORD-MM-1001"):
            r = api.post_agent(message, thread_id)
            assert r.status_code == 200
            assert "escalations" not in r.json()
            snapshot = api.state(thread_id)
            assert _pending_interrupts(snapshot) == []
            assert snapshot.next == ()
            assert not any(ir.get("status") == "awaiting_human" for ir in snapshot.values["intent_results"])

        assert api.env.esc_write.call_count == 1  # the refund ran exactly once

        # Replaying the old approval on the now-idle thread is rejected and must not refund again.
        assert api.resume(thread_id, [{"index": 0, "action": "approve"}]).status_code == 409
        assert api.env.esc_write.call_count == 1


# ===================================================================
# 12: unauthorized callers
# ===================================================================

class TestUnauthorizedAccess:
    def test_other_customer_gets_403_not_409_on_paused_thread(self, api):
        """403 comes before the pending check, so the pause state is not revealed."""
        thread_id = api.start_refund_escalation()
        checkpoint_before = api.state(thread_id).config["configurable"]["checkpoint_id"]

        r = api.post_agent("Hello", thread_id, claims=BOB)

        assert r.status_code == 403
        assert api.state(thread_id).config["configurable"]["checkpoint_id"] == checkpoint_before
        assert api.store_snapshot(thread_id, CUSTOMER_BOB)["messages"] == []

    @pytest.mark.parametrize("caller", [ALICE, BOB, OTHER_TENANT_REVIEWER], ids=["owner-customer", "other-customer", "other-tenant-reviewer"])
    def test_non_reviewers_and_other_tenants_cannot_resume(self, api, caller):
        thread_id = api.start_refund_escalation()

        r = api.resume(thread_id, [{"index": 0, "action": "approve"}], claims=caller)

        assert r.status_code == 403
        api.env.esc_write.assert_not_called()
        snapshot = api.state(thread_id)
        assert snapshot.next == ("escalation_handler",)
        assert len(_pending_interrupts(snapshot)) == 1
