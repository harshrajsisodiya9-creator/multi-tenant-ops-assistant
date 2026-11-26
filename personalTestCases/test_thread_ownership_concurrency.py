"""Thread ownership, per-thread locking and resume-safety regression tests.

Ownership tests go through HTTP (TestClient). Concurrency tests call the endpoint
functions directly from Python threads, which is how FastAPI runs these sync
endpoints (in a thread pool), and lets each thread pass its own claims.

No sleeps: a request is held inside the graph with an Event-gated fake LLM, and
an instrumented lock signals the moment a second caller finds it already held.
`join(timeout=...)` / `wait(timeout=...)` are safety caps only.
"""
from __future__ import annotations

import re
import threading
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import app.main as main
from app.auth import TokenClaims
from app.graph.conversation_store import conversation_store
from app.graph.schemas import RouterOutput

from .conftest import (
    CUSTOMER_ALICE,
    CUSTOMER_BOB,
    FOLLOW_UP_RESPONSE,
    INITIAL_RESPONSE,
    TENANT_MEGAMART,
    TENANT_TINY_CANDLES,
    _fake_response_llm,
)
from .test_api_turn_lifecycle import (  # noqa: F401  (api is a pytest fixture)
    ALICE,
    BOB,
    PENDING_REFUND_ARGS,
    REFUND_MESSAGE,
    REVIEWER,
    api,
)

WAIT = 10  # safety cap in seconds; never the thing a test relies on
ALICE_OTHER_TENANT = TokenClaims(tenant_id=TENANT_TINY_CANDLES, customer_id=CUSTOMER_ALICE, email="alice@tc.com")


# ===================================================================
# Helpers
# ===================================================================

class ObservedLock:
    """threading.Lock that sets `contended` when an acquirer finds it already held."""

    def __init__(self):
        self._lock = threading.Lock()
        self.contended = threading.Event()

    def acquire(self, blocking=True, timeout=-1):
        if self._lock.acquire(blocking=False):
            return True
        self.contended.set()
        return self._lock.acquire(blocking, timeout)

    def release(self):
        self._lock.release()

    def locked(self):
        return self._lock.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


class Gate:
    """Holds the first response-LLM call matching `predicate` until released."""

    def __init__(self, predicate):
        self.predicate = predicate
        self.entered = threading.Event()
        self.release = threading.Event()


class Worker(threading.Thread):
    def __init__(self, fn, *args, **kwargs):
        super().__init__(daemon=True)
        self._call = (fn, args, kwargs)
        self.result = None
        self.error = None

    def run(self):
        fn, args, kwargs = self._call
        try:
            self.result = fn(*args, **kwargs)
        except BaseException as exc:  # surfaced by the test, never swallowed
            self.error = exc


def call_agent(message: str, thread_id: str | None = None, claims: TokenClaims = ALICE):
    try:
        return 200, main.agent_endpoint(main.AgentRequest(message=message, thread_id=thread_id), claims=claims)
    except HTTPException as exc:
        return exc.status_code, {"detail": exc.detail}


def call_resume(thread_id: str, decisions: list[dict], claims: TokenClaims = REVIEWER):
    request = main.ResumeRequest(thread_id=thread_id, decisions=[main.HumanDecision(**d) for d in decisions])
    try:
        return 200, main.agent_resume_endpoint(request, claims=claims)
    except HTTPException as exc:
        return exc.status_code, {"detail": exc.detail}


def _user_content(messages) -> str:
    return messages[1]["content"]


def _is_follow_up(messages) -> bool:
    return "FOLLOW-UP" in messages[0]["content"]


@pytest.fixture()
def conc(api, monkeypatch):
    """Fresh lock registry with observable locks, an Event-gated response LLM and an
    invoke log recording when each graph invocation starts and ends."""
    monkeypatch.setattr(main, "_THREAD_LOCKS", {})
    monkeypatch.setattr(main, "_new_thread_lock", ObservedLock)

    gates: list[Gate] = []

    def gated_response(messages):
        for gate in gates:
            if not gate.release.is_set() and gate.predicate(messages):
                gate.entered.set()
                assert gate.release.wait(WAIT), "gate never released"
        return _fake_response_llm(messages)

    api.env.response_llm.invoke.side_effect = gated_response

    log: list[tuple[str, str]] = []
    log_guard = threading.Lock()
    real_invoke = api.graph.invoke

    def logged_invoke(graph_input, config=None, **kwargs):
        label = graph_input["user_message"] if isinstance(graph_input, dict) else "resume"
        with log_guard:
            log.append(("start", label))
        try:
            return real_invoke(graph_input, config=config, **kwargs)
        finally:
            with log_guard:
                log.append(("end", label))

    def gate(predicate) -> Gate:
        g = Gate(predicate)
        gates.append(g)
        return g

    def lock_of(thread_id: str) -> ObservedLock:
        return main._THREAD_LOCKS[thread_id].lock

    with patch.object(api.graph, "invoke", side_effect=logged_invoke):
        yield api, gate, log, lock_of
    for g in gates:  # never leave a worker blocked if a test fails early
        g.release.set()


def _new_thread(message: str = "Track my order ORD-MM-1001") -> str:
    status, body = call_agent(message)
    assert status == 200, body
    return body["thread_id"]


def _start_refund_escalation(api) -> str:
    api.env.router_llm.invoke.return_value = RouterOutput(
        intent="refund", entities={"order_id": "ORD-MM-1001"}, confidence=0.9,
    )
    status, body = call_agent(REFUND_MESSAGE)
    assert status == 200 and body["status"] == "interrupted", body
    return body["thread_id"]


def _finish(*workers: Worker) -> None:
    for w in workers:
        w.join(WAIT)
        assert not w.is_alive(), "worker did not finish"
        if w.error:
            raise w.error


# ===================================================================
# Ownership
# ===================================================================

class TestThreadOwnership:
    def test_unknown_client_thread_id_is_rejected_not_claimed(self, api):
        with patch.object(api.graph, "invoke", wraps=api.graph.invoke) as invoke_spy:
            first = api.post_agent("Hello", "client-chosen-id")
            retry = api.post_agent("Hello", "client-chosen-id")

        for r in (first, retry):
            assert r.status_code == 403
            assert r.json() == {"detail": main.THREAD_NOT_ACCESSIBLE}
        invoke_spy.assert_not_called()
        assert "client-chosen-id" not in main._THREAD_OWNERS
        assert api.state("client-chosen-id").values == {}
        assert conversation_store.get_snapshot(TENANT_MEGAMART, CUSTOMER_ALICE, "client-chosen-id")["messages"] == []

    def test_new_conversation_gets_server_generated_registered_id(self, api):
        a = api.post_agent("Track my order ORD-MM-1001")
        b = api.post_agent("Track my order ORD-MM-1002")

        ids = [a.json()["thread_id"], b.json()["thread_id"]]
        assert all(re.fullmatch(r"[0-9a-f]{32}", tid) for tid in ids)
        assert ids[0] != ids[1]
        for tid in ids:
            assert main._THREAD_OWNERS[tid] == (TENANT_MEGAMART, CUSTOMER_ALICE)

    def test_owner_can_reuse_thread(self, api):
        thread_id = api.post_agent("Track my order ORD-MM-1001").json()["thread_id"]

        r = api.post_agent("Track my order ORD-MM-1002", thread_id)

        assert r.status_code == 200
        assert r.json()["thread_id"] == thread_id
        assert api.state(thread_id).values["user_message"] == "Track my order ORD-MM-1002"

    @pytest.mark.parametrize("intruder", [BOB, ALICE_OTHER_TENANT], ids=["other-customer", "other-tenant-same-customer-id"])
    def test_non_owner_rejected_exactly_like_unknown_thread(self, api, intruder):
        thread_id = api.post_agent("Track my order ORD-MM-1001").json()["thread_id"]
        checkpoint_before = api.state(thread_id).config["configurable"]["checkpoint_id"]
        unknown = api.post_agent("Hello", "never-issued", claims=intruder)

        r = api.post_agent("Hello", thread_id, claims=intruder)

        assert r.status_code == unknown.status_code == 403
        assert r.json() == unknown.json()  # existence of the thread is not revealed
        assert api.state(thread_id).config["configurable"]["checkpoint_id"] == checkpoint_before
        assert main._THREAD_OWNERS[thread_id] == (TENANT_MEGAMART, CUSTOMER_ALICE)

    def test_concurrent_registration_assigns_exactly_one_owner(self, api):
        n = 16
        barrier = threading.Barrier(n)
        results: dict[int, bool] = {}

        def register(i):
            barrier.wait(WAIT)
            results[i] = main._register_thread("contested", (TENANT_MEGAMART, f"CUST-{i}"))

        workers = [Worker(register, i) for i in range(n)]
        for w in workers:
            w.start()
        _finish(*workers)

        winners = [i for i, won in results.items() if won]
        assert len(winners) == 1
        assert main._THREAD_OWNERS["contested"] == (TENANT_MEGAMART, f"CUST-{winners[0]}")

    def test_registering_an_existing_thread_never_overwrites_owner(self, api):
        assert main._register_thread("t-1", (TENANT_MEGAMART, CUSTOMER_ALICE)) is True
        assert main._register_thread("t-1", (TENANT_MEGAMART, CUSTOMER_BOB)) is False
        assert main._THREAD_OWNERS["t-1"] == (TENANT_MEGAMART, CUSTOMER_ALICE)


# ===================================================================
# Concurrency
# ===================================================================

class TestPerThreadLocking:
    def test_two_agent_requests_on_same_thread_are_serialized(self, conc):
        api, gate, log, lock_of = conc
        thread_id = _new_thread()
        log.clear()
        first_msg, second_msg = "Track my order ORD-MM-1001 again", "Track my order ORD-MM-1002"
        g = gate(lambda m: first_msg in _user_content(m))

        a = Worker(call_agent, first_msg, thread_id)
        a.start()
        assert g.entered.wait(WAIT)
        lock = lock_of(thread_id)
        assert lock.locked()

        b = Worker(call_agent, second_msg, thread_id)
        b.start()
        assert lock.contended.wait(WAIT)  # b is blocked on the same lock a holds
        assert ("start", second_msg) not in log

        g.release.set()
        _finish(a, b)
        assert a.result[0] == 200 and b.result[0] == 200
        assert log == [("start", first_msg), ("end", first_msg), ("start", second_msg), ("end", second_msg)]
        assert main._THREAD_LOCKS == {}

    def test_requests_on_different_threads_run_independently(self, conc):
        api, gate, log, lock_of = conc
        thread_x, thread_y = _new_thread(), _new_thread("Track my order ORD-MM-1002")
        held_msg = "Track my order ORD-MM-1001 again"
        g = gate(lambda m: held_msg in _user_content(m))

        a = Worker(call_agent, held_msg, thread_x)
        a.start()
        assert g.entered.wait(WAIT)

        b = Worker(call_agent, "Track my order ORD-MM-1002", thread_y)
        b.start()
        b.join(WAIT)
        assert not b.is_alive(), "thread Y was blocked by thread X's request"
        assert b.result[0] == 200
        assert a.is_alive()  # X's request is still inside the graph
        assert not lock_of(thread_x).contended.is_set()

        g.release.set()
        _finish(a)
        assert a.result[0] == 200

    def test_second_request_gets_409_when_first_creates_interrupt(self, conc):
        api, gate, log, lock_of = conc
        thread_id = _new_thread()
        api.env.router_llm.invoke.return_value = RouterOutput(
            intent="refund", entities={"order_id": "ORD-MM-1001"}, confidence=0.9,
        )
        g = gate(lambda m: REFUND_MESSAGE in _user_content(m) and not _is_follow_up(m))

        a = Worker(call_agent, REFUND_MESSAGE, thread_id)
        a.start()
        assert g.entered.wait(WAIT)  # inside response_agent, before the interrupt exists
        assert api.state(thread_id).next != ("escalation_handler",)

        b = Worker(call_agent, "Track my order ORD-MM-1002", thread_id)
        b.start()
        assert lock_of(thread_id).contended.wait(WAIT)

        g.release.set()
        _finish(a, b)
        assert a.result[0] == 200 and a.result[1]["status"] == "interrupted"
        assert b.result == (409, {"detail": main.THREAD_AWAITING_APPROVAL})

        snapshot = api.state(thread_id)
        assert snapshot.next == ("escalation_handler",)
        assert [i.value[0]["pending_tool_call"]["args"] for t in snapshot.tasks for i in t.interrupts] == [PENDING_REFUND_ARGS]
        assert ("start", "Track my order ORD-MM-1002") not in log
        assert "Track my order ORD-MM-1002" not in [
            m["content"] for m in conversation_store.get_snapshot(TENANT_MEGAMART, CUSTOMER_ALICE, thread_id)["messages"]
        ]
        api.env.esc_write.assert_not_called()

    def test_agent_waits_for_in_progress_resume(self, conc):
        api, gate, log, lock_of = conc
        thread_id = _start_refund_escalation(api)
        log.clear()
        g = gate(_is_follow_up)

        a = Worker(call_resume, thread_id, [{"index": 0, "action": "approve"}])
        a.start()
        assert g.entered.wait(WAIT)  # refund executed; follow-up not yet composed
        assert api.state(thread_id).next  # resume still in progress

        b = Worker(call_agent, "Track my order ORD-MM-1002", thread_id)
        b.start()
        assert lock_of(thread_id).contended.wait(WAIT)

        g.release.set()
        _finish(a, b)
        assert a.result[0] == 200 and a.result[1]["response"] == FOLLOW_UP_RESPONSE
        assert b.result[0] == 200 and b.result[1]["response"] == INITIAL_RESPONSE
        assert log == [
            ("start", "resume"), ("end", "resume"),
            ("start", "Track my order ORD-MM-1002"), ("end", "Track my order ORD-MM-1002"),
        ]
        assert api.env.esc_write.call_count == 1

    def test_resume_waits_for_agent_turn_that_creates_the_interrupt(self, conc):
        api, gate, log, lock_of = conc
        thread_id = _new_thread()
        api.env.router_llm.invoke.return_value = RouterOutput(
            intent="refund", entities={"order_id": "ORD-MM-1001"}, confidence=0.9,
        )
        g = gate(lambda m: REFUND_MESSAGE in _user_content(m) and not _is_follow_up(m))

        a = Worker(call_agent, REFUND_MESSAGE, thread_id)
        a.start()
        assert g.entered.wait(WAIT)

        b = Worker(call_resume, thread_id, [{"index": 0, "action": "approve"}])
        b.start()
        assert lock_of(thread_id).contended.wait(WAIT)

        g.release.set()
        _finish(a, b)
        assert a.result[1]["status"] == "interrupted"
        assert b.result[0] == 200  # saw the interrupt created by the finished turn
        assert b.result[1]["intent_results"][0]["status"] == "fulfilled"
        api.env.esc_write.assert_called_once_with(
            "initiate_refund", TENANT_MEGAMART, CUSTOMER_ALICE, **PENDING_REFUND_ARGS,
        )


class TestLockRegistryCleanup:
    def test_lock_entry_survives_handoff_and_is_removed_after_last_user(self, monkeypatch):
        monkeypatch.setattr(main, "_THREAD_LOCKS", {})
        created: list[ObservedLock] = []

        def factory():
            created.append(ObservedLock())
            return created[-1]

        monkeypatch.setattr(main, "_new_thread_lock", factory)
        a_in, a_go, b_in, b_go, c_in = (threading.Event() for _ in range(5))

        def hold(entered, go=None):
            with main._thread_lock("t"):
                entered.set()
                if go is not None:
                    assert go.wait(WAIT)

        a = Worker(hold, a_in, a_go)
        a.start()
        assert a_in.wait(WAIT)
        entry = main._THREAD_LOCKS["t"]

        b = Worker(hold, b_in, b_go)
        b.start()
        assert entry.lock.contended.wait(WAIT)
        assert entry.users == 2

        a_go.set()
        assert b_in.wait(WAIT)
        a.join(WAIT)
        # a's cleanup ran while b held the lock: the entry must not have been dropped
        assert main._THREAD_LOCKS["t"] is entry
        assert entry.users == 1

        entry.lock.contended.clear()
        c = Worker(hold, c_in)
        c.start()
        assert entry.lock.contended.wait(WAIT)  # c waits on the same lock, not a fresh one
        assert not c_in.is_set()

        b_go.set()
        _finish(a, b, c)
        assert len(created) == 1
        assert main._THREAD_LOCKS == {}

    def test_many_threads_on_few_ids_keep_mutual_exclusion_and_leave_no_entries(self, monkeypatch):
        monkeypatch.setattr(main, "_THREAD_LOCKS", {})
        ids = ["t-a", "t-b"]
        inside = {tid: 0 for tid in ids}
        max_inside = {tid: 0 for tid in ids}
        guard = threading.Lock()
        n_workers, rounds = 8, 50
        barrier = threading.Barrier(n_workers)

        def work(i):
            barrier.wait(WAIT)
            for r in range(rounds):
                tid = ids[(i + r) % len(ids)]
                with main._thread_lock(tid):
                    with guard:
                        inside[tid] += 1
                        max_inside[tid] = max(max_inside[tid], inside[tid])
                    with guard:
                        inside[tid] -= 1

        workers = [Worker(work, i) for i in range(n_workers)]
        for w in workers:
            w.start()
        _finish(*workers)

        assert max_inside == {"t-a": 1, "t-b": 1}
        assert main._THREAD_LOCKS == {}


# ===================================================================
# Resume safety
# ===================================================================

class TestResumeSafety:
    def test_valid_pending_interrupt_resumes(self, api):
        thread_id = _start_refund_escalation(api)

        status, body = call_resume(thread_id, [{"index": 0, "action": "approve"}])

        assert status == 200
        assert body["response"] == FOLLOW_UP_RESPONSE
        assert body["intent_results"][0]["status"] == "fulfilled"
        api.env.esc_write.assert_called_once_with(
            "initiate_refund", TENANT_MEGAMART, CUSTOMER_ALICE, **PENDING_REFUND_ARGS,
        )

    @pytest.mark.parametrize("decision", [
        {"index": 0, "action": "approve"},
        {"index": 0, "action": "deny", "reason": "No."},
    ], ids=["after-approval", "after-denial"])
    def test_completed_hitl_thread_resume_returns_409(self, api, decision):
        thread_id = _start_refund_escalation(api)
        assert call_resume(thread_id, [decision])[0] == 200
        writes_after_first = api.env.esc_write.call_count

        with patch.object(api.graph, "invoke", wraps=api.graph.invoke) as invoke_spy:
            status, body = call_resume(thread_id, [{"index": 0, "action": "approve"}])

        assert (status, body) == (409, {"detail": main.THREAD_NOT_AWAITING_APPROVAL})
        assert "intent_results" not in body
        invoke_spy.assert_not_called()
        assert api.env.esc_write.call_count == writes_after_first

    def test_unknown_thread_cannot_be_resumed(self, api):
        with patch.object(api.graph, "invoke", wraps=api.graph.invoke) as invoke_spy:
            status, body = call_resume("never-issued", [{"index": 0, "action": "approve"}])

        assert (status, body) == (403, {"detail": main.THREAD_NOT_ACCESSIBLE})
        invoke_spy.assert_not_called()

    def test_other_tenant_reviewer_gets_same_response_as_unknown_thread(self, api):
        thread_id = _start_refund_escalation(api)
        other_reviewer = TokenClaims(
            tenant_id=TENANT_TINY_CANDLES, customer_id="REVIEWER-TC", email="r@tc.com", role="reviewer",
        )

        with patch.object(api.graph, "invoke", wraps=api.graph.invoke) as invoke_spy:
            foreign = call_resume(thread_id, [{"index": 0, "action": "approve"}], claims=other_reviewer)
            unknown = call_resume("never-issued", [{"index": 0, "action": "approve"}], claims=other_reviewer)

        assert foreign == unknown == (403, {"detail": main.THREAD_NOT_ACCESSIBLE})
        invoke_spy.assert_not_called()
        api.env.esc_write.assert_not_called()
        assert api.state(thread_id).next == ("escalation_handler",)

    def test_customer_cannot_resume_even_own_thread_over_http(self, api):
        thread_id = _start_refund_escalation(api)

        for caller in (ALICE, BOB):
            r = api.resume(thread_id, [{"index": 0, "action": "approve"}], claims=caller)
            assert r.status_code == 403
            assert "intent_results" not in r.json()

        api.env.esc_write.assert_not_called()
        assert api.state(thread_id).next == ("escalation_handler",)

    def test_interrupt_outside_approval_node_is_not_resumable(self, api, monkeypatch):
        """Only a pause at the escalation/approval node counts as awaiting approval."""
        thread_id = _new_thread()
        monkeypatch.setattr(main, "_pending_interrupt_nodes", lambda config: {"some_other_node"})

        with patch.object(api.graph, "invoke", wraps=api.graph.invoke) as invoke_spy:
            status, body = call_resume(thread_id, [{"index": 0, "action": "approve"}])

        assert (status, body) == (409, {"detail": main.THREAD_NOT_AWAITING_APPROVAL})
        invoke_spy.assert_not_called()
