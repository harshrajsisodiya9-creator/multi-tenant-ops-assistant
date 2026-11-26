"""Graph-level integration tests against the real compiled graph from build_graph().

Unlike the node-level tests, these run real routing (route_intents / Send fan-out),
the real reduce_processed_results reducer on processed_results, the real MemorySaver
checkpointer and the real interrupt() inside escalation_handler_node.

Only external boundaries are mocked: the Groq LLM clients, data_access lookups,
the policy gate decision, the RAG lookup and the read/write tool invokers.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from langgraph.types import Command

from app.graph.builder import build_graph
from app.graph.schemas import RouterOutput

from .conftest import (
    CUSTOMER_ALICE,
    FAQ_ANSWER,
    FOLLOW_UP_RESPONSE,
    INITIAL_RESPONSE,
    REFUND_AMOUNT,
    TENANT_MEGAMART,
)


def _config() -> dict:
    return {"configurable": {"thread_id": f"test-{uuid4().hex}"}}


def _turn(message: str, thread_config: dict) -> dict:
    return {
        "tenant_id": TENANT_MEGAMART,
        "customer_id": CUSTOMER_ALICE,
        "user_message": message,
        "conversation_id": thread_config["configurable"]["thread_id"],
    }


# ===================================================================
# Send() fan-out + operator.add reducer
# ===================================================================

class TestSendFanOutReducer:
    def test_two_different_executors_merge_via_reducer(self, env):
        """order_status -> quick_resolver and faq -> knowledge_lookup run as parallel
        Send() branches; both results are merged into intent_results."""
        message = "Where is ORD-MM-1001 and what is your return policy?"
        env.router_llm.invoke.return_value = RouterOutput(
            intent="order_status",
            entities={"order_id": "ORD-MM-1001"},
            confidence=0.93,
            sub_intents=[{"intent": "faq", "entities": {}, "confidence": 0.9}],
        )
        graph = build_graph()
        config = _config()

        final = graph.invoke(_turn(message, config), config=config)

        env.router_llm.invoke.assert_called_once()
        assert len(final["processed_results"]) == 2
        assert len(final["intent_results"]) == 2
        by_intent = {r["intent"]: r for r in final["intent_results"]}
        assert set(by_intent) == {"order_status", "faq"}

        assert by_intent["order_status"]["status"] == "fulfilled"
        assert by_intent["order_status"]["domain_result"] == (
            "Order ORD-MM-1001 (Wireless Headphones) status: delivered."
        )
        env.executor_order.assert_called_once_with(TENANT_MEGAMART, CUSTOMER_ALICE, "ORD-MM-1001")

        assert by_intent["faq"]["status"] == "fulfilled"
        assert by_intent["faq"]["domain_result"] == FAQ_ANSWER
        env.faq.assert_called_once_with(TENANT_MEGAMART, message)

        env.policy.assert_called_once()  # faq skips the gate; only order_status is evaluated
        assert final["has_tool_calls"] is False
        assert final["response_phase"] == "complete"
        assert final["response"] == INITIAL_RESPONSE
        user_content = env.response_llm.invoke.call_args[0][0][1]["content"]
        assert "Wireless Headphones" in user_content
        assert FAQ_ANSWER in user_content
        assert graph.get_state(config).next == ()

    def test_same_executor_two_branches_merge_via_reducer(self, env):
        """Two order IDs fan out to two parallel quick_resolver invocations; the
        reducer keeps both results instead of the last write winning."""
        message = "Track my order ORD-MM-1001, ORD-MM-1002"
        graph = build_graph()
        config = _config()

        final = graph.invoke(_turn(message, config), config=config)

        env.router_llm.invoke.assert_not_called()  # keyword fast path
        assert len(final["processed_results"]) == 2
        results = {r["entities"]["order_id"]: r for r in final["intent_results"]}
        assert set(results) == {"ORD-MM-1001", "ORD-MM-1002"}
        assert all(r["intent"] == "order_status" and r["status"] == "fulfilled" for r in results.values())
        assert "Wireless Headphones" in results["ORD-MM-1001"]["domain_result"]
        assert "Winter Jacket" in results["ORD-MM-1002"]["domain_result"]
        assert env.executor_order.call_count == 2
        assert final["response"] == INITIAL_RESPONSE

    def test_second_turn_on_same_thread_contains_only_its_own_results(self, env):
        """Regression: before_router clears processed_results, so turn 1 results do not
        leak into turn 2 on the same checkpointed thread_id (as /agent allows)."""
        graph = build_graph()
        config = _config()

        graph.invoke(_turn("Track my order ORD-MM-1001", config), config=config)
        second = graph.invoke(_turn("Track my order ORD-MM-1002", config), config=config)

        assert len(second["processed_results"]) == 1
        assert [r["entities"]["order_id"] for r in second["intent_results"]] == ["ORD-MM-1002"]
        user_content = env.response_llm.invoke.call_args[0][0][1]["content"]
        assert "ORD-MM-1002" in user_content
        assert "ORD-MM-1001" not in user_content.split("Intent results:")[1]

    def test_conversation_memory_survives_the_reset(self, env):
        """Execution state is cleared, but conversation history from turn 1 reaches turn 2."""
        graph = build_graph()
        config = _config()

        graph.invoke(_turn("Track my order ORD-MM-1001", config), config=config)
        second = graph.invoke(_turn("Track my order ORD-MM-1002", config), config=config)

        assert second["conversation_history"] == [
            {"role": "human", "content": "Track my order ORD-MM-1001"},
            {"role": "assistant", "content": INITIAL_RESPONSE},
        ]
        assert [s["order_id"] for s in second["turn_summaries"]] == ["ORD-MM-1001"]


class TestConditionalWriteRevalidation:
    def test_stale_condition_blocks_write_in_compiled_graph(self, env):
        """A condition that changes after the evidence read must block the write."""
        message = "If my order hasn't shipped, cancel ORD-MM-1001"
        env.router_llm.invoke.return_value = RouterOutput(
            intent="cancel_order",
            entities={"order_id": "ORD-MM-1001"},
            confidence=0.92,
            conditional={
                "request_type": "conditional_action",
                "condition": {
                    "field": "order_status",
                    "operator": "not_equals",
                    "value": "shipped",
                },
            },
        )

        statuses = iter(("pending", "shipped", "shipped"))

        def read_order(tool_name, tenant_id, customer_id, **kwargs):
            assert tool_name == "get_order_details"
            assert (tenant_id, customer_id) == (TENANT_MEGAMART, CUSTOMER_ALICE)
            return {"order_id": kwargs["order_id"], "order_status": next(statuses)}

        env.condition_read.side_effect = read_order
        env.wx_read.side_effect = read_order
        graph = build_graph()
        config = _config()

        final = graph.invoke(_turn(message, config), config=config)

        result = final["intent_results"][0]
        assert result["intent"] == "cancel_order"
        assert result["status"] == "blocked"
        assert "no longer met" in result["domain_result"].lower()
        env.condition_read.assert_called_once_with(
            "get_order_details", TENANT_MEGAMART, CUSTOMER_ALICE,
            order_id="ORD-MM-1001",
        )
        assert env.wx_read.call_count == 2  # workflow read and final revalidation
        env.wx_write.assert_not_called()


# ===================================================================
# HITL: real interrupt() + MemorySaver + Command(resume=...)
# ===================================================================

class TestHITLLifecycle:
    MESSAGE = "I want a refund for ORD-MM-1001"

    def _start_refund_escalation(self, env):
        """Run turn 1 until escalation_handler interrupts for approval of a large refund."""
        env.router_llm.invoke.return_value = RouterOutput(
            intent="refund", entities={"order_id": "ORD-MM-1001"}, confidence=0.9,
        )
        graph = build_graph()
        config = _config()
        graph.invoke(_turn(self.MESSAGE, config), config=config)
        return graph, config

    def test_graph_pauses_at_escalation_without_executing_write(self, env):
        graph, config = self._start_refund_escalation(env)

        snapshot = graph.get_state(config)
        assert snapshot.next == ("escalation_handler",)
        interrupts = [i for task in snapshot.tasks for i in task.interrupts]
        assert len(interrupts) == 1
        escalations = interrupts[0].value
        assert len(escalations) == 1
        assert escalations[0]["index"] == 0
        assert escalations[0]["intent"] == "refund"
        assert escalations[0]["reason_code"] == "GLOBAL_REFUND_CEILING"
        assert escalations[0]["pending_tool_call"] == {
            "tool": "initiate_refund",
            "args": {"order_id": "ORD-MM-1001", "amount": REFUND_AMOUNT, "refund_type": "original_method"},
            "reasons": ["GLOBAL_REFUND_CEILING"],
        }

        values = snapshot.values
        assert values["response_phase"] == "partial"
        assert values["response"] == INITIAL_RESPONSE
        assert values["intent_results"][0]["status"] == "awaiting_human"
        env.wx_write.assert_not_called()
        env.esc_write.assert_not_called()

    def test_resume_with_approval_executes_write_and_follows_up(self, env):
        graph, config = self._start_refund_escalation(env)

        final = graph.invoke(
            Command(resume={"decisions": [{"index": 0, "action": "approve"}]}),
            config=config,
        )

        env.wx_write.assert_not_called()
        env.esc_write.assert_called_once_with(
            "initiate_refund", TENANT_MEGAMART, CUSTOMER_ALICE,
            order_id="ORD-MM-1001", amount=REFUND_AMOUNT, refund_type="original_method",
        )
        assert [c[0][0] for c in env.esc_read.call_args_list] == [
            "get_order_details", "get_return_eligibility",
        ]

        ir = final["intent_results"][0]
        assert ir["status"] == "fulfilled"
        assert ir["_newly_resolved"] is True
        assert ir["domain_result"] == f"Refund of {REFUND_AMOUNT} processed."
        assert ir["tool_calls"][-1] == {
            "tool": "initiate_refund",
            "args": {"order_id": "ORD-MM-1001", "amount": REFUND_AMOUNT, "refund_type": "original_method"},
            "result": {"success": True, "message": f"Refund of {REFUND_AMOUNT} processed."},
        }
        # hitl_action_check ran on the executed write
        assert [f["flag"] for f in ir["audit_flags"]] == ["NEAR_CEILING"]

        assert final["response_phase"] == "follow_up"
        assert final["response"] == FOLLOW_UP_RESPONSE
        assert graph.get_state(config).next == ()

    def test_resume_with_denial_executes_nothing(self, env):
        graph, config = self._start_refund_escalation(env)

        final = graph.invoke(
            Command(resume={"decisions": [{"index": 0, "action": "deny", "reason": "Not warranted."}]}),
            config=config,
        )

        env.esc_read.assert_not_called()
        env.esc_write.assert_not_called()
        env.wx_write.assert_not_called()
        ir = final["intent_results"][0]
        assert ir["status"] == "blocked"
        assert ir["reason_code"] == "HUMAN_DENIED"
        assert ir["domain_result"] == "Not warranted."
        assert final["response"] == FOLLOW_UP_RESPONSE
        assert graph.get_state(config).next == ()

    def test_next_turn_after_approval_does_not_re_escalate(self, env):
        """Regression: the approved refund's stale awaiting_human entry must not
        re-trigger an interrupt (and a second refund) on the next turn."""
        graph, config = self._start_refund_escalation(env)
        graph.invoke(
            Command(resume={"decisions": [{"index": 0, "action": "approve"}]}),
            config=config,
        )
        assert env.esc_write.call_count == 1

        second = graph.invoke(_turn("Track my order ORD-MM-1002", config), config=config)

        assert graph.get_state(config).next == ()  # no new interrupt
        assert env.esc_write.call_count == 1  # refund not executed again
        assert len(second["processed_results"]) == 1
        assert [r["intent"] for r in second["intent_results"]] == ["order_status"]
        assert second["intent_results"][0]["status"] == "fulfilled"
        assert second["response_phase"] == "complete"
        assert second["response"] == INITIAL_RESPONSE

    @pytest.mark.parametrize("bad_index", [-1, True, "0", 1])
    def test_resume_with_invalid_index_executes_nothing(self, env, bad_index):
        """End-to-end regression for the index validation fix: an invalid decision
        index must not approve or mutate anything, even through the real graph."""
        graph, config = self._start_refund_escalation(env)

        final = graph.invoke(
            Command(resume={"decisions": [{"index": bad_index, "action": "approve"}]}),
            config=config,
        )

        env.esc_read.assert_not_called()
        env.esc_write.assert_not_called()
        env.wx_write.assert_not_called()
        ir = final["intent_results"][0]
        assert ir["status"] == "awaiting_human"
        assert "_newly_resolved" not in ir
        # follow_up_response found nothing newly resolved, so the turn-1 response stands
        assert final["response"] == INITIAL_RESPONSE


# ===================================================================
# Checkpoint / state lifecycle across turns
# ===================================================================

class TestCheckpointLifecycle:
    """Verify that the LangGraph checkpointer preserves correct state between turns
    while execution state is properly isolated per turn."""

    def test_separate_threads_have_independent_state(self, env):
        """Two different thread_ids must not share checkpointed state."""
        graph = build_graph()
        config_a = _config()
        config_b = _config()

        graph.invoke(_turn("Track my order ORD-MM-1001", config_a), config=config_a)
        graph.invoke(_turn("Track my order ORD-MM-1002", config_b), config=config_b)

        state_a = graph.get_state(config_a)
        state_b = graph.get_state(config_b)
        assert state_a.values["intent_results"][0]["entities"]["order_id"] == "ORD-MM-1001"
        assert state_b.values["intent_results"][0]["entities"]["order_id"] == "ORD-MM-1002"

    def test_conversation_store_survives_across_threads(self, env):
        """The conversation store is external to the checkpointer, so messages
        from thread A must not appear in thread B's history."""
        graph = build_graph()
        config_a = _config()
        config_b = _config()

        graph.invoke(
            {**_turn("Track my order ORD-MM-1001", config_a), "conversation_id": "conv-a"},
            config=config_a,
        )
        result_b = graph.invoke(
            {**_turn("Track my order ORD-MM-1002", config_b), "conversation_id": "conv-b"},
            config=config_b,
        )
        assert "ORD-MM-1001" not in str(result_b.get("conversation_history", []))

    def test_state_fully_terminal_after_complete_turn(self, env):
        """After a complete turn the graph should be in terminal state with no pending nodes."""
        graph = build_graph()
        config = _config()

        final = graph.invoke(_turn("Track my order ORD-MM-1001", config), config=config)

        snapshot = graph.get_state(config)
        assert snapshot.next == ()
        assert final["response_phase"] == "complete"
        assert final["pre_check_bail"] is False
