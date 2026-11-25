"""Tests for the in-memory conversation memory system."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from app.graph.conversation_store import ConversationStore, conversation_store, MAX_HISTORY
from app.graph.nodes.before_router import before_router_node
from app.graph.nodes.intent_executor import _extract_order_context
from app.graph.state import AgentState

T = "tenant-1"
C = "cust-1"
CONV = "conv-abc"


# ── 1. ConversationStore basics ─────────────────────────────────────

class TestConversationStore:
    def setup_method(self):
        self.store = ConversationStore()

    def test_empty_snapshot(self):
        snap = self.store.get_snapshot(T, C, CONV)
        assert snap == {"messages": [], "customer_context": {}, "orders_context": []}

    def test_add_human_message(self):
        self.store.add_human_message(T, C, CONV, "hello")
        snap = self.store.get_snapshot(T, C, CONV)
        assert len(snap["messages"]) == 1
        assert snap["messages"][0] == {"role": "human", "content": "hello"}

    def test_add_ai_response(self):
        self.store.add_human_message(T, C, CONV, "hi")
        self.store.add_ai_response(T, C, CONV, "hello!")
        snap = self.store.get_snapshot(T, C, CONV)
        assert len(snap["messages"]) == 2
        assert snap["messages"][1]["role"] == "assistant"

    def test_snapshot_is_deepcopy(self):
        self.store.add_human_message(T, C, CONV, "hi")
        snap = self.store.get_snapshot(T, C, CONV)
        snap["messages"].append({"role": "human", "content": "injected"})
        snap2 = self.store.get_snapshot(T, C, CONV)
        assert len(snap2["messages"]) == 1

    def test_trim_messages(self):
        for i in range(MAX_HISTORY + 5):
            self.store.add_human_message(T, C, CONV, f"msg-{i}")
        snap = self.store.get_snapshot(T, C, CONV)
        assert len(snap["messages"]) == MAX_HISTORY
        assert snap["messages"][0]["content"] == f"msg-{5}"

    def test_dedup_consecutive_human(self):
        self.store.add_human_message(T, C, CONV, "same")
        self.store.add_human_message(T, C, CONV, "same")
        snap = self.store.get_snapshot(T, C, CONV)
        assert len(snap["messages"]) == 1

    def test_dedup_does_not_block_different_messages(self):
        self.store.add_human_message(T, C, CONV, "first")
        self.store.add_human_message(T, C, CONV, "second")
        snap = self.store.get_snapshot(T, C, CONV)
        assert len(snap["messages"]) == 2


# ── 2. Composite key isolation ──────────────────────────────────────

class TestKeyIsolation:
    def setup_method(self):
        self.store = ConversationStore()

    def test_different_tenants_isolated(self):
        self.store.add_human_message("t1", C, CONV, "t1 msg")
        self.store.add_human_message("t2", C, CONV, "t2 msg")
        s1 = self.store.get_snapshot("t1", C, CONV)
        s2 = self.store.get_snapshot("t2", C, CONV)
        assert len(s1["messages"]) == 1
        assert s1["messages"][0]["content"] == "t1 msg"
        assert len(s2["messages"]) == 1
        assert s2["messages"][0]["content"] == "t2 msg"

    def test_different_customers_isolated(self):
        self.store.add_human_message(T, "c1", CONV, "c1 msg")
        self.store.add_human_message(T, "c2", CONV, "c2 msg")
        assert self.store.get_snapshot(T, "c1", CONV)["messages"][0]["content"] == "c1 msg"
        assert self.store.get_snapshot(T, "c2", CONV)["messages"][0]["content"] == "c2 msg"

    def test_different_conversations_isolated(self):
        self.store.add_human_message(T, C, "conv1", "msg1")
        self.store.add_human_message(T, C, "conv2", "msg2")
        assert self.store.get_snapshot(T, C, "conv1")["messages"][0]["content"] == "msg1"
        assert self.store.get_snapshot(T, C, "conv2")["messages"][0]["content"] == "msg2"


# ── 3. Customer/order context ───────────────────────────────────────

class TestContextUpdates:
    def setup_method(self):
        self.store = ConversationStore()

    def test_update_customer_context(self):
        self.store.update_customer_context(T, C, CONV, {"email": "a@b.com"})
        snap = self.store.get_snapshot(T, C, CONV)
        assert snap["customer_context"]["email"] == "a@b.com"

    def test_customer_context_merges(self):
        self.store.update_customer_context(T, C, CONV, {"email": "a@b.com"})
        self.store.update_customer_context(T, C, CONV, {"name": "Alice"})
        snap = self.store.get_snapshot(T, C, CONV)
        assert snap["customer_context"] == {"email": "a@b.com", "name": "Alice"}

    def test_update_orders_context(self):
        self.store.update_orders_context(T, C, CONV, [
            {"order_id": "ORD-1", "product": "Widget", "status": "delivered", "amount": 100},
        ])
        snap = self.store.get_snapshot(T, C, CONV)
        assert len(snap["orders_context"]) == 1
        assert snap["orders_context"][0]["order_id"] == "ORD-1"

    def test_orders_context_deduplicates(self):
        self.store.update_orders_context(T, C, CONV, [
            {"order_id": "ORD-1", "product": "Widget", "status": "delivered", "amount": 100},
        ])
        self.store.update_orders_context(T, C, CONV, [
            {"order_id": "ORD-1", "product": "Widget", "status": "refunded", "amount": 100},
        ])
        snap = self.store.get_snapshot(T, C, CONV)
        assert len(snap["orders_context"]) == 1
        assert snap["orders_context"][0]["status"] == "refunded"

    def test_orders_context_multiple_orders(self):
        self.store.update_orders_context(T, C, CONV, [
            {"order_id": "ORD-1", "product": "Widget", "status": "delivered", "amount": 100},
        ])
        self.store.update_orders_context(T, C, CONV, [
            {"order_id": "ORD-2", "product": "Gadget", "status": "shipped", "amount": 200},
        ])
        snap = self.store.get_snapshot(T, C, CONV)
        assert len(snap["orders_context"]) == 2


# ── 4. Thread safety ────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_writes(self):
        store = ConversationStore()
        errors: list[Exception] = []

        def writer(cid: str):
            try:
                for i in range(50):
                    store.add_human_message(T, cid, CONV, f"msg-{i}")
                    store.add_ai_response(T, cid, CONV, f"resp-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"c{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for i in range(5):
            snap = store.get_snapshot(T, f"c{i}", CONV)
            assert len(snap["messages"]) == MAX_HISTORY


# ── 5. before_router node ──────────────────────────────────────────

class TestBeforeRouterNode:
    def setup_method(self):
        conversation_store.reset()

    def test_first_message_empty_history(self):
        state: AgentState = {
            "tenant_id": T,
            "customer_id": C,
            "conversation_id": CONV,
            "user_message": "hello",
        }
        result = before_router_node(state)
        assert result["conversation_history"] == []
        assert result["customer_context"] == {}
        assert result["orders_context"] == []

    def test_second_message_has_history(self):
        conversation_store.add_human_message(T, C, CONV, "first question")
        conversation_store.add_ai_response(T, C, CONV, "first answer")

        state: AgentState = {
            "tenant_id": T,
            "customer_id": C,
            "conversation_id": CONV,
            "user_message": "second question",
        }
        result = before_router_node(state)
        assert len(result["conversation_history"]) == 2
        assert result["conversation_history"][0]["content"] == "first question"
        assert result["conversation_history"][1]["content"] == "first answer"

    def test_stores_human_message(self):
        state: AgentState = {
            "tenant_id": T,
            "customer_id": C,
            "conversation_id": CONV,
            "user_message": "hello",
        }
        before_router_node(state)
        snap = conversation_store.get_snapshot(T, C, CONV)
        assert len(snap["messages"]) == 1
        assert snap["messages"][0]["content"] == "hello"

    def test_preserves_order_context(self):
        conversation_store.update_orders_context(T, C, CONV, [
            {"order_id": "ORD-1", "product": "Widget", "status": "delivered", "amount": 100},
        ])
        state: AgentState = {
            "tenant_id": T,
            "customer_id": C,
            "conversation_id": CONV,
            "user_message": "what about my order?",
        }
        result = before_router_node(state)
        assert len(result["orders_context"]) == 1
        assert result["orders_context"][0]["order_id"] == "ORD-1"


# ── 6. Order context extraction ─────────────────────────────────────

class TestOrderContextExtraction:
    def test_extracts_from_get_order_details(self):
        tool_calls = [
            {
                "tool": "get_order_details",
                "args": {"order_id": "ORD-1"},
                "result": {"order_id": "ORD-1", "product": "Widget", "status": "delivered", "amount": 100},
            },
        ]
        orders = _extract_order_context(tool_calls)
        assert len(orders) == 1
        assert orders[0]["order_id"] == "ORD-1"

    def test_skips_blocked_calls(self):
        tool_calls = [
            {"tool": "get_order_details", "args": {}, "blocked": True, "reason": "not in allowlist"},
        ]
        assert _extract_order_context(tool_calls) == []

    def test_skips_non_order_tools(self):
        tool_calls = [
            {"tool": "get_return_eligibility", "args": {}, "result": {"eligible": True}},
        ]
        assert _extract_order_context(tool_calls) == []

    def test_skips_empty_order_id(self):
        tool_calls = [
            {"tool": "get_order_details", "args": {}, "result": {"product": "Widget"}},
        ]
        assert _extract_order_context(tool_calls) == []


# ── 7. Router context injection ─────────────────────────────────────

class TestRouterContextInjection:
    def test_context_prefix_with_history(self):
        from app.graph.nodes.router import _build_context_prefix

        state: AgentState = {
            "tenant_id": T,
            "user_message": "yes",
            "conversation_history": [
                {"role": "human", "content": "cancel order ORD-1"},
                {"role": "assistant", "content": "Are you sure?"},
            ],
        }
        prefix = _build_context_prefix(state)
        assert "previous conversations" in prefix
        assert "cancel order ORD-1" in prefix

    def test_context_prefix_with_orders(self):
        from app.graph.nodes.router import _build_context_prefix

        state: AgentState = {
            "tenant_id": T,
            "user_message": "refund it",
            "orders_context": [
                {"order_id": "ORD-1", "product": "Widget", "status": "delivered"},
            ],
        }
        prefix = _build_context_prefix(state)
        assert "Known orders:" in prefix
        assert "ORD-1" in prefix

    def test_context_prefix_empty_when_no_history(self):
        from app.graph.nodes.router import _build_context_prefix

        state: AgentState = {
            "tenant_id": T,
            "user_message": "hello",
        }
        prefix = _build_context_prefix(state)
        assert prefix == ""


# ── 8. ReAct agent does not inject context into messages ──────────

class TestReActNoContextInjection:
    def test_reason_node_no_context_system_message(self):
        """reason_node must not insert a 'Context from prior turns' system message."""
        from unittest.mock import MagicMock, patch

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        fake_response = AIMessage(content="Order ORD-1 is shipped.")
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = fake_response

        from app.graph.agent_prod.react_subgraph import reason_node

        state = {
            "messages": [
                SystemMessage(content="You are a tool-calling agent."),
                HumanMessage(content="Order: ORD-1\nRequest: cancel"),
            ],
            "tool_call_log": [],
            "iteration": 0,
            "final_result": "",
            "allowed_tools": ["get_order_details"],
            "order_id": "ORD-1",
            "constraints": {},
            "tenant_id": "t1",
            "customer_id": "c1",
            "pending_approval": {},
        }

        result = reason_node(state, {"configurable": {"llm": mock_llm}})

        invoked_messages = mock_llm.invoke.call_args[0][0]
        system_contents = [
            m.content for m in invoked_messages if isinstance(m, SystemMessage)
        ]
        assert not any("Context from prior turns" in c for c in system_contents)
        assert result["final_result"] == "Order ORD-1 is shipped."

    def test_react_agent_state_has_no_context_fields(self):
        """ReactAgentState should not carry conversation_history or orders_context."""
        from app.graph.agent_prod.react_subgraph import ReactAgentState

        fields = ReactAgentState.__annotations__
        assert "conversation_history" not in fields
        assert "orders_context" not in fields
        assert "customer_context" not in fields


# ── 9. AgentState has new fields ────────────────────────────────────

class TestAgentStateFields:
    def test_new_fields_exist(self):
        state: AgentState = {
            "tenant_id": T,
            "user_message": "hi",
            "conversation_id": "abc",
            "conversation_history": [],
            "customer_context": {},
            "orders_context": [],
        }
        assert state["conversation_id"] == "abc"
        assert state["conversation_history"] == []
        assert state["customer_context"] == {}
        assert state["orders_context"] == []


# ── 10. ReactAgentState has new fields ──────────────────────────────

class TestReactAgentStateFields:
    def test_new_fields_exist(self):
        from app.graph.agent_prod.react_subgraph import ReactAgentState

        state: ReactAgentState = {
            "messages": [],
            "tool_call_log": [],
            "iteration": 0,
            "final_result": "",
            "allowed_tools": [],
            "order_id": "",
            "constraints": {},
            "tenant_id": T,
            "customer_id": C,
            "pending_approval": {},
            "customer_context": {"email": "a@b.com"},
            "orders_context": [{"order_id": "ORD-1"}],
        }
        assert state["customer_context"] == {"email": "a@b.com"}


# ── 11. Clear and reset ─────────────────────────────────────────────

class TestClearReset:
    def setup_method(self):
        self.store = ConversationStore()

    def test_clear_single(self):
        self.store.add_human_message(T, C, CONV, "hi")
        self.store.clear(T, C, CONV)
        snap = self.store.get_snapshot(T, C, CONV)
        assert snap["messages"] == []

    def test_clear_does_not_affect_others(self):
        self.store.add_human_message(T, C, "conv1", "one")
        self.store.add_human_message(T, C, "conv2", "two")
        self.store.clear(T, C, "conv1")
        assert self.store.get_snapshot(T, C, "conv2")["messages"][0]["content"] == "two"

    def test_reset_clears_all(self):
        self.store.add_human_message(T, C, "conv1", "one")
        self.store.add_human_message(T, C, "conv2", "two")
        self.store.reset()
        assert self.store.get_snapshot(T, C, "conv1")["messages"] == []
        assert self.store.get_snapshot(T, C, "conv2")["messages"] == []


# ── 12. Graph wiring ────────────────────────────────────────────────

class TestGraphWiring:
    def test_before_router_exists_in_graph(self):
        from app.graph.builder import build_graph
        graph = build_graph()
        nodes = graph.get_graph().nodes
        assert "before_router" in nodes

    def test_before_router_before_router_node(self):
        from app.graph.builder import build_graph
        graph = build_graph()
        g = graph.get_graph()
        before_router_edges = [e for e in g.edges if e[0] == "before_router"]
        targets = {e[1] for e in before_router_edges}
        assert "router" in targets

    def test_start_goes_to_before_router(self):
        from app.graph.builder import build_graph
        graph = build_graph()
        g = graph.get_graph()
        start_edges = [e for e in g.edges if e[0] == "__start__"]
        targets = {e[1] for e in start_edges}
        assert "before_router" in targets


# ── 13. Send fan-out passes context ─────────────────────────────────

class TestFanOutContext:
    def test_route_intents_includes_context_fields(self):
        from app.graph.builder import route_intents

        state: AgentState = {
            "tenant_id": T,
            "customer_id": C,
            "user_message": "refund ORD-1",
            "conversation_id": CONV,
            "conversation_history": [{"role": "human", "content": "prior"}],
            "customer_context": {"email": "a@b.com"},
            "orders_context": [{"order_id": "ORD-1"}],
            "intent_results": [
                {
                    "intent": "refund",
                    "entities": {"order_id": "ORD-1"},
                    "policy_action": "allow",
                    "reason_code": "ALLOWED",
                    "constraints": {},
                    "status": "pending",
                },
            ],
        }
        sends = route_intents(state)
        assert len(sends) == 1
        payload = sends[0].arg
        assert payload["conversation_id"] == CONV
        assert "conversation_history" not in payload
        assert payload["orders_context"] == [{"order_id": "ORD-1"}]


# ── 14. Response agent stores response ──────────────────────────────

class TestResponseAgentStoresResponse:
    def setup_method(self):
        conversation_store.reset()

    @patch("app.graph.nodes.response_agent.ChatGroq")
    def test_stores_ai_response(self, mock_groq_cls):
        from app.graph.nodes.response_agent import response_agent_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Here is your answer.")
        mock_groq_cls.return_value = mock_llm

        state: AgentState = {
            "tenant_id": T,
            "customer_id": C,
            "conversation_id": CONV,
            "user_message": "refund my order",
            "intent_results": [
                {"intent": "refund", "status": "fulfilled", "domain_result": "Refund done."},
            ],
            "response_phase": "complete",
        }
        result = response_agent_node(state)
        assert result["response"] == "Here is your answer."

        snap = conversation_store.get_snapshot(T, C, CONV)
        assert len(snap["messages"]) == 1
        assert snap["messages"][0]["role"] == "assistant"
        assert snap["messages"][0]["content"] == "Here is your answer."


# ── 15. No customer_id treated as empty string ──────────────────────

class TestNullCustomerId:
    def setup_method(self):
        conversation_store.reset()

    def test_none_customer_id_works(self):
        state: AgentState = {
            "tenant_id": T,
            "customer_id": None,
            "conversation_id": CONV,
            "user_message": "hello",
        }
        result = before_router_node(state)
        assert result["conversation_history"] == []
        snap = conversation_store.get_snapshot(T, "", CONV)
        assert len(snap["messages"]) == 1


# ── 16. Empty conversation_id defaults gracefully ───────────────────

class TestEmptyConversationId:
    def setup_method(self):
        conversation_store.reset()

    def test_empty_conversation_id(self):
        state: AgentState = {
            "tenant_id": T,
            "customer_id": C,
            "user_message": "hi",
        }
        result = before_router_node(state)
        assert result["conversation_history"] == []


# ── 17. Singleton conversation_store ────────────────────────────────

class TestSingleton:
    def test_module_level_store_is_shared(self):
        from app.graph.conversation_store import conversation_store as store1
        from app.graph.nodes.before_router import conversation_store as store2
        from app.graph.nodes.response_agent import conversation_store as store3
        assert store1 is store2
        assert store1 is store3
