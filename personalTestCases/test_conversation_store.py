"""Tests for ConversationStore.

Covers: message trimming, turn summary trimming, concurrent updates,
tenant/customer isolation, duplicate human message dedup,
orders context upsert, clear and reset.
"""
from __future__ import annotations

import threading

import pytest

import app.graph.conversation_store as conversation_store_module
from app.graph.conversation_store import MAX_HISTORY, MAX_TURN_SUMMARIES, ConversationStore


@pytest.fixture()
def store():
    return ConversationStore()


TENANT_A = "tenant-a"
TENANT_B = "tenant-b"
CUST_1 = "CUST-001"
CUST_2 = "CUST-002"
CONV = "conv-1"


# ===================================================================
# Basic operations
# ===================================================================

class TestBasicOperations:
    def test_empty_snapshot(self, store):
        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        assert snap["messages"] == []
        assert snap["customer_context"] == {}
        assert snap["orders_context"] == []
        assert snap["turn_summaries"] == []

    def test_add_human_then_ai(self, store):
        store.add_human_message(TENANT_A, CUST_1, CONV, "Hello")
        store.add_ai_response(TENANT_A, CUST_1, CONV, "Hi there!")
        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        assert len(snap["messages"]) == 2
        assert snap["messages"][0] == {"role": "human", "content": "Hello"}
        assert snap["messages"][1] == {"role": "assistant", "content": "Hi there!"}

    def test_snapshot_is_deep_copy(self, store):
        store.add_human_message(TENANT_A, CUST_1, CONV, "Hello")
        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        snap["messages"].append({"role": "human", "content": "injected"})
        snap2 = store.get_snapshot(TENANT_A, CUST_1, CONV)
        assert len(snap2["messages"]) == 1


# ===================================================================
# Duplicate human message dedup
# ===================================================================

class TestHumanMessageDedup:
    def test_consecutive_duplicate_ignored(self, store):
        store.add_human_message(TENANT_A, CUST_1, CONV, "Hello")
        store.add_human_message(TENANT_A, CUST_1, CONV, "Hello")
        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        assert len(snap["messages"]) == 1

    def test_non_consecutive_duplicate_kept(self, store):
        store.add_human_message(TENANT_A, CUST_1, CONV, "Hello")
        store.add_ai_response(TENANT_A, CUST_1, CONV, "Hi")
        store.add_human_message(TENANT_A, CUST_1, CONV, "Hello")
        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        assert len(snap["messages"]) == 3

    def test_different_message_not_deduped(self, store):
        store.add_human_message(TENANT_A, CUST_1, CONV, "Hello")
        store.add_human_message(TENANT_A, CUST_1, CONV, "Help me")
        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        assert len(snap["messages"]) == 2


# ===================================================================
# Message trimming
# ===================================================================

class TestMessageTrimming:
    def test_trims_beyond_max_history(self, store):
        for i in range(MAX_HISTORY + 5):
            store.add_human_message(TENANT_A, CUST_1, CONV, f"msg-{i}")
            store.add_ai_response(TENANT_A, CUST_1, CONV, f"reply-{i}")
        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        assert len(snap["messages"]) == MAX_HISTORY

    def test_oldest_messages_trimmed(self, store):
        for i in range(MAX_HISTORY + 2):
            store.add_ai_response(TENANT_A, CUST_1, CONV, f"msg-{i}")
        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        assert snap["messages"][0]["content"] == f"msg-{2}"
        assert snap["messages"][-1]["content"] == f"msg-{MAX_HISTORY + 1}"


# ===================================================================
# Turn summary trimming
# ===================================================================

class TestTurnSummaryTrimming:
    def test_trims_beyond_max_turn_summaries(self, store):
        for i in range(MAX_TURN_SUMMARIES + 5):
            store.add_turn_summary(TENANT_A, CUST_1, CONV, [{"intent": f"intent-{i}"}])
        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        assert len(snap["turn_summaries"]) == MAX_TURN_SUMMARIES
        assert snap["turn_summaries"][0]["intent"] == "intent-5"

    def test_batch_summaries_extended(self, store):
        store.add_turn_summary(TENANT_A, CUST_1, CONV, [
            {"intent": "order_status"},
            {"intent": "refund"},
        ])
        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        assert len(snap["turn_summaries"]) == 2


# ===================================================================
# Orders context upsert
# ===================================================================

class TestOrdersContext:
    def test_new_orders_appended(self, store):
        store.update_orders_context(TENANT_A, CUST_1, CONV, [
            {"order_id": "ORD-1", "product": "Widget"},
        ])
        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        assert len(snap["orders_context"]) == 1
        assert snap["orders_context"][0]["order_id"] == "ORD-1"

    def test_existing_order_updated(self, store):
        store.update_orders_context(TENANT_A, CUST_1, CONV, [
            {"order_id": "ORD-1", "product": "Widget", "status": "pending"},
        ])
        store.update_orders_context(TENANT_A, CUST_1, CONV, [
            {"order_id": "ORD-1", "product": "Widget", "status": "shipped"},
        ])
        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        assert len(snap["orders_context"]) == 1
        assert snap["orders_context"][0]["status"] == "shipped"

    def test_order_without_id_skipped(self, store):
        store.update_orders_context(TENANT_A, CUST_1, CONV, [
            {"product": "Widget"},
        ])
        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        assert len(snap["orders_context"]) == 0

    def test_customer_context_merged(self, store):
        store.update_customer_context(TENANT_A, CUST_1, CONV, {"name": "Alice"})
        store.update_customer_context(TENANT_A, CUST_1, CONV, {"email": "alice@example.com"})
        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        assert snap["customer_context"] == {"name": "Alice", "email": "alice@example.com"}


# ===================================================================
# Tenant / customer isolation
# ===================================================================

class TestTenantIsolation:
    def test_different_tenants_isolated(self, store):
        store.add_human_message(TENANT_A, CUST_1, CONV, "tenant-a message")
        store.add_human_message(TENANT_B, CUST_1, CONV, "tenant-b message")
        snap_a = store.get_snapshot(TENANT_A, CUST_1, CONV)
        snap_b = store.get_snapshot(TENANT_B, CUST_1, CONV)
        assert len(snap_a["messages"]) == 1
        assert snap_a["messages"][0]["content"] == "tenant-a message"
        assert snap_b["messages"][0]["content"] == "tenant-b message"

    def test_different_customers_isolated(self, store):
        store.add_human_message(TENANT_A, CUST_1, CONV, "cust-1 message")
        store.add_human_message(TENANT_A, CUST_2, CONV, "cust-2 message")
        snap_1 = store.get_snapshot(TENANT_A, CUST_1, CONV)
        snap_2 = store.get_snapshot(TENANT_A, CUST_2, CONV)
        assert snap_1["messages"][0]["content"] == "cust-1 message"
        assert snap_2["messages"][0]["content"] == "cust-2 message"

    def test_different_conversations_isolated(self, store):
        store.add_human_message(TENANT_A, CUST_1, "conv-1", "first conv")
        store.add_human_message(TENANT_A, CUST_1, "conv-2", "second conv")
        snap_1 = store.get_snapshot(TENANT_A, CUST_1, "conv-1")
        snap_2 = store.get_snapshot(TENANT_A, CUST_1, "conv-2")
        assert len(snap_1["messages"]) == 1
        assert snap_1["messages"][0]["content"] == "first conv"
        assert snap_2["messages"][0]["content"] == "second conv"


# ===================================================================
# Clear and reset
# ===================================================================

class TestClearAndReset:
    def test_clear_removes_single_conversation(self, store):
        store.add_human_message(TENANT_A, CUST_1, "conv-1", "msg")
        store.add_human_message(TENANT_A, CUST_1, "conv-2", "msg")
        store.clear(TENANT_A, CUST_1, "conv-1")
        snap_1 = store.get_snapshot(TENANT_A, CUST_1, "conv-1")
        snap_2 = store.get_snapshot(TENANT_A, CUST_1, "conv-2")
        assert snap_1["messages"] == []
        assert len(snap_2["messages"]) == 1

    def test_reset_clears_all(self, store):
        store.add_human_message(TENANT_A, CUST_1, CONV, "msg")
        store.add_human_message(TENANT_B, CUST_2, CONV, "msg")
        store.reset()
        assert store.get_snapshot(TENANT_A, CUST_1, CONV)["messages"] == []
        assert store.get_snapshot(TENANT_B, CUST_2, CONV)["messages"] == []


# ===================================================================
# Concurrent updates
# ===================================================================

class TestConcurrentUpdates:
    def test_concurrent_messages_no_data_loss(self, store, monkeypatch):
        """Multiple threads writing messages concurrently must not lose data.

        Temporarily lift the normal history cap so every write can be checked.
        """
        num_threads = 10
        messages_per_thread = 20
        total_written = num_threads * messages_per_thread
        monkeypatch.setattr(conversation_store_module, "MAX_HISTORY", total_written)
        barrier = threading.Barrier(num_threads)

        def writer(thread_id: int) -> None:
            barrier.wait()
            for i in range(messages_per_thread):
                store.add_ai_response(TENANT_A, CUST_1, CONV, f"t{thread_id}-msg{i}")

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        contents = [m["content"] for m in snap["messages"]]
        expected = {
            f"t{thread_id}-msg{i}"
            for thread_id in range(num_threads)
            for i in range(messages_per_thread)
        }
        assert len(contents) == total_written
        assert set(contents) == expected

    def test_concurrent_messages_all_same_role(self, store):
        """Every message surviving a concurrent write must retain its role."""
        num_threads = 5
        barrier = threading.Barrier(num_threads)

        def writer(thread_id: int) -> None:
            barrier.wait()
            for i in range(10):
                store.add_ai_response(TENANT_A, CUST_1, CONV, f"t{thread_id}-{i}")

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        assert all(m["role"] == "assistant" for m in snap["messages"])

    def test_concurrent_context_updates_no_data_loss(self, store):
        """Concurrent customer_context updates must merge without losing keys."""
        num_threads = 10
        barrier = threading.Barrier(num_threads)

        def updater(thread_id: int) -> None:
            barrier.wait()
            store.update_customer_context(
                TENANT_A, CUST_1, CONV, {f"key_{thread_id}": f"val_{thread_id}"},
            )

        threads = [threading.Thread(target=updater, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = store.get_snapshot(TENANT_A, CUST_1, CONV)
        for tid in range(num_threads):
            assert snap["customer_context"][f"key_{tid}"] == f"val_{tid}"
