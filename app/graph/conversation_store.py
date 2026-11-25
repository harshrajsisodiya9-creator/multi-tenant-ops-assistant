from __future__ import annotations

import threading
from copy import deepcopy

MAX_HISTORY = 10


class ConversationStore:
    def __init__(self) -> None:
        self._store: dict[tuple[str, str, str], dict] = {}
        self._lock = threading.Lock()

    def _key(self, tenant_id: str, customer_id: str, conversation_id: str) -> tuple[str, str, str]:
        return (tenant_id, customer_id or "", conversation_id)

    def get_snapshot(self, tenant_id: str, customer_id: str, conversation_id: str) -> dict:
        key = self._key(tenant_id, customer_id, conversation_id)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return {"messages": [], "customer_context": {}, "orders_context": []}
            return deepcopy(entry)

    def add_human_message(self, tenant_id: str, customer_id: str, conversation_id: str, content: str) -> None:
        key = self._key(tenant_id, customer_id, conversation_id)
        with self._lock:
            entry = self._store.setdefault(key, {"messages": [], "customer_context": {}, "orders_context": []})
            if entry["messages"] and entry["messages"][-1].get("role") == "human" and entry["messages"][-1].get("content") == content:
                return
            entry["messages"].append({"role": "human", "content": content})
            self._trim(entry)

    def add_ai_response(self, tenant_id: str, customer_id: str, conversation_id: str, content: str) -> None:
        key = self._key(tenant_id, customer_id, conversation_id)
        with self._lock:
            entry = self._store.setdefault(key, {"messages": [], "customer_context": {}, "orders_context": []})
            entry["messages"].append({"role": "assistant", "content": content})
            self._trim(entry)

    def update_customer_context(self, tenant_id: str, customer_id: str, conversation_id: str, context: dict) -> None:
        key = self._key(tenant_id, customer_id, conversation_id)
        with self._lock:
            entry = self._store.setdefault(key, {"messages": [], "customer_context": {}, "orders_context": []})
            entry["customer_context"].update(context)

    def update_orders_context(self, tenant_id: str, customer_id: str, conversation_id: str, orders: list[dict]) -> None:
        key = self._key(tenant_id, customer_id, conversation_id)
        with self._lock:
            entry = self._store.setdefault(key, {"messages": [], "customer_context": {}, "orders_context": []})
            existing_ids = {o["order_id"] for o in entry["orders_context"]}
            for order in orders:
                oid = order.get("order_id")
                if not oid:
                    continue
                if oid in existing_ids:
                    entry["orders_context"] = [
                        order if o["order_id"] == oid else o
                        for o in entry["orders_context"]
                    ]
                else:
                    entry["orders_context"].append(order)
                    existing_ids.add(oid)

    def _trim(self, entry: dict) -> None:
        if len(entry["messages"]) > MAX_HISTORY:
            entry["messages"] = entry["messages"][-MAX_HISTORY:]

    def clear(self, tenant_id: str, customer_id: str, conversation_id: str) -> None:
        key = self._key(tenant_id, customer_id, conversation_id)
        with self._lock:
            self._store.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._store.clear()


conversation_store = ConversationStore()
