"""Tests for the processed_results reducer in app.graph.state.

Covers: None clears, lists append, supplied update order, no input mutation,
and that AgentState actually wires the field to this reducer.
"""
from __future__ import annotations

from typing import get_type_hints

from app.graph.state import AgentState, reduce_processed_results


class TestReduceProcessedResults:
    def test_none_update_clears_existing(self):
        assert reduce_processed_results([{"intent": "refund"}], None) == []

    def test_none_update_on_empty_returns_empty(self):
        assert reduce_processed_results([], None) == []

    def test_none_update_on_none_returns_empty(self):
        assert reduce_processed_results(None, None) == []

    def test_list_update_appends(self):
        current = [{"intent": "order_status"}]
        update = [{"intent": "faq"}]
        assert reduce_processed_results(current, update) == [{"intent": "order_status"}, {"intent": "faq"}]

    def test_list_update_onto_none_current(self):
        assert reduce_processed_results(None, [{"intent": "faq"}]) == [{"intent": "faq"}]

    def test_empty_list_update_keeps_current(self):
        current = [{"intent": "faq"}]
        assert reduce_processed_results(current, []) == [{"intent": "faq"}]

    def test_list_updates_accumulate_in_supplied_order(self):
        """The reducer concatenates updates in the order it receives them."""
        value = reduce_processed_results([], None)
        for branch in ("A", "B", "C"):
            value = reduce_processed_results(value, [{"branch": branch}])
        assert value == [{"branch": "A"}, {"branch": "B"}, {"branch": "C"}]

    def test_clear_then_append_starts_fresh(self):
        """A new turn (None) followed by that turn's results contains only that turn."""
        value = [{"turn": 1}]
        value = reduce_processed_results(value, None)
        value = reduce_processed_results(value, [{"turn": 2}])
        assert value == [{"turn": 2}]

    def test_inputs_not_mutated(self):
        current = [{"intent": "order_status"}]
        update = [{"intent": "faq"}]
        reduce_processed_results(current, update)
        assert current == [{"intent": "order_status"}]
        assert update == [{"intent": "faq"}]


class TestAgentStateWiring:
    def test_processed_results_uses_custom_reducer(self):
        hint = get_type_hints(AgentState, include_extras=True)["processed_results"]
        assert hint.__metadata__ == (reduce_processed_results,)
