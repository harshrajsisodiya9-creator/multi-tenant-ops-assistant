"""Tests for route_intents and the builder routing functions.

Covers: Send() dispatch logic for each executor type, conditional routing,
after_pre_check, after_merge, after_response_agent, after_escalation.
"""
from __future__ import annotations

import pytest

from app.graph.builder import (
    after_escalation,
    after_merge,
    after_pre_check,
    after_response_agent,
    route_intents,
)

from .conftest import (
    CUSTOMER_ALICE,
    TENANT_MEGAMART,
    make_base_state,
    make_intent_result,
)


# ===================================================================
# after_pre_check
# ===================================================================

class TestAfterPreCheck:
    def test_bail_when_pre_check_bail_true(self):
        state = {"pre_check_bail": True}
        assert after_pre_check(state) == "bail"

    def test_continue_when_pre_check_bail_false(self):
        state = {"pre_check_bail": False}
        assert after_pre_check(state) == "continue"

    def test_continue_when_pre_check_bail_missing(self):
        state = {}
        assert after_pre_check(state) == "continue"


# ===================================================================
# route_intents (Send fan-out)
# ===================================================================

class TestRouteIntents:
    def _make_state_with_intents(self, intent_results: list[dict]) -> dict:
        state = make_base_state(user_message="test message")
        state["intent_results"] = intent_results
        return state

    def test_order_status_routes_to_quick_resolver(self):
        intents = [make_intent_result(
            intent="order_status", policy_action="allow",
            reason_code="ORDER_STATUS_ALLOWED",
            entities={"order_id": "ORD-MM-1001"},
        )]
        sends = route_intents(self._make_state_with_intents(intents))
        assert len(sends) == 1
        assert sends[0].node == "quick_resolver"

    def test_faq_routes_to_knowledge_lookup(self):
        intents = [make_intent_result(
            intent="faq", policy_action="allow",
            reason_code="FAQ_ALLOWED",
        )]
        sends = route_intents(self._make_state_with_intents(intents))
        assert len(sends) == 1
        assert sends[0].node == "knowledge_lookup"

    def test_complaint_routes_to_knowledge_lookup(self):
        intents = [make_intent_result(
            intent="complaint", policy_action="allow",
            reason_code="COMPLAINT_ALLOWED",
        )]
        sends = route_intents(self._make_state_with_intents(intents))
        assert len(sends) == 1
        assert sends[0].node == "knowledge_lookup"

    def test_refund_routes_to_write_executor(self):
        intents = [make_intent_result(
            intent="refund", policy_action="allow",
            reason_code="REFUND_ALLOWED",
            entities={"order_id": "ORD-MM-1001"},
        )]
        sends = route_intents(self._make_state_with_intents(intents))
        assert len(sends) == 1
        assert sends[0].node == "write_executor"

    def test_cancel_order_routes_to_write_executor(self):
        intents = [make_intent_result(
            intent="cancel_order", policy_action="allow",
            reason_code="CANCEL_ALLOWED",
            entities={"order_id": "ORD-MM-1001"},
        )]
        sends = route_intents(self._make_state_with_intents(intents))
        assert len(sends) == 1
        assert sends[0].node == "write_executor"

    def test_conditional_routes_to_conditional_executor(self):
        intents = [make_intent_result(
            intent="cancel_order", policy_action="allow",
            reason_code="CANCEL_ALLOWED",
            entities={"order_id": "ORD-MM-1001"},
            conditional={
                "request_type": "conditional_action",
                "condition": {"field": "order_status", "operator": "not_equals", "value": "shipped"},
            },
        )]
        sends = route_intents(self._make_state_with_intents(intents))
        assert len(sends) == 1
        assert sends[0].node == "conditional_executor"

    def test_blocked_routes_to_quick_resolver(self):
        intents = [make_intent_result(
            intent="refund", policy_action="block",
            reason_code="RETURNS_NOT_ALLOWED",
        )]
        sends = route_intents(self._make_state_with_intents(intents))
        assert len(sends) == 1
        assert sends[0].node == "quick_resolver"

    def test_escalate_routes_to_quick_resolver(self):
        intents = [make_intent_result(
            intent="complaint", policy_action="escalate",
            reason_code="ESCALATION_KEYWORDS_DETECTED",
        )]
        sends = route_intents(self._make_state_with_intents(intents))
        assert len(sends) == 1
        assert sends[0].node == "quick_resolver"

    def test_conversational_intent_routes_to_quick_resolver(self):
        for intent in ("greeting", "thanks", "goodbye", "chitchat"):
            intents = [make_intent_result(
                intent=intent, policy_action="allow",
                reason_code=f"{intent.upper()}_ALLOWED",
            )]
            sends = route_intents(self._make_state_with_intents(intents))
            assert len(sends) == 1
            assert sends[0].node == "quick_resolver", f"Failed for {intent}"

    def test_multiple_intents_produce_multiple_sends(self):
        intents = [
            make_intent_result(intent="order_status", policy_action="allow",
                               reason_code="ORDER_STATUS_ALLOWED", entities={"order_id": "ORD-1"}),
            make_intent_result(intent="refund", policy_action="allow",
                               reason_code="REFUND_ALLOWED", entities={"order_id": "ORD-2"}),
        ]
        sends = route_intents(self._make_state_with_intents(intents))
        assert len(sends) == 2
        nodes = {s.node for s in sends}
        assert "quick_resolver" in nodes
        assert "write_executor" in nodes

    def test_send_payload_carries_current_intent(self):
        """Each Send() must carry current_intent with the correct entities."""
        intents = [make_intent_result(
            intent="order_status", policy_action="allow",
            reason_code="ORDER_STATUS_ALLOWED",
            entities={"order_id": "ORD-MM-1001"},
        )]
        sends = route_intents(self._make_state_with_intents(intents))
        assert len(sends) == 1
        payload = sends[0].arg
        assert payload["current_intent"]["intent"] == "order_status"
        assert payload["current_intent"]["entities"]["order_id"] == "ORD-MM-1001"

    def test_non_pending_status_skipped(self):
        intents = [make_intent_result(
            intent="order_status", policy_action="allow",
            reason_code="ORDER_STATUS_ALLOWED", status="fulfilled",
        )]
        sends = route_intents(self._make_state_with_intents(intents))
        assert len(sends) == 0

    def test_agent_intents_empty_routes_to_quick_resolver(self):
        """AGENT_INTENTS is currently empty, so no intent goes to lookup_agent."""
        from app.graph.nodes.intent_executor import AGENT_INTENTS
        assert len(AGENT_INTENTS) == 0


# ===================================================================
# after_merge
# ===================================================================

class TestAfterMerge:
    def test_needs_action_check_when_tool_calls(self):
        state = {"has_tool_calls": True}
        assert after_merge(state) == "needs_action_check"

    def test_skip_action_check_when_no_tool_calls(self):
        state = {"has_tool_calls": False}
        assert after_merge(state) == "skip_action_check"

    def test_skip_when_missing(self):
        state = {}
        assert after_merge(state) == "skip_action_check"


# ===================================================================
# after_response_agent
# ===================================================================

class TestAfterResponseAgent:
    def test_complete_when_complete(self):
        state = {"response_phase": "complete"}
        assert after_response_agent(state) == "complete"

    def test_needs_escalation_when_partial(self):
        state = {"response_phase": "partial"}
        assert after_response_agent(state) == "needs_escalation"

    def test_complete_when_missing(self):
        state = {}
        assert after_response_agent(state) == "complete"


# ===================================================================
# after_escalation
# ===================================================================

class TestAfterEscalation:
    def test_end_when_not_follow_up(self):
        state = {"response_phase": "complete"}
        assert after_escalation(state) == "end"

    def test_follow_up_when_no_hitl_writes(self):
        state = {
            "response_phase": "follow_up",
            "intent_results": [
                {"_newly_resolved": True, "tool_calls": []},
            ],
        }
        assert after_escalation(state) == "follow_up"

    def test_hitl_action_check_when_hitl_writes(self):
        state = {
            "response_phase": "follow_up",
            "intent_results": [
                {"_newly_resolved": True, "tool_calls": [{"tool": "initiate_refund"}]},
            ],
        }
        assert after_escalation(state) == "hitl_action_check"

    def test_follow_up_when_empty_intent_results(self):
        """Empty intent_results with follow_up phase: no HITL writes → follow_up."""
        state = {"response_phase": "follow_up", "intent_results": []}
        assert after_escalation(state) == "follow_up"
