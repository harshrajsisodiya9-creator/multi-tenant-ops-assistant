"""Tests for response_agent_node.

Covers: empty intent_results, normal response composition, follow-up phase,
turn summary building, conversation store interaction.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.graph.nodes.response_agent import (
    _build_turn_summary,
    _format_turn_summaries,
    response_agent_node,
)

from .conftest import CUSTOMER_ALICE, TENANT_MEGAMART, make_base_state


# ===================================================================
# _format_turn_summaries
# ===================================================================

class TestFormatTurnSummaries:
    def test_empty(self):
        assert _format_turn_summaries([]) == ""

    def test_single_summary(self):
        summaries = [{"intent": "refund", "order_id": "ORD-1", "status": "fulfilled", "reason_code": ""}]
        result = _format_turn_summaries(summaries)
        assert "refund" in result
        assert "ORD-1" in result
        assert "fulfilled" in result

    def test_summary_with_reason_code(self):
        summaries = [{"intent": "refund", "order_id": "", "status": "blocked", "reason_code": "RETURNS_NOT_ALLOWED"}]
        result = _format_turn_summaries(summaries)
        assert "RETURNS_NOT_ALLOWED" in result

    def test_summary_without_order_id_omits_for_clause(self):
        summaries = [{"intent": "faq", "order_id": "", "status": "fulfilled", "reason_code": ""}]
        result = _format_turn_summaries(summaries)
        assert "- faq (fulfilled)" in result
        assert "faq for " not in result


# ===================================================================
# _build_turn_summary
# ===================================================================

class TestBuildTurnSummary:
    def test_builds_from_intent_results(self):
        results = [
            {"intent": "refund", "entities": {"order_id": "ORD-1"}, "status": "fulfilled", "reason_code": "REFUND_ALLOWED"},
            {"intent": "greeting", "entities": {}, "status": "fulfilled", "reason_code": ""},
        ]
        summaries = _build_turn_summary(results)
        assert len(summaries) == 2
        assert summaries[0]["intent"] == "refund"
        assert summaries[0]["order_id"] == "ORD-1"
        assert summaries[1]["intent"] == "greeting"
        assert summaries[1]["order_id"] == ""


# ===================================================================
# response_agent_node
# ===================================================================

class TestResponseAgentNode:
    def test_empty_intent_results_returns_error(self):
        state = make_base_state()
        state["intent_results"] = []
        state["response_phase"] = "complete"
        result = response_agent_node(state)
        assert "sorry" in result["response"].lower() or "couldn't" in result["response"].lower()

    def test_missing_intent_results_returns_error(self):
        state = make_base_state()
        state["response_phase"] = "complete"
        result = response_agent_node(state)
        assert "sorry" in result["response"].lower() or "couldn't" in result["response"].lower()

    @patch("app.graph.nodes.response_agent.ChatGroq")
    @patch("app.graph.nodes.response_agent.conversation_store")
    def test_normal_phase_invokes_llm(self, mock_store, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_response = MagicMock()
        mock_response.content = "Your order ORD-MM-1001 was delivered on Sept 20."
        mock_llm.invoke.return_value = mock_response

        state = make_base_state(user_message="Where is my order ORD-MM-1001?")
        state["response_phase"] = "complete"
        state["intent_results"] = [
            {
                "intent": "order_status",
                "status": "fulfilled",
                "domain_result": "Order ORD-MM-1001 (Wireless Headphones) status: delivered.",
            },
        ]

        result = response_agent_node(state)
        assert result["response"] == "Your order ORD-MM-1001 was delivered on Sept 20."
        mock_llm.invoke.assert_called_once()

        call_args = mock_llm.invoke.call_args[0][0]
        user_content = call_args[1]["content"]
        assert "Wireless Headphones" in user_content
        assert "delivered" in user_content

    @patch("app.graph.nodes.response_agent.ChatGroq")
    @patch("app.graph.nodes.response_agent.conversation_store")
    def test_conversation_store_updated(self, mock_store, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_response = MagicMock()
        mock_response.content = "Response text."
        mock_llm.invoke.return_value = mock_response

        state = make_base_state(conversation_id="conv-123")
        state["response_phase"] = "complete"
        state["intent_results"] = [
            {"intent": "greeting", "status": "fulfilled", "domain_result": "Hello!"},
        ]

        response_agent_node(state)
        mock_store.add_ai_response.assert_called_once_with(
            TENANT_MEGAMART, CUSTOMER_ALICE, "conv-123", "Response text.",
        )
        mock_store.add_turn_summary.assert_called_once()

    @patch("app.graph.nodes.response_agent.ChatGroq")
    @patch("app.graph.nodes.response_agent.conversation_store")
    def test_no_store_update_without_conversation_id(self, mock_store, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_response = MagicMock()
        mock_response.content = "Response text."
        mock_llm.invoke.return_value = mock_response

        state = make_base_state(conversation_id="")
        state["response_phase"] = "complete"
        state["intent_results"] = [
            {"intent": "greeting", "status": "fulfilled", "domain_result": "Hello!"},
        ]

        response_agent_node(state)
        mock_store.add_ai_response.assert_not_called()

    @patch("app.graph.nodes.response_agent.ChatGroq")
    @patch("app.graph.nodes.response_agent.conversation_store")
    def test_follow_up_phase_uses_newly_resolved(self, mock_store, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_response = MagicMock()
        mock_response.content = "Your refund has been approved."
        mock_llm.invoke.return_value = mock_response

        state = make_base_state()
        state["response_phase"] = "follow_up"
        state["intent_results"] = [
            {
                "intent": "order_status",
                "status": "fulfilled",
                "domain_result": "Delivered.",
            },
            {
                "intent": "refund",
                "status": "fulfilled",
                "domain_result": "Refund of 1000 processed.",
                "_newly_resolved": True,
            },
        ]

        result = response_agent_node(state)
        assert result["response"] == "Your refund has been approved."
        call_args = mock_llm.invoke.call_args[0][0]
        system_prompt = call_args[0]["content"]
        assert "FOLLOW-UP" in system_prompt

        user_content = call_args[1]["content"]
        assert "Refund of 1000 processed" in user_content
        assert "Delivered." not in user_content

    def test_follow_up_no_newly_resolved_returns_empty(self):
        state = make_base_state()
        state["response_phase"] = "follow_up"
        state["intent_results"] = [
            {"intent": "order_status", "status": "fulfilled", "domain_result": "Delivered."},
        ]
        result = response_agent_node(state)
        assert result == {}

    @patch("app.graph.nodes.response_agent.ChatGroq")
    @patch("app.graph.nodes.response_agent.conversation_store")
    def test_turn_summaries_included_in_context(self, mock_store, mock_groq_cls):
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_response = MagicMock()
        mock_response.content = "Here is your info."
        mock_llm.invoke.return_value = mock_response

        state = make_base_state()
        state["response_phase"] = "complete"
        state["turn_summaries"] = [
            {"intent": "order_status", "order_id": "ORD-MM-1001", "status": "fulfilled", "reason_code": ""},
        ]
        state["intent_results"] = [
            {"intent": "refund", "status": "fulfilled", "domain_result": "Refund done."},
        ]

        response_agent_node(state)
        call_args = mock_llm.invoke.call_args[0][0]
        user_content = call_args[1]["content"]
        assert "Previously resolved" in user_content

    @patch("app.graph.nodes.response_agent.ChatGroq")
    def test_llm_invocation_failure_propagates(self, mock_groq_cls):
        """When ChatGroq.invoke raises, the exception propagates (no internal try/except)."""
        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_llm.invoke.side_effect = RuntimeError("Groq API down")

        state = make_base_state()
        state["response_phase"] = "complete"
        state["intent_results"] = [
            {"intent": "greeting", "status": "fulfilled", "domain_result": "Hello!"},
        ]
        with pytest.raises(RuntimeError, match="Groq API down"):
            response_agent_node(state)
