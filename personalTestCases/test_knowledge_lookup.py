"""Tests for knowledge_lookup_node.

Covers: successful RAG lookup, error handling, returned state structure.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.graph.nodes.intent_executor import _execute_faq, knowledge_lookup_node

from .conftest import TENANT_MEGAMART, make_base_state, make_intent_result


def _make_knowledge_state(intent_result: dict, user_message: str = "How do I return an item?") -> dict:
    state = make_base_state(user_message=user_message)
    state["current_intent"] = intent_result
    return state


class TestKnowledgeLookupNode:
    @patch("app.graph.nodes.intent_executor._execute_faq")
    def test_successful_lookup(self, mock_faq):
        mock_faq.return_value = "You can return items within 7 days of delivery."
        ir = make_intent_result(
            intent="faq", policy_action="allow", reason_code="FAQ_ALLOWED",
        )
        state = _make_knowledge_state(ir)
        result = knowledge_lookup_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert pr["domain_result"] == "You can return items within 7 days of delivery."

    @patch("app.graph.nodes.intent_executor._execute_faq")
    def test_no_answer_found(self, mock_faq):
        mock_faq.return_value = "I couldn't find an answer."
        ir = make_intent_result(
            intent="faq", policy_action="allow", reason_code="FAQ_ALLOWED",
        )
        state = _make_knowledge_state(ir)
        result = knowledge_lookup_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        assert "couldn't find" in pr["domain_result"].lower()

    @patch("app.graph.nodes.intent_executor._execute_faq")
    def test_complaint_intent_uses_rag(self, mock_faq):
        mock_faq.return_value = "We're sorry to hear about your experience."
        ir = make_intent_result(
            intent="complaint", policy_action="allow", reason_code="COMPLAINT_ALLOWED",
        )
        state = _make_knowledge_state(ir, user_message="I am very unhappy with my purchase")
        result = knowledge_lookup_node(state)

        pr = result["processed_results"][0]
        assert pr["status"] == "fulfilled"
        mock_faq.assert_called_once_with(TENANT_MEGAMART, "I am very unhappy with my purchase")

    @patch("app.graph.nodes.intent_executor._execute_faq")
    def test_returns_processed_results_list(self, mock_faq):
        mock_faq.return_value = "Answer text."
        ir = make_intent_result(intent="faq", policy_action="allow", reason_code="FAQ_ALLOWED")
        state = _make_knowledge_state(ir)
        result = knowledge_lookup_node(state)

        assert "processed_results" in result
        assert isinstance(result["processed_results"], list)
        assert len(result["processed_results"]) == 1

    @patch("app.graph.nodes.intent_executor._execute_faq")
    def test_intent_data_preserved_in_result(self, mock_faq):
        mock_faq.return_value = "Answer."
        ir = make_intent_result(
            intent="faq", policy_action="allow", reason_code="FAQ_ALLOWED",
            entities={"some_key": "some_value"},
        )
        state = _make_knowledge_state(ir)
        result = knowledge_lookup_node(state)
        pr = result["processed_results"][0]
        assert pr["intent"] == "faq"
        assert pr["entities"]["some_key"] == "some_value"

    @patch("app.graph.nodes.intent_executor._execute_faq", side_effect=RuntimeError("RAG unavailable"))
    def test_error_in_execute_faq_propagates(self, mock_faq):
        ir = make_intent_result(intent="faq", policy_action="allow", reason_code="FAQ_ALLOWED")
        state = _make_knowledge_state(ir)
        with pytest.raises(RuntimeError, match="RAG unavailable"):
            knowledge_lookup_node(state)


class TestExecuteFaqHelper:
    def test_valid_uuid_calls_rag_and_closes_session(self):
        tid = str(uuid4())
        mock_db = MagicMock()
        with (
            patch("app.database.SessionLocal", return_value=mock_db),
            patch("app.hybrid_search.answer_question", return_value={"answer": "7-day return window."}) as mock_rag,
        ):
            answer = _execute_faq(tid, "What is the return policy?")

        assert answer == "7-day return window."
        mock_rag.assert_called_once()
        mock_db.close.assert_called_once()

    def test_non_uuid_tenant_id_raises_value_error(self):
        mock_db = MagicMock()
        with patch("app.database.SessionLocal", return_value=mock_db):
            with pytest.raises(ValueError):
                _execute_faq("megamart", "What is the return policy?")
        mock_db.close.assert_called_once()
