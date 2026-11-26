"""Tests for router_node and its helper functions.

Covers: entity extraction, escalation detection, keyword matching,
conversational detection, pre-check fast-path, LLM fallback,
conditional detection, order-id normalization.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.graph.nodes.router import (
    _build_context_prefix,
    _check_conversational,
    _count_keyword_matches,
    _detect_conditional,
    _detect_conditional_question,
    _detect_escalation,
    _extract_entities,
    _has_conjunction,
    _is_informational,
    _normalize_order_entities,
    _pre_check,
    router_node,
)

from .conftest import CUSTOMER_ALICE, TENANT_MEGAMART, make_base_state


# ===================================================================
# _extract_entities
# ===================================================================

class TestExtractEntities:
    def test_single_order_id(self):
        entities = _extract_entities("Where is my order ORD-MM-1001?")
        assert entities["order_id"] == "ORD-MM-1001"
        assert "all_order_ids" not in entities

    def test_multiple_order_ids(self):
        entities = _extract_entities("Check ORD-MM-1001 and ORD-MM-1002")
        assert entities["order_id"] == "ORD-MM-1001"
        assert entities["all_order_ids"] == ["ORD-MM-1001", "ORD-MM-1002"]

    def test_hash_style_order_id(self):
        entities = _extract_entities("Status of order #12345")
        assert entities["order_id"] == "#12345"

    def test_email_extraction(self):
        entities = _extract_entities("My email is alice@example.com")
        assert entities["email"] == "alice@example.com"

    def test_no_entities(self):
        entities = _extract_entities("I want a refund please")
        assert entities == {}

    def test_both_order_and_email(self):
        entities = _extract_entities("Order ORD-MM-1001, email: test@test.com")
        assert entities["order_id"] == "ORD-MM-1001"
        assert entities["email"] == "test@test.com"


# ===================================================================
# _normalize_order_entities
# ===================================================================

class TestNormalizeOrderEntities:
    def test_list_order_id_normalized_to_first(self):
        target = {"order_id": ["ord-mm-1001", "ord-mm-1002"]}
        _normalize_order_entities(target, {})
        assert target["order_id"] == "ORD-MM-1001"
        assert target["all_order_ids"] == ["ORD-MM-1001", "ORD-MM-1002"]

    def test_single_string_order_id(self):
        target = {"order_id": "ord-mm-1001"}
        _normalize_order_entities(target, {})
        assert target["order_id"] == "ORD-MM-1001"
        assert "all_order_ids" not in target

    def test_regex_ids_merged(self):
        target = {"order_id": "ORD-MM-1001"}
        regex_entities = {"all_order_ids": ["ORD-MM-1001", "ORD-MM-1002"]}
        _normalize_order_entities(target, regex_entities)
        assert target["order_id"] == "ORD-MM-1001"
        assert target["all_order_ids"] == ["ORD-MM-1001", "ORD-MM-1002"]

    def test_empty_order_id_cleaned(self):
        target = {"order_id": ""}
        _normalize_order_entities(target, {})
        assert "order_id" not in target


# ===================================================================
# _detect_escalation
# ===================================================================

class TestDetectEscalation:
    def test_no_escalation(self):
        assert _detect_escalation("Where is my order?") == []

    def test_legal_keyword(self):
        result = _detect_escalation("I'll contact my lawyer about this")
        assert "lawyer" in result

    def test_multiple_keywords(self):
        result = _detect_escalation("This is fraud, I'll sue you in court")
        assert "fraud" in result
        assert "sue" in result
        assert "court" in result

    def test_case_insensitive(self):
        result = _detect_escalation("CHARGEBACK NOW")
        assert "chargeback" in result


# ===================================================================
# _is_informational / _has_conjunction / _check_conversational
# ===================================================================

class TestHelperFunctions:
    def test_is_informational_how_to(self):
        assert _is_informational("how to return an item") is True

    def test_is_informational_negative(self):
        assert _is_informational("cancel my order") is False

    def test_has_conjunction(self):
        assert _has_conjunction("cancel my order and also refund it") is True

    def test_has_conjunction_negative(self):
        assert _has_conjunction("cancel my order") is False

    def test_check_conversational_greeting(self):
        result = _check_conversational("hello there")
        assert result is not None
        assert result[0] == "greeting"

    def test_check_conversational_thanks(self):
        result = _check_conversational("thank you so much")
        assert result is not None
        assert result[0] == "thanks"

    def test_check_conversational_goodbye(self):
        result = _check_conversational("goodbye for now")
        assert result is not None
        assert result[0] == "goodbye"

    def test_check_conversational_none(self):
        assert _check_conversational("refund order ORD-MM-1001") is None


class TestCountKeywordMatches:
    def test_order_status_keyword(self):
        matches = _count_keyword_matches("track my order please")
        assert any(m[0] == "order_status" for m in matches)

    def test_cancel_keyword(self):
        matches = _count_keyword_matches("cancel my order now")
        assert any(m[0] == "cancel_order" for m in matches)

    def test_refund_keyword(self):
        matches = _count_keyword_matches("i want a refund")
        assert any(m[0] == "refund" for m in matches)

    def test_no_keyword_match(self):
        matches = _count_keyword_matches("hello how are you")
        assert matches == []


# ===================================================================
# _detect_conditional / _detect_conditional_question
# ===================================================================

class TestDetectConditional:
    def test_detect_conditional_action(self):
        assert _detect_conditional("if it hasn't shipped, cancel it") is True

    def test_detect_conditional_negative(self):
        assert _detect_conditional("cancel my order") is False

    def test_detect_conditional_question(self):
        assert _detect_conditional_question("is my order shipped yet?") is True

    def test_detect_conditional_question_negative(self):
        assert _detect_conditional_question("cancel my order") is False


# ===================================================================
# _pre_check (combined fast-path logic)
# ===================================================================

class TestPreCheck:
    def test_greeting_fast_path(self):
        intent, entities, confidence, escalation, is_cond = _pre_check("Hello!")
        assert intent == "greeting"
        assert confidence >= 0.9

    def test_informational_falls_through(self):
        intent, entities, confidence, _, _ = _pre_check("How to return an item?")
        assert intent is None

    def test_single_keyword_match(self):
        intent, entities, confidence, _, _ = _pre_check("Track my order ORD-MM-1001")
        assert intent == "order_status"
        assert entities.get("order_id") == "ORD-MM-1001"

    def test_multi_keyword_falls_through(self):
        intent, _, _, _, _ = _pre_check("Cancel my order and also refund it")
        assert intent is None

    def test_conjunction_with_single_match_falls_through(self):
        intent, _, _, _, _ = _pre_check("Refund my order and also check shipping status")
        assert intent is None

    def test_escalation_detected_alongside(self):
        _, _, _, escalation, _ = _pre_check("I want a refund or I'll sue")
        assert "sue" in escalation

    def test_conditional_detected(self):
        _, _, _, _, is_cond = _pre_check("If it hasn't shipped, cancel it")
        assert is_cond is True


# ===================================================================
# router_node
# ===================================================================

class TestRouterNode:
    def test_fast_path_greeting(self):
        state = make_base_state(user_message="Hello!")
        state["router_output"] = None
        result = router_node(state)
        ro = result["router_output"]
        assert ro["intent"] == "greeting"
        assert ro["confidence"] >= 0.9

    def test_fast_path_order_status(self):
        state = make_base_state(user_message="Track my order ORD-MM-1001")
        result = router_node(state)
        ro = result["router_output"]
        assert ro["intent"] == "order_status"
        assert ro["entities"]["order_id"] == "ORD-MM-1001"

    @patch("app.graph.nodes.router.ChatGroq")
    def test_llm_fallback_invoked_for_ambiguous(self, mock_groq_cls):
        """When _pre_check can't resolve, router_node should call the LLM."""
        from app.graph.schemas import RouterOutput as RO

        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = RO(
            intent="refund",
            entities={"order_id": "ORD-MM-1001"},
            confidence=0.92,
        )
        state = make_base_state(user_message="I need help with something complex")
        result = router_node(state)

        mock_llm.with_structured_output.assert_called_once_with(RO)
        mock_structured.invoke.assert_called_once()
        messages = mock_structured.invoke.call_args[0][0]
        assert any("I need help with something complex" in m.get("content", "") for m in messages)

        ro = result["router_output"]
        assert ro["intent"] == "refund"
        assert ro["entities"]["order_id"] == "ORD-MM-1001"
        assert ro["confidence"] == 0.92

    @patch("app.graph.nodes.router.ChatGroq")
    def test_regex_entities_merged_into_llm_result(self, mock_groq_cls):
        from app.graph.schemas import RouterOutput as RO

        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = RO(
            intent="refund",
            entities={},
            confidence=0.85,
        )
        state = make_base_state(
            user_message="Refund order ORD-MM-1001, email alice@example.com and also cancel",
        )
        result = router_node(state)
        entities = result["router_output"]["entities"]
        assert entities.get("order_id") == "ORD-MM-1001"
        assert entities.get("email") == "alice@example.com"

    @patch("app.graph.nodes.router.ChatGroq")
    def test_escalation_signals_merged(self, mock_groq_cls):
        from app.graph.schemas import RouterOutput as RO

        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = RO(
            intent="complaint",
            entities={},
            confidence=0.80,
            escalation_signals=[],
        )
        state = make_base_state(user_message="This is fraud, I want my money back now")
        result = router_node(state)
        assert "fraud" in result["router_output"]["escalation_signals"]

    def test_fast_path_skipped_for_conditional(self):
        """Conditional messages should NOT take the fast path even if keyword matches."""
        state = make_base_state(user_message="If it hasn't shipped, cancel my order ORD-MM-1001")
        with patch("app.graph.nodes.router.ChatGroq") as mock_groq_cls:
            from app.graph.schemas import RouterOutput as RO
            mock_llm = MagicMock()
            mock_groq_cls.return_value = mock_llm
            mock_structured = MagicMock()
            mock_llm.with_structured_output.return_value = mock_structured
            mock_structured.invoke.return_value = RO(
                intent="cancel_order",
                entities={"order_id": "ORD-MM-1001"},
                confidence=0.92,
            )
            result = router_node(state)
            mock_structured.invoke.assert_called_once()
            ro = result["router_output"]
            assert ro["intent"] == "cancel_order"
            assert ro["entities"]["order_id"] == "ORD-MM-1001"
            assert ro["confidence"] == 0.92

    @patch("app.graph.nodes.router.ChatGroq")
    def test_refund_below_0_9_confidence_falls_through_to_llm(self, mock_groq_cls):
        """_KEYWORD_RULES gives refund 0.88 confidence (< 0.9), so router_node uses the LLM."""
        from app.graph.schemas import RouterOutput as RO

        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = RO(
            intent="refund",
            entities={"order_id": "ORD-MM-1001"},
            confidence=0.95,
        )
        state = make_base_state(user_message="I want a refund for ORD-MM-1001")
        result = router_node(state)
        mock_structured.invoke.assert_called_once()
        assert result["router_output"]["intent"] == "refund"

    @patch("app.graph.nodes.router.ChatGroq")
    def test_sub_intents_order_entities_normalized(self, mock_groq_cls):
        from app.graph.schemas import RouterOutput as RO

        mock_llm = MagicMock()
        mock_groq_cls.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = RO(
            intent="order_status",
            entities={"order_id": "ord-mm-1001"},
            confidence=0.9,
            sub_intents=[
                {"intent": "cancel_order", "entities": {"order_id": ["ord-mm-1002", "ord-mm-1003"]}, "confidence": 0.88},
            ],
        )
        state = make_base_state(user_message="Check ORD-MM-1001 and cancel ORD-MM-1002 and ORD-MM-1003")
        result = router_node(state)
        sub = result["router_output"]["sub_intents"][0]
        assert sub["entities"]["order_id"] == "ORD-MM-1002"
        assert sub["entities"]["all_order_ids"] == ["ORD-MM-1002", "ORD-MM-1003"]


class TestBuildContextPrefix:
    def test_empty_context(self):
        assert _build_context_prefix(make_base_state()) == ""

    def test_includes_history_and_orders(self):
        state = make_base_state(
            conversation_history=[
                {"role": "human", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            orders_context=[
                {"order_id": "ORD-MM-1001", "product": "Headphones", "status": "delivered"},
            ],
        )
        prefix = _build_context_prefix(state)
        assert "Customer: Hi" in prefix
        assert "Assistant: Hello!" in prefix
        assert "ORD-MM-1001 — Headphones (delivered)" in prefix
