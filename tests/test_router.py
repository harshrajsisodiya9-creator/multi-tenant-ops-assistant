from __future__ import annotations

import pytest

from app.graph.nodes import (
    _count_keyword_matches,
    _detect_escalation,
    _extract_entities,
    _has_conjunction,
    _is_informational,
    _pre_check,
)

ENTITY_CASES = [
    ("My order ORD-MM-1001 hasn't arrived", {"order_id": "ORD-MM-1001"}),
    ("Order #12345 is wrong", {"order_id": "#12345"}),
    ("Contact me at alice@example.com", {"email": "alice@example.com"}),
    ("Order ORD-TC-2001 email bob@test.com", {"order_id": "ORD-TC-2001", "email": "bob@test.com"}),
    ("Hello, I need help", {}),
]


@pytest.mark.parametrize("message, expected", ENTITY_CASES)
def test_extract_entities(message: str, expected: dict):
    result = _extract_entities(message)
    for key, val in expected.items():
        assert result.get(key) == val


def test_extract_multiple_order_ids():
    result = _extract_entities("Check ORD-MM-1001 and ORD-MM-1003")
    assert result["order_id"] == "ORD-MM-1001"
    assert result["all_order_ids"] == ["ORD-MM-1001", "ORD-MM-1003"]


def test_extract_single_order_no_all_key():
    result = _extract_entities("Check ORD-MM-1001")
    assert result["order_id"] == "ORD-MM-1001"
    assert "all_order_ids" not in result


ESCALATION_CASES = [
    ("I will file a chargeback", ["chargeback"]),
    ("My lawyer will contact you", ["lawyer"]),
    ("This is fraud and I will sue", ["fraud", "sue"]),
    ("I want a refund please", []),
    ("I'll take legal action", ["legal"]),
]


@pytest.mark.parametrize("message, expected", ESCALATION_CASES)
def test_detect_escalation(message: str, expected: list[str]):
    result = _detect_escalation(message)
    assert set(result) == set(expected)


PRE_CHECK_CASES = [
    ("Where is my order #1234?", "order_status", True),
    ("Track my order ORD-MM-1001", "order_status", True),
    ("I want to cancel my order", "cancel_order", True),
    ("Cancel my subscription", "cancel_order", True),
    ("I want a refund for my order", "refund", False),
    ("Can you return my item?", "refund", False),
    ("What is your return policy?", None, False),
    ("Tell me a joke", None, False),
    ("How do I unsubscribe?", None, False),
    ("What is the cancellation policy?", None, False),
    ("How can I return an item?", None, False),
    ("Tell me about the refund process", None, False),
    ("What are the steps to cancel my order?", None, False),
]


INFORMATIONAL_CASES = [
    ("how do I unsubscribe", True),
    ("what is the cancellation policy", True),
    ("how can I return an item", True),
    ("tell me about the refund process", True),
    ("explain your return policy", True),
    ("what's the procedure for cancellation", True),
    ("cancel my order", False),
    ("I want a refund", False),
    ("track my order", False),
    ("unsubscribe now", False),
]


@pytest.mark.parametrize("message, expected", INFORMATIONAL_CASES)
def test_is_informational(message: str, expected: bool):
    assert _is_informational(message.lower()) == expected


@pytest.mark.parametrize("message, expected_intent, high_confidence", PRE_CHECK_CASES)
def test_pre_check(message: str, expected_intent: str | None, high_confidence: bool):
    intent, entities, confidence, escalation = _pre_check(message)
    assert intent == expected_intent
    if high_confidence:
        assert confidence >= 0.9


# ── Multi-intent detection ──────────────────────────────────────────


CONJUNCTION_CASES = [
    ("I want a refund and also cancel my order", True),
    ("Refund this plus check my order status", True),
    ("Where is my order as well as I need a return", True),
    ("I want a refund for my order", False),
    ("Cancel my order", False),
]


@pytest.mark.parametrize("message, expected", CONJUNCTION_CASES)
def test_has_conjunction(message: str, expected: bool):
    assert _has_conjunction(message.lower()) == expected


MULTI_KEYWORD_CASES = [
    ("I want a refund and cancel my order", 2),
    ("Track my order and also get a refund", 2),
    ("Where is my order", 1),
    ("Tell me a joke", 0),
    ("Cancel my order", 1),
]


@pytest.mark.parametrize("message, expected_count", MULTI_KEYWORD_CASES)
def test_count_keyword_matches(message: str, expected_count: int):
    matches = _count_keyword_matches(message.lower())
    assert len(matches) == expected_count


def test_pre_check_multi_intent_forces_llm():
    intent, entities, confidence, escalation = _pre_check(
        "I want a refund and cancel my order"
    )
    assert intent is None
    assert confidence == 0.0


def test_pre_check_conjunction_with_single_keyword_forces_llm():
    intent, entities, confidence, escalation = _pre_check(
        "Cancel my order and also check something"
    )
    assert intent is None
    assert confidence == 0.0


def test_pre_check_single_intent_still_works():
    intent, entities, confidence, escalation = _pre_check(
        "Track my order ORD-MM-1001"
    )
    assert intent == "order_status"
    assert confidence >= 0.9


ROUTER_EVAL_DATASET = [
    {"message": "What is your return policy?", "expected_intent": "faq"},
    {"message": "Do you ship internationally?", "expected_intent": "faq"},
    {"message": "What payment methods do you accept?", "expected_intent": "faq"},
    {"message": "How long does delivery take?", "expected_intent": "faq"},
    {"message": "What are your business hours?", "expected_intent": "faq"},
    {"message": "Are your products organic?", "expected_intent": "faq"},
    {"message": "Where is my order ORD-MM-1001?", "expected_intent": "order_status"},
    {"message": "Track my order #5678", "expected_intent": "order_status"},
    {"message": "When will my package arrive?", "expected_intent": "order_status"},
    {"message": "What's the shipping status of ORD-TC-2001?", "expected_intent": "order_status"},
    {"message": "Has my order been shipped yet?", "expected_intent": "order_status"},
    {"message": "I want a refund for order ORD-MM-1001", "expected_intent": "refund"},
    {"message": "Can I return my headphones?", "expected_intent": "refund"},
    {"message": "I'd like my money back", "expected_intent": "refund"},
    {"message": "Return my recent purchase please", "expected_intent": "refund"},
    {"message": "This product is defective, I want a refund", "expected_intent": "refund"},
    {"message": "I need to return an item I bought last week", "expected_intent": "refund"},
    {"message": "Cancel my order ORD-MM-1007", "expected_intent": "cancel_order"},
    {"message": "I want to cancel my subscription", "expected_intent": "cancel_order"},
    {"message": "Please cancel order #9999", "expected_intent": "cancel_order"},
    {"message": "I changed my mind, cancel the order", "expected_intent": "cancel_order"},
    {"message": "Stop my subscription immediately", "expected_intent": "cancel_order"},
    {"message": "The product arrived damaged and this is unacceptable!", "expected_intent": "complaint"},
    {"message": "I'm very unhappy with the service quality", "expected_intent": "complaint"},
    {"message": "Your delivery was extremely late", "expected_intent": "complaint"},
    {"message": "The wrong item was sent to me", "expected_intent": "complaint"},
    {"message": "I keep getting the runaround from support", "expected_intent": "complaint"},
    {"message": "What's the weather like today?", "expected_intent": "out_of_scope"},
    {"message": "Tell me a joke", "expected_intent": "out_of_scope"},
    {"message": "Can you write me a poem?", "expected_intent": "out_of_scope"},
    {"message": "Who won the cricket match?", "expected_intent": "out_of_scope"},
    {"message": "Help me with my homework", "expected_intent": "out_of_scope"},
    {"message": "I want to return this AND cancel my other order", "expected_intent": "refund"},
    {"message": "Where is my order? I might sue if it doesn't come", "expected_intent": "order_status"},
]
