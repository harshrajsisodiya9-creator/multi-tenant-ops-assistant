from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.graph.agent.react_agent import (
    _action_check_in_loop,
    _parse_agent_response,
    run_react_agent,
)


class TestParseAgentResponse:
    def test_valid_tool_call(self):
        text = '{"tool": "get_order_details", "args": {"order_id": "ORD-1"}}'
        result = _parse_agent_response(text)
        assert result["tool"] == "get_order_details"

    def test_valid_final_answer(self):
        text = '{"final_answer": "Your refund has been processed."}'
        result = _parse_agent_response(text)
        assert result["final_answer"] == "Your refund has been processed."

    def test_multiline_with_json(self):
        text = 'Let me check.\n{"tool": "get_order_details", "args": {"order_id": "ORD-1"}}'
        result = _parse_agent_response(text)
        assert result["tool"] == "get_order_details"

    def test_invalid_json(self):
        text = "I don't understand, can you clarify?"
        result = _parse_agent_response(text)
        assert result is None

    def test_empty_string(self):
        result = _parse_agent_response("")
        assert result is None


class TestActionCheckInLoop:
    def test_refund_within_limit(self):
        result = _action_check_in_loop(
            "initiate_refund", {"amount": 5000}, {"max_amount": 10000}
        )
        assert result is None

    def test_refund_exceeds_limit(self):
        result = _action_check_in_loop(
            "initiate_refund", {"amount": 15000}, {"max_amount": 10000}
        )
        assert result is not None
        assert "exceeds" in result.lower()

    def test_wrong_refund_type(self):
        result = _action_check_in_loop(
            "initiate_refund",
            {"amount": 100, "refund_type": "store_credit"},
            {"refund_type": "original_method"},
        )
        assert result is not None
        assert "original_method" in result

    def test_cancel_blocked(self):
        result = _action_check_in_loop(
            "cancel_order", {}, {"cancel_blocked": True}
        )
        assert result is not None

    def test_cancel_allowed(self):
        result = _action_check_in_loop(
            "cancel_order", {}, {}
        )
        assert result is None

    def test_unknown_tool_no_check(self):
        result = _action_check_in_loop(
            "get_order_details", {}, {"max_amount": 10000}
        )
        assert result is None


class TestRunReactAgent:
    def _mock_llm_responses(self, responses: list[str]):
        mock_llm = MagicMock()
        mock_responses = []
        for r in responses:
            mock_msg = MagicMock()
            mock_msg.content = r
            mock_responses.append(mock_msg)
        mock_llm.invoke = MagicMock(side_effect=mock_responses)
        return mock_llm

    def test_simple_refund_flow(self):
        llm = self._mock_llm_responses([
            '{"tool": "get_order_details", "args": {"order_id": "ORD-MM-1001"}}',
            '{"tool": "get_return_eligibility", "args": {"order_id": "ORD-MM-1001"}}',
            '{"tool": "initiate_refund", "args": {"order_id": "ORD-MM-1001", "amount": 4500, "refund_type": "original_method"}}',
            '{"final_answer": "Your refund of ₹4,500 for Wireless Headphones has been processed."}',
        ])

        result = run_react_agent(
            tenant_id="megamart",
            order_id="ORD-MM-1001",
            user_message="I want a refund for my headphones",
            intent="refund",
            constraints={"refund_type": "original_method"},
            llm=llm,
        )

        assert "4,500" in result["result"]
        assert len(result["tool_calls"]) == 3
        assert result["iterations"] == 4

    def test_cancel_flow(self):
        llm = self._mock_llm_responses([
            '{"tool": "get_order_details", "args": {"order_id": "ORD-MM-1007"}}',
            '{"tool": "cancel_order", "args": {"order_id": "ORD-MM-1007"}}',
            '{"final_answer": "Your order ORD-MM-1007 has been cancelled."}',
        ])

        result = run_react_agent(
            tenant_id="megamart",
            order_id="ORD-MM-1007",
            user_message="Cancel my order",
            intent="cancel_order",
            llm=llm,
        )

        assert "cancelled" in result["result"].lower()
        assert len(result["tool_calls"]) == 2

    def test_unknown_tool_handled(self):
        llm = self._mock_llm_responses([
            '{"tool": "nonexistent_tool", "args": {}}',
            '{"final_answer": "Sorry, I could not process that."}',
        ])

        result = run_react_agent(
            tenant_id="megamart",
            order_id="ORD-MM-1001",
            user_message="refund",
            intent="refund",
            llm=llm,
        )

        assert result["iterations"] == 2
        assert len(result["tool_calls"]) == 0

    def test_action_check_blocks_tool(self):
        llm = self._mock_llm_responses([
            '{"tool": "get_order_details", "args": {"order_id": "ORD-MM-1001"}}',
            '{"tool": "initiate_refund", "args": {"order_id": "ORD-MM-1001", "amount": 4500, "refund_type": "store_credit"}}',
            '{"tool": "initiate_refund", "args": {"order_id": "ORD-MM-1001", "amount": 4500, "refund_type": "original_method"}}',
            '{"final_answer": "Refund processed via original method."}',
        ])

        result = run_react_agent(
            tenant_id="megamart",
            order_id="ORD-MM-1001",
            user_message="refund",
            intent="refund",
            constraints={"refund_type": "original_method"},
            llm=llm,
        )

        blocked_calls = [tc for tc in result["tool_calls"] if tc.get("blocked")]
        assert len(blocked_calls) == 1
        assert blocked_calls[0]["tool"] == "initiate_refund"

    def test_max_iterations_reached(self):
        llm = self._mock_llm_responses(
            ['{"tool": "get_order_details", "args": {"order_id": "ORD-MM-1001"}}'] * 10
        )

        result = run_react_agent(
            tenant_id="megamart",
            order_id="ORD-MM-1001",
            user_message="refund",
            intent="refund",
            llm=llm,
        )

        assert result["iterations"] == 8
        assert "maximum iterations" in result["result"].lower()

    def test_non_json_response_returns_text(self):
        llm = self._mock_llm_responses([
            "I apologize but I cannot process this request.",
        ])

        result = run_react_agent(
            tenant_id="megamart",
            order_id="ORD-MM-1001",
            user_message="refund",
            intent="refund",
            llm=llm,
        )

        assert "cannot process" in result["result"].lower()
        assert result["iterations"] == 1

    def test_amount_exceeds_max_blocked(self):
        llm = self._mock_llm_responses([
            '{"tool": "initiate_refund", "args": {"order_id": "ORD-MM-1001", "amount": 15000}}',
            '{"final_answer": "The refund amount exceeds the maximum allowed."}',
        ])

        result = run_react_agent(
            tenant_id="megamart",
            order_id="ORD-MM-1001",
            user_message="refund 15000",
            intent="refund",
            constraints={"max_amount": 10000},
            llm=llm,
        )

        blocked = [tc for tc in result["tool_calls"] if tc.get("blocked")]
        assert len(blocked) == 1

    def test_item_level_refund(self):
        llm = self._mock_llm_responses([
            '{"tool": "get_order_details", "args": {"order_id": "ORD-MM-1009"}}',
            '{"tool": "get_return_eligibility", "args": {"order_id": "ORD-MM-1009", "item_id": "ITM-1009-1"}}',
            '{"tool": "initiate_refund", "args": {"order_id": "ORD-MM-1009", "amount": 2800, "refund_type": "original_method", "item_id": "ITM-1009-1"}}',
            '{"final_answer": "Refund of ₹2,800 processed for Wireless Mouse."}',
        ])

        result = run_react_agent(
            tenant_id="megamart",
            order_id="ORD-MM-1009",
            user_message="refund the mouse from my order",
            intent="refund",
            constraints={"refund_type": "original_method"},
            llm=llm,
        )

        assert "2,800" in result["result"]
        assert len(result["tool_calls"]) == 3
