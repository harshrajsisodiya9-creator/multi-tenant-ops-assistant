from __future__ import annotations

from typing import Annotated, TypedDict


def reduce_processed_results(
    current: list[dict] | None, update: list[dict] | None,
) -> list[dict]:
    # None clears the list so each turn starts empty; lists append so parallel Send() branches merge.
    if update is None:
        return []
    return (current or []) + update


class AgentState(TypedDict, total=False):
    tenant_id: str
    customer_id: str | None
    user_message: str
    customer_email: str | None
    conversation_id: str
    conversation_history: list[dict]
    customer_context: dict
    orders_context: list[dict]
    turn_summaries: list[dict]
    router_output: dict | None
    policy_decision: dict | None
    response: str
    intent_results: list[dict]
    response_phase: str
    pre_check_bail: bool
    current_intent: dict
    processed_results: Annotated[list[dict], reduce_processed_results]
    has_tool_calls: bool
