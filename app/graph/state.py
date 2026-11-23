from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    tenant_id: str
    customer_id: str | None
    user_message: str
    customer_email: str | None
    router_output: dict | None
    policy_decision: dict | None
    response: str
    intent_results: list[dict]
    response_phase: str
    pre_check_bail: bool
    current_intent: dict
    processed_results: Annotated[list[dict], operator.add]
    has_tool_calls: bool
