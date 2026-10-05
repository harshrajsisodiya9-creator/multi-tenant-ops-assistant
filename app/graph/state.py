from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    tenant_id: str
    user_message: str
    customer_email: str | None
    router_output: dict | None
    policy_decision: dict | None
    response: str
    intent_results: list[dict]
    response_phase: str
    pre_check_bail: bool
