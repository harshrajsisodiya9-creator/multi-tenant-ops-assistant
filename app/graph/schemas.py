from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

INTENT_TYPES = Literal[
    "faq",
    "order_status",
    "refund",
    "cancel_order",
    "complaint",
    "out_of_scope",
    "greeting",
    "thanks",
    "chitchat",
    "goodbye",
]


class SubIntent(BaseModel):
    intent: INTENT_TYPES
    entities: dict = {}
    confidence: float


class RouterOutput(BaseModel):
    intent: INTENT_TYPES
    entities: dict = {}
    confidence: float
    escalation_signals: list[str] = []
    sub_intents: list[SubIntent] = []


class PolicyDecision(BaseModel):
    action: Literal["allow", "block", "escalate", "clarify"]
    reason_code: str
    constraints: dict = {}


class IntentResult(BaseModel):
    intent: str
    entities: dict = {}
    policy_action: str
    reason_code: str
    constraints: dict = {}
    domain_result: str = ""
    status: str = "pending"
    tool_calls: list[dict] = []
