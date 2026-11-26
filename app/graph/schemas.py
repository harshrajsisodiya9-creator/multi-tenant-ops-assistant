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

CONDITION_FIELDS = Literal[
    "order_status",
    "fulfillment_status",
    "is_refundable",
    "is_return_eligible",
]

CONDITION_OPERATORS = Literal[
    "equals",
    "not_equals",
    "in",
    "not_in",
    "is_true",
    "is_false",
]

REQUEST_TYPES = Literal[
    "conditional_action",
    "conditional_question",
]


class Condition(BaseModel):
    field: CONDITION_FIELDS
    operator: CONDITION_OPERATORS
    value: str | list[str] = ""


class ConditionalAction(BaseModel):
    request_type: REQUEST_TYPES
    condition: Condition


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
    conditional: ConditionalAction | None = None


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
    conditional: dict | None = None
