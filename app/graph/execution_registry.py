"""Server-side execution classification for intents and tools.

Every registered operation must have an explicit classification.
Unknown or unclassified operations fail closed (never dispatched to a write executor).
"""
from __future__ import annotations


class ExecutionType:
    READ = "read"
    WRITE = "write"


INTENT_EXECUTION_TYPE: dict[str, str] = {
    # Read intents
    "order_status": ExecutionType.READ,
    "faq": ExecutionType.READ,
    "complaint": ExecutionType.READ,
    "greeting": ExecutionType.READ,
    "thanks": ExecutionType.READ,
    "chitchat": ExecutionType.READ,
    "goodbye": ExecutionType.READ,
    "out_of_scope": ExecutionType.READ,

    # Write intents
    "refund": ExecutionType.WRITE,
    "cancel_order": ExecutionType.WRITE,
}

TOOL_EXECUTION_TYPE: dict[str, str] = {
    # Read tools
    "get_order_details": ExecutionType.READ,
    "get_return_eligibility": ExecutionType.READ,
    "get_order_tracking": ExecutionType.READ,
    "list_my_orders": ExecutionType.READ,

    # Write tools
    "initiate_refund": ExecutionType.WRITE,
    "cancel_order": ExecutionType.WRITE,
    "initiate_return": ExecutionType.WRITE,
    "get_shipping_label": ExecutionType.WRITE,
}

WRITE_INTENTS: frozenset[str] = frozenset(
    k for k, v in INTENT_EXECUTION_TYPE.items() if v == ExecutionType.WRITE
)

READ_TOOLS: frozenset[str] = frozenset(
    k for k, v in TOOL_EXECUTION_TYPE.items() if v == ExecutionType.READ
)

WRITE_TOOLS: frozenset[str] = frozenset(
    k for k, v in TOOL_EXECUTION_TYPE.items() if v == ExecutionType.WRITE
)


def get_execution_type(intent: str) -> str | None:
    return INTENT_EXECUTION_TYPE.get(intent)


def is_write_intent(intent: str) -> bool:
    return INTENT_EXECUTION_TYPE.get(intent) == ExecutionType.WRITE


def is_write_tool(tool_name: str) -> bool:
    return TOOL_EXECUTION_TYPE.get(tool_name) == ExecutionType.WRITE
