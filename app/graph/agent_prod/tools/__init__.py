from app.graph.agent_prod.tools.order_tools import (
    cancel_order,
    get_order_details,
    get_order_tracking,
    get_return_eligibility,
    get_shipping_label,
    initiate_refund,
    initiate_return,
    list_my_orders,
)

TOOL_REGISTRY: dict[str, callable] = {
    "get_order_details": get_order_details,
    "get_return_eligibility": get_return_eligibility,
    "initiate_return": initiate_return,
    "initiate_refund": initiate_refund,
    "get_shipping_label": get_shipping_label,
    "cancel_order": cancel_order,
    "get_order_tracking": get_order_tracking,
    "list_my_orders": list_my_orders,
}

TOOL_DESCRIPTIONS: list[dict] = [
    {
        "name": "get_order_details",
        "description": "Look up full order details including items, status, dates, and amounts.",
        "parameters": ["order_id"],
    },
    {
        "name": "get_return_eligibility",
        "description": "Check if an order or specific item is eligible for return based on tenant policy.",
        "parameters": ["order_id", "item_id (optional)"],
    },
    {
        "name": "initiate_return",
        "description": "Start a return process for an order or specific item.",
        "parameters": ["order_id", "item_id (optional)", "reason (optional)"],
    },
    {
        "name": "initiate_refund",
        "description": "Process a refund for a given amount via the specified refund method.",
        "parameters": ["order_id", "amount", "refund_type", "item_id (optional)"],
    },
    {
        "name": "get_shipping_label",
        "description": "Generate a return shipping label for the order.",
        "parameters": ["order_id"],
    },
    {
        "name": "cancel_order",
        "description": "Cancel a pending/processing order.",
        "parameters": ["order_id"],
    },
    {
        "name": "get_order_tracking",
        "description": "Get tracking information for a shipped or delivered order.",
        "parameters": ["order_id"],
    },
    {
        "name": "list_my_orders",
        "description": "List all orders for the current customer.",
        "parameters": [],
    },
]
