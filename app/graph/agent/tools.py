from __future__ import annotations

from datetime import date

from app.graph.mock_data import get_order, get_order_item
from app.graph.policy_loader import resolve_policy


def get_order_details(tenant_id: str, order_id: str) -> dict:
    order = get_order(tenant_id, order_id)
    if not order:
        return {"error": f"Order {order_id} not found."}
    return {
        "order_id": order["order_id"],
        "customer_email": order["customer_email"],
        "order_status": order.get("order_status", order.get("status")),
        "total_amount": order.get("total_amount", order.get("amount")),
        "order_date": order.get("order_date"),
        "delivery_date": order.get("delivery_date"),
        "items": order.get("items", []),
    }


def get_return_eligibility(
    tenant_id: str, order_id: str, item_id: str | None = None
) -> dict:
    order = get_order(tenant_id, order_id)
    if not order:
        return {"eligible": False, "reason": f"Order {order_id} not found."}

    if item_id:
        item = get_order_item(tenant_id, order_id, item_id)
        if not item:
            return {"eligible": False, "reason": f"Item {item_id} not found in order {order_id}."}
        category = item["category"]
        amount = item["amount"]
    else:
        category = order.get("category", "")
        amount = order.get("amount", 0)

    policy = resolve_policy(tenant_id, category)

    if not policy.return_policy.allowed:
        return {
            "eligible": False,
            "reason": f"Returns not allowed for {category}.",
            "category": category,
        }

    delivery_date_str = order.get("delivery_date")
    if delivery_date_str:
        delivery_date = date.fromisoformat(delivery_date_str)
        days_since = (date.today() - delivery_date).days
        if days_since > policy.return_policy.window_days:
            return {
                "eligible": False,
                "reason": f"Return window expired ({policy.return_policy.window_days} days). Delivered {days_since} days ago.",
                "category": category,
                "window_days": policy.return_policy.window_days,
                "days_since_delivery": days_since,
            }

    return {
        "eligible": True,
        "category": category,
        "amount": amount,
        "window_days": policy.return_policy.window_days,
        "condition": policy.return_policy.condition,
        "refund_type": policy.return_policy.refund_type,
    }


def initiate_return(
    tenant_id: str, order_id: str, item_id: str | None = None, reason: str = ""
) -> dict:
    order = get_order(tenant_id, order_id)
    if not order:
        return {"success": False, "error": f"Order {order_id} not found."}

    if item_id:
        item = get_order_item(tenant_id, order_id, item_id)
        if not item:
            return {"success": False, "error": f"Item {item_id} not found."}
        product = item["product"]
    else:
        product = order.get("product", "item")

    return {
        "success": True,
        "return_id": f"RET-{order_id}-001",
        "product": product,
        "message": f"Return initiated for {product} from order {order_id}.",
    }


def initiate_refund(
    tenant_id: str,
    order_id: str,
    amount: int | float,
    refund_type: str = "original_method",
    item_id: str | None = None,
) -> dict:
    order = get_order(tenant_id, order_id)
    if not order:
        return {"success": False, "error": f"Order {order_id} not found."}

    if item_id:
        item = get_order_item(tenant_id, order_id, item_id)
        if not item:
            return {"success": False, "error": f"Item {item_id} not found."}
        product = item["product"]
    else:
        product = order.get("product", "item")

    return {
        "success": True,
        "refund_id": f"REF-{order_id}-001",
        "amount": amount,
        "refund_type": refund_type,
        "product": product,
        "message": f"Refund of ₹{amount:,.0f} initiated for {product} ({order_id}) via {refund_type}.",
    }


def get_shipping_label(tenant_id: str, order_id: str) -> dict:
    order = get_order(tenant_id, order_id)
    if not order:
        return {"success": False, "error": f"Order {order_id} not found."}
    return {
        "success": True,
        "label_url": f"https://shipping.example.com/label/{order_id}",
        "carrier": order.get("shipping_carrier", "Standard"),
        "message": f"Shipping label generated for return of order {order_id}.",
    }


def cancel_order(tenant_id: str, order_id: str) -> dict:
    order = get_order(tenant_id, order_id)
    if not order:
        return {"success": False, "error": f"Order {order_id} not found."}

    status = order.get("order_status", order.get("status", "")).lower()
    cancellable = {"pending", "processing", "confirmed"}
    if status not in cancellable:
        return {
            "success": False,
            "error": f"Order {order_id} cannot be cancelled (status: {status}).",
        }

    return {
        "success": True,
        "order_id": order_id,
        "product": order.get("product", "item"),
        "message": f"Order {order_id} ({order.get('product', 'item')}) has been cancelled.",
    }


def get_order_tracking(tenant_id: str, order_id: str) -> dict:
    order = get_order(tenant_id, order_id)
    if not order:
        return {"error": f"Order {order_id} not found."}

    tracking = order.get("tracking_number")
    carrier = order.get("shipping_carrier")
    status = order.get("order_status", order.get("status", "unknown"))

    if not tracking:
        return {
            "order_id": order_id,
            "status": status,
            "tracking_number": None,
            "message": f"Order {order_id} has not been shipped yet (status: {status}).",
        }

    return {
        "order_id": order_id,
        "status": status,
        "tracking_number": tracking,
        "carrier": carrier,
        "delivery_date": order.get("delivery_date"),
        "message": f"Order {order_id} — {carrier} tracking: {tracking}. Status: {status}.",
    }


TOOL_REGISTRY: dict[str, callable] = {
    "get_order_details": get_order_details,
    "get_return_eligibility": get_return_eligibility,
    "initiate_return": initiate_return,
    "initiate_refund": initiate_refund,
    "get_shipping_label": get_shipping_label,
    "cancel_order": cancel_order,
    "get_order_tracking": get_order_tracking,
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
]
