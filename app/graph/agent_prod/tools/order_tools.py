from __future__ import annotations

from datetime import date
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.graph.policy_loader import resolve_policy
from app.models import Order, OrderItem, RefundHistory


def _get_db() -> Session:
    return SessionLocal()


def _find_order(db: Session, tenant_id: str, order_id: str) -> Order | None:
    stmt = select(Order).where(
        Order.tenant_id == UUID(tenant_id),
        Order.order_id == order_id,
    )
    return db.execute(stmt).scalar_one_or_none()


def _find_item(db: Session, order_id: str, item_id: str) -> OrderItem | None:
    stmt = select(OrderItem).where(
        OrderItem.order_id == order_id,
        OrderItem.item_id == item_id,
    )
    return db.execute(stmt).scalar_one_or_none()


def get_order_details(tenant_id: str, order_id: str) -> dict:
    db = _get_db()
    try:
        order = _find_order(db, tenant_id, order_id)
        if not order:
            return {"error": f"Order {order_id} not found."}
        return {
            "order_id": order.order_id,
            "customer_email": order.customer_email,
            "order_status": order.status,
            "total_amount": order.total_amount,
            "order_date": order.order_date.isoformat() if order.order_date else None,
            "delivery_date": order.delivery_date.isoformat() if order.delivery_date else None,
            "items": [
                {
                    "item_id": item.item_id,
                    "product": item.product,
                    "category": item.category,
                    "amount": item.amount,
                    "status": item.status,
                }
                for item in order.items
            ],
        }
    finally:
        db.close()


def get_return_eligibility(
    tenant_id: str, order_id: str, item_id: str | None = None
) -> dict:
    db = _get_db()
    try:
        order = _find_order(db, tenant_id, order_id)
        if not order:
            return {"eligible": False, "reason": f"Order {order_id} not found."}

        if item_id:
            item = _find_item(db, order_id, item_id)
            if not item:
                return {"eligible": False, "reason": f"Item {item_id} not found in order {order_id}."}
            category = item.category
            amount = item.amount
        else:
            category = order.category
            amount = order.amount

        policy = resolve_policy(tenant_id, category)

        if not policy.return_policy.allowed:
            return {
                "eligible": False,
                "reason": f"Returns not allowed for {category}.",
                "category": category,
            }

        if order.delivery_date:
            days_since = (date.today() - order.delivery_date).days
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
    finally:
        db.close()


def initiate_return(
    tenant_id: str, order_id: str, item_id: str | None = None, reason: str = ""
) -> dict:
    db = _get_db()
    try:
        order = _find_order(db, tenant_id, order_id)
        if not order:
            return {"success": False, "error": f"Order {order_id} not found."}

        if item_id:
            item = _find_item(db, order_id, item_id)
            if not item:
                return {"success": False, "error": f"Item {item_id} not found."}
            product = item.product
        else:
            product = order.product

        return {
            "success": True,
            "return_id": f"RET-{order_id}-001",
            "product": product,
            "message": f"Return initiated for {product} from order {order_id}.",
        }
    finally:
        db.close()


def initiate_refund(
    tenant_id: str,
    order_id: str,
    amount: int | float,
    refund_type: str = "original_method",
    item_id: str | None = None,
) -> dict:
    db = _get_db()
    try:
        order = _find_order(db, tenant_id, order_id)
        if not order:
            return {"success": False, "error": f"Order {order_id} not found."}

        if item_id:
            item = _find_item(db, order_id, item_id)
            if not item:
                return {"success": False, "error": f"Item {item_id} not found."}
            product = item.product
        else:
            product = order.product

        refund = RefundHistory(
            tenant_id=UUID(tenant_id),
            customer_email=order.customer_email,
            order_id=order_id,
            amount=int(amount),
            refund_date=date.today(),
        )
        db.add(refund)
        db.commit()

        return {
            "success": True,
            "refund_id": f"REF-{order_id}-001",
            "amount": amount,
            "refund_type": refund_type,
            "product": product,
            "message": f"Refund of ₹{amount:,.0f} initiated for {product} ({order_id}) via {refund_type}.",
        }
    finally:
        db.close()


def get_shipping_label(tenant_id: str, order_id: str) -> dict:
    db = _get_db()
    try:
        order = _find_order(db, tenant_id, order_id)
        if not order:
            return {"success": False, "error": f"Order {order_id} not found."}
        return {
            "success": True,
            "label_url": f"https://shipping.example.com/label/{order_id}",
            "carrier": order.shipping_carrier or "Standard",
            "message": f"Shipping label generated for return of order {order_id}.",
        }
    finally:
        db.close()


def cancel_order(tenant_id: str, order_id: str) -> dict:
    db = _get_db()
    try:
        order = _find_order(db, tenant_id, order_id)
        if not order:
            return {"success": False, "error": f"Order {order_id} not found."}

        cancellable = {"pending", "processing", "confirmed"}
        if order.status.lower() not in cancellable:
            return {
                "success": False,
                "error": f"Order {order_id} cannot be cancelled (status: {order.status}).",
            }

        order.status = "cancelled"
        db.commit()

        return {
            "success": True,
            "order_id": order_id,
            "product": order.product,
            "message": f"Order {order_id} ({order.product}) has been cancelled.",
        }
    finally:
        db.close()


def get_order_tracking(tenant_id: str, order_id: str) -> dict:
    db = _get_db()
    try:
        order = _find_order(db, tenant_id, order_id)
        if not order:
            return {"error": f"Order {order_id} not found."}

        if not order.tracking_number:
            return {
                "order_id": order_id,
                "status": order.status,
                "tracking_number": None,
                "message": f"Order {order_id} has not been shipped yet (status: {order.status}).",
            }

        return {
            "order_id": order_id,
            "status": order.status,
            "tracking_number": order.tracking_number,
            "carrier": order.shipping_carrier,
            "delivery_date": order.delivery_date.isoformat() if order.delivery_date else None,
            "message": f"Order {order_id} — {order.shipping_carrier} tracking: {order.tracking_number}. Status: {order.status}.",
        }
    finally:
        db.close()


def get_orders_by_email(tenant_id: str, customer_email: str) -> dict:
    db = _get_db()
    try:
        stmt = select(Order).where(
            Order.tenant_id == UUID(tenant_id),
            Order.customer_email == customer_email,
        )
        orders = db.execute(stmt).scalars().all()
        if not orders:
            return {"orders": [], "message": "No orders found for this email."}
        return {
            "orders": [
                {
                    "order_id": o.order_id,
                    "product": o.product,
                    "amount": o.total_amount,
                    "status": o.status,
                    "order_date": o.order_date.isoformat() if o.order_date else None,
                }
                for o in orders
            ]
        }
    finally:
        db.close()
