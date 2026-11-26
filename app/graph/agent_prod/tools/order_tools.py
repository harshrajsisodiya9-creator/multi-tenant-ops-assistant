from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.graph.agent_prod.tool_context import ToolContext
from app.graph.policy_loader import resolve_policy
from app.models import OperationLog, Order, OrderItem, RefundHistory

logger = logging.getLogger(__name__)


def _get_db() -> Session:
    return SessionLocal()


# ---------------------------------------------------------------------------
# Idempotency helpers
# ---------------------------------------------------------------------------

def _make_idempotency_key(operation: str, **kwargs) -> str:
    parts = [operation]
    for k in sorted(kwargs):
        parts.append(f"{k}={kwargs[k]}")
    return hashlib.sha256(":".join(parts).encode()).hexdigest()[:32]


def _claim_idempotency(
    db: Session,
    tenant_id: str,
    customer_id: str,
    idempotency_key: str,
    operation: str,
    request_args: dict,
) -> OperationLog | None:
    """Try to claim an idempotency slot.

    Returns the existing OperationLog if the key is already taken (completed
    or pending), None if the slot was successfully claimed for this caller.
    """
    existing = db.execute(
        select(OperationLog).where(
            OperationLog.tenant_id == UUID(tenant_id),
            OperationLog.customer_id == customer_id,
            OperationLog.idempotency_key == idempotency_key,
        )
    ).scalar_one_or_none()
    if existing:
        return existing

    log = OperationLog(
        tenant_id=UUID(tenant_id),
        customer_id=customer_id,
        idempotency_key=idempotency_key,
        operation=operation,
        status="pending",
        request_args=json.dumps(request_args, default=str),
    )
    db.add(log)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        return db.execute(
            select(OperationLog).where(
                OperationLog.tenant_id == UUID(tenant_id),
                OperationLog.customer_id == customer_id,
                OperationLog.idempotency_key == idempotency_key,
            )
        ).scalar_one_or_none()
    return None


def _complete_idempotency(
    db: Session, log: OperationLog, status: str, result: dict,
) -> None:
    log.status = status
    log.result = json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Customer-level advisory lock (PostgreSQL only)
# ---------------------------------------------------------------------------

def _customer_advisory_lock(db: Session, tenant_id: str, customer_id: str) -> None:
    if db.bind.dialect.name == "sqlite":
        return
    h = hashlib.sha256(f"{tenant_id}:{customer_id}".encode()).digest()
    lock_id = int.from_bytes(h[:8], byteorder="big", signed=True)
    db.execute(text("SELECT pg_advisory_xact_lock(:lid)"), {"lid": lock_id})


def _find_order(
    db: Session, ctx: ToolContext, order_id: str, *, for_update: bool = False,
) -> Order | None:
    stmt = select(Order).where(
        Order.tenant_id == UUID(ctx.tenant_id),
        Order.customer_id == ctx.customer_id,
        Order.order_id == order_id,
    )
    if for_update and db.bind.dialect.name != "sqlite":
        stmt = stmt.with_for_update()
    return db.execute(stmt).scalar_one_or_none()


def _find_item(db: Session, ctx: ToolContext, order_id: str, item_id: str) -> OrderItem | None:
    stmt = (
        select(OrderItem)
        .join(Order, OrderItem.order_id == Order.order_id)
        .where(
            Order.tenant_id == UUID(ctx.tenant_id),
            Order.customer_id == ctx.customer_id,
            OrderItem.order_id == order_id,
            OrderItem.item_id == item_id,
        )
    )
    return db.execute(stmt).scalar_one_or_none()


def get_order_details(ctx: ToolContext, order_id: str) -> dict:
    db = _get_db()
    try:
        order = _find_order(db, ctx, order_id)
        if not order:
            return {"error": f"Order {order_id} not found."}
        return {
            "order_id": order.order_id,
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
    ctx: ToolContext, order_id: str, item_id: str | None = None
) -> dict:
    db = _get_db()
    try:
        order = _find_order(db, ctx, order_id)
        if not order:
            return {"eligible": False, "reason": f"Order {order_id} not found."}

        if item_id:
            item = _find_item(db, ctx, order_id, item_id)
            if not item:
                return {"eligible": False, "reason": f"Item {item_id} not found in order {order_id}."}
            category = item.category
            amount = item.amount
        else:
            category = order.category
            amount = order.amount

        policy = resolve_policy(ctx.tenant_id, category)

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
    ctx: ToolContext, order_id: str, item_id: str | None = None, reason: str = ""
) -> dict:
    db = _get_db()
    try:
        idem_key = _make_idempotency_key(
            "initiate_return", order_id=order_id,
            item_id=item_id or "", tenant_id=ctx.tenant_id,
        )
        request_args = {"order_id": order_id, "item_id": item_id, "reason": reason}

        _customer_advisory_lock(db, ctx.tenant_id, ctx.customer_id)

        existing_op = _claim_idempotency(
            db, ctx.tenant_id, ctx.customer_id, idem_key,
            "initiate_return", request_args,
        )
        if existing_op:
            if existing_op.status == "completed":
                return json.loads(existing_op.result)
            return {"success": False, "error": f"Return already processed for order {order_id}."}

        op_log = db.execute(
            select(OperationLog).where(
                OperationLog.tenant_id == UUID(ctx.tenant_id),
                OperationLog.customer_id == ctx.customer_id,
                OperationLog.idempotency_key == idem_key,
            )
        ).scalar_one()

        order = _find_order(db, ctx, order_id, for_update=True)
        if not order:
            result = {"success": False, "error": f"Order {order_id} not found."}
            _complete_idempotency(db, op_log, "failed", result)
            db.commit()
            return result

        if item_id:
            item = _find_item(db, ctx, order_id, item_id)
            if not item:
                result = {"success": False, "error": f"Item {item_id} not found."}
                _complete_idempotency(db, op_log, "failed", result)
                db.commit()
                return result
            product = item.product
            item_category = item.category
        else:
            product = order.product
            item_category = None

        policy_block = _check_return_policy(ctx.tenant_id, order, item_category)
        if policy_block:
            _complete_idempotency(db, op_log, "failed", policy_block)
            db.commit()
            return policy_block

        result = {
            "success": True,
            "return_id": f"RET-{order_id}-001",
            "product": product,
            "message": f"Return initiated for {product} from order {order_id}.",
        }
        _complete_idempotency(db, op_log, "completed", result)
        db.commit()
        return result
    finally:
        db.close()


def _check_return_policy(
    tenant_id: str, order: Order, item_category: str | None = None,
) -> dict | None:
    """Re-run return policy inside a write tool's transaction.
    Returns an error dict if blocked, None if allowed."""
    category = item_category or order.category
    policy = resolve_policy(tenant_id, category)

    if not policy.return_policy.allowed:
        return {"success": False, "error": f"Returns not allowed for {category}."}

    if order.delivery_date:
        days_since = (date.today() - order.delivery_date).days
        if days_since > policy.return_policy.window_days:
            return {"success": False, "error": f"Return window expired ({days_since} days since delivery, limit {policy.return_policy.window_days})."}

    return None


def initiate_refund(
    ctx: ToolContext,
    order_id: str,
    amount: int | float,
    refund_type: str = "original_method",
    item_id: str | None = None,
) -> dict:
    db = _get_db()
    try:
        idem_key = _make_idempotency_key(
            "initiate_refund", order_id=order_id, tenant_id=ctx.tenant_id,
        )
        request_args = {
            "order_id": order_id, "amount": amount,
            "refund_type": refund_type, "item_id": item_id,
        }

        _customer_advisory_lock(db, ctx.tenant_id, ctx.customer_id)

        existing_op = _claim_idempotency(
            db, ctx.tenant_id, ctx.customer_id, idem_key,
            "initiate_refund", request_args,
        )
        if existing_op:
            if existing_op.status == "completed":
                return json.loads(existing_op.result)
            return {"success": False, "error": f"Refund already exists for order {order_id}."}

        op_log = db.execute(
            select(OperationLog).where(
                OperationLog.tenant_id == UUID(ctx.tenant_id),
                OperationLog.customer_id == ctx.customer_id,
                OperationLog.idempotency_key == idem_key,
            )
        ).scalar_one()

        order = _find_order(db, ctx, order_id, for_update=True)
        if not order:
            result = {"success": False, "error": f"Order {order_id} not found."}
            _complete_idempotency(db, op_log, "failed", result)
            db.commit()
            return result

        existing_refund = db.execute(
            select(RefundHistory).where(
                RefundHistory.tenant_id == UUID(ctx.tenant_id),
                RefundHistory.order_id == order_id,
            )
        ).scalar_one_or_none()
        if existing_refund:
            result = {"success": False, "error": f"Refund already exists for order {order_id}."}
            _complete_idempotency(db, op_log, "failed", result)
            db.commit()
            return result

        if item_id:
            item = _find_item(db, ctx, order_id, item_id)
            if not item:
                result = {"success": False, "error": f"Item {item_id} not found."}
                _complete_idempotency(db, op_log, "failed", result)
                db.commit()
                return result
            product = item.product
            item_category = item.category
        else:
            product = order.product
            item_category = None

        policy_block = _check_return_policy(ctx.tenant_id, order, item_category)
        if policy_block:
            _complete_idempotency(db, op_log, "failed", policy_block)
            db.commit()
            return policy_block

        refund = RefundHistory(
            tenant_id=UUID(ctx.tenant_id),
            customer_id=ctx.customer_id,
            customer_email=order.customer_email,
            order_id=order_id,
            amount=int(amount),
            refund_date=date.today(),
        )
        db.add(refund)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            return {"success": False, "error": f"Refund already exists for order {order_id}."}

        result = {
            "success": True,
            "refund_id": f"REF-{order_id}-001",
            "amount": amount,
            "refund_type": refund_type,
            "product": product,
            "message": f"Refund of ₹{amount:,.0f} initiated for {product} ({order_id}) via {refund_type}.",
        }
        _complete_idempotency(db, op_log, "completed", result)
        db.commit()
        return result
    finally:
        db.close()


def get_shipping_label(ctx: ToolContext, order_id: str) -> dict:
    db = _get_db()
    try:
        order = _find_order(db, ctx, order_id, for_update=True)
        if not order:
            return {"success": False, "error": f"Order {order_id} not found."}

        policy_block = _check_return_policy(ctx.tenant_id, order)
        if policy_block:
            return policy_block

        return {
            "success": True,
            "label_url": f"https://shipping.example.com/label/{order_id}",
            "carrier": order.shipping_carrier or "Standard",
            "message": f"Shipping label generated for return of order {order_id}.",
        }
    finally:
        db.close()


def cancel_order(ctx: ToolContext, order_id: str) -> dict:
    db = _get_db()
    try:
        idem_key = _make_idempotency_key(
            "cancel_order", order_id=order_id, tenant_id=ctx.tenant_id,
        )
        request_args = {"order_id": order_id}

        _customer_advisory_lock(db, ctx.tenant_id, ctx.customer_id)

        existing_op = _claim_idempotency(
            db, ctx.tenant_id, ctx.customer_id, idem_key,
            "cancel_order", request_args,
        )
        if existing_op:
            if existing_op.status == "completed":
                return json.loads(existing_op.result)
            return {"success": False, "error": f"Cancellation already processed for order {order_id}."}

        op_log = db.execute(
            select(OperationLog).where(
                OperationLog.tenant_id == UUID(ctx.tenant_id),
                OperationLog.customer_id == ctx.customer_id,
                OperationLog.idempotency_key == idem_key,
            )
        ).scalar_one()

        order = _find_order(db, ctx, order_id, for_update=True)
        if not order:
            result = {"success": False, "error": f"Order {order_id} not found."}
            _complete_idempotency(db, op_log, "failed", result)
            db.commit()
            return result

        cancellable = {"pending", "processing", "confirmed"}
        if order.status.lower() not in cancellable:
            result = {
                "success": False,
                "error": f"Order {order_id} cannot be cancelled (status: {order.status}).",
            }
            _complete_idempotency(db, op_log, "failed", result)
            db.commit()
            return result

        order.status = "cancelled"

        result = {
            "success": True,
            "order_id": order_id,
            "product": order.product,
            "message": f"Order {order_id} ({order.product}) has been cancelled.",
        }
        _complete_idempotency(db, op_log, "completed", result)
        db.commit()
        return result
    finally:
        db.close()


def get_order_tracking(ctx: ToolContext, order_id: str) -> dict:
    db = _get_db()
    try:
        order = _find_order(db, ctx, order_id)
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


def list_my_orders(ctx: ToolContext) -> dict:
    db = _get_db()
    try:
        stmt = select(Order).where(
            Order.tenant_id == UUID(ctx.tenant_id),
            Order.customer_id == ctx.customer_id,
        )
        orders = db.execute(stmt).scalars().all()
        if not orders:
            return {"orders": [], "message": "No orders found."}
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
