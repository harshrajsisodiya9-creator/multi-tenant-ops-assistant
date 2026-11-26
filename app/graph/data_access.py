"""Factory for data access — routes to mock_data or real DB based on AGENT_MODE."""
from __future__ import annotations

from datetime import date, timedelta
from uuid import UUID

from app.config import setting


def get_tenant_info(tenant_id: str) -> dict | None:
    if setting.agent_mode == "prod":
        from app.database import SessionLocal
        from app.models import Tenant

        db = SessionLocal()
        try:
            tenant = db.query(Tenant).filter(Tenant.id == UUID(tenant_id)).first()
            if not tenant:
                return None
            return {"active": True, "name": tenant.name}
        finally:
            db.close()

    from app.graph.mock_data import get_tenant_info as _mock
    return _mock(tenant_id)


def get_customer_orders(tenant_id: str, customer_id: str) -> list[dict]:
    if setting.agent_mode == "prod":
        from sqlalchemy import select

        from app.database import SessionLocal
        from app.models import Order

        db = SessionLocal()
        try:
            stmt = select(Order).where(
                Order.tenant_id == UUID(tenant_id),
                Order.customer_id == customer_id,
            )
            orders = db.execute(stmt).scalars().all()
            return [
                {
                    "order_id": o.order_id,
                    "customer_id": o.customer_id,
                    "customer_email": o.customer_email,
                    "category": o.category,
                    "product": o.product,
                    "amount": o.total_amount,
                    "total_amount": o.total_amount,
                    "status": o.status,
                    "order_date": o.order_date.isoformat() if o.order_date else None,
                    "delivery_date": o.delivery_date.isoformat() if o.delivery_date else None,
                    "tracking_number": o.tracking_number,
                    "shipping_carrier": o.shipping_carrier,
                    "items": [
                        {
                            "item_id": item.item_id,
                            "product": item.product,
                            "category": item.category,
                            "amount": item.amount,
                            "status": item.status,
                        }
                        for item in o.items
                    ],
                }
                for o in orders
            ]
        finally:
            db.close()

    from app.graph.mock_data import get_orders_by_customer as _mock
    return _mock(tenant_id, customer_id)


def get_order_scoped(tenant_id: str, customer_id: str, order_id: str) -> dict | None:
    if setting.agent_mode == "prod":
        from sqlalchemy import select

        from app.database import SessionLocal
        from app.models import Order

        db = SessionLocal()
        try:
            stmt = select(Order).where(
                Order.tenant_id == UUID(tenant_id),
                Order.customer_id == customer_id,
                Order.order_id == order_id,
            )
            o = db.execute(stmt).scalar_one_or_none()
            if not o:
                return None
            return {
                "order_id": o.order_id,
                "customer_id": o.customer_id,
                "customer_email": o.customer_email,
                "category": o.category,
                "product": o.product,
                "amount": o.total_amount,
                "total_amount": o.total_amount,
                "status": o.status,
                "order_date": o.order_date.isoformat() if o.order_date else None,
                "delivery_date": o.delivery_date.isoformat() if o.delivery_date else None,
                "tracking_number": o.tracking_number,
                "shipping_carrier": o.shipping_carrier,
                "items": [
                    {
                        "item_id": item.item_id,
                        "product": item.product,
                        "category": item.category,
                        "amount": item.amount,
                        "status": item.status,
                    }
                    for item in o.items
                ],
            }
        finally:
            db.close()

    from app.graph.mock_data import get_order_scoped as _mock
    return _mock(tenant_id, customer_id, order_id)


def get_refund_count_30d(tenant_id: str, customer_id: str) -> int:
    if setting.agent_mode == "prod":
        from sqlalchemy import func, select

        from app.database import SessionLocal
        from app.models import RefundHistory

        cutoff = date.today() - timedelta(days=30)
        db = SessionLocal()
        try:
            stmt = select(func.count()).select_from(RefundHistory).where(
                RefundHistory.tenant_id == UUID(tenant_id),
                RefundHistory.customer_id == customer_id,
                RefundHistory.refund_date >= cutoff,
            )
            return db.execute(stmt).scalar() or 0
        finally:
            db.close()

    from app.graph.mock_data import get_refund_count_30d_by_customer as _mock
    return _mock(tenant_id, customer_id)
