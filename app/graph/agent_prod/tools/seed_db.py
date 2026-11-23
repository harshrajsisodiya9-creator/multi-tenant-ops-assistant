"""Seed the customers, orders, order_items, and refund_history tables from mock_data."""
from __future__ import annotations

from datetime import date

from app.database import SessionLocal
from app.graph.mock_data import MOCK_ORDERS, MOCK_REFUND_HISTORY
from app.models import Customer, Order, OrderItem, RefundHistory, Tenant


def _build_customer_id(tenant_prefix: str, seq: int) -> str:
    return f"CUST-{tenant_prefix}-{seq:03d}"


_TENANT_PREFIX = {
    "megamart": "MM",
    "tiny-candles": "TC",
}


def seed_orders():
    db = SessionLocal()
    try:
        tenants = {t.name: t for t in db.query(Tenant).all()}

        for tenant_name, orders in MOCK_ORDERS.items():
            tenant = tenants.get(tenant_name)
            if not tenant:
                print(f"Skipping {tenant_name}: tenant not found in DB")
                continue

            prefix = _TENANT_PREFIX.get(tenant_name, tenant_name[:2].upper())
            emails_seen: dict[str, str] = {}
            customer_seq = 0

            for order_data in orders.values():
                email = order_data["customer_email"]

                if email not in emails_seen:
                    customer_seq += 1
                    cust_id = _build_customer_id(prefix, customer_seq)
                    existing_cust = db.query(Customer).filter_by(customer_id=cust_id).first()
                    if not existing_cust:
                        name = email.split("@")[0].capitalize()
                        customer = Customer(
                            customer_id=cust_id,
                            tenant_id=tenant.id,
                            email=email,
                            name=name,
                        )
                        db.add(customer)
                    emails_seen[email] = cust_id

                existing = db.query(Order).filter_by(order_id=order_data["order_id"]).first()
                if existing:
                    if not existing.customer_id:
                        existing.customer_id = emails_seen[email]
                    continue

                order = Order(
                    order_id=order_data["order_id"],
                    tenant_id=tenant.id,
                    customer_id=emails_seen[email],
                    customer_email=order_data["customer_email"],
                    category=order_data.get("category", ""),
                    product=order_data.get("product", ""),
                    amount=order_data.get("amount", 0),
                    total_amount=order_data.get("total_amount", 0),
                    status=order_data.get("status", "pending"),
                    order_date=date.fromisoformat(order_data["order_date"]) if order_data.get("order_date") else None,
                    delivery_date=date.fromisoformat(order_data["delivery_date"]) if order_data.get("delivery_date") else None,
                    tracking_number=order_data.get("tracking_number"),
                    shipping_carrier=order_data.get("shipping_carrier"),
                )
                db.add(order)

                for item_data in order_data.get("items", []):
                    item = OrderItem(
                        item_id=item_data["item_id"],
                        order_id=order_data["order_id"],
                        product=item_data.get("product", ""),
                        category=item_data.get("category", ""),
                        amount=item_data.get("amount", 0),
                        status=item_data.get("status", "pending"),
                    )
                    db.add(item)

        for tenant_name, emails in MOCK_REFUND_HISTORY.items():
            tenant = tenants.get(tenant_name)
            if not tenant:
                continue

            for email, refunds in emails.items():
                for ref in refunds:
                    existing = db.query(RefundHistory).filter_by(
                        tenant_id=tenant.id,
                        order_id=ref["order_id"],
                    ).first()
                    if existing:
                        continue

                    cust = db.query(Customer).filter_by(
                        tenant_id=tenant.id, email=email,
                    ).first()

                    refund = RefundHistory(
                        tenant_id=tenant.id,
                        customer_id=cust.customer_id if cust else None,
                        customer_email=email,
                        order_id=ref["order_id"],
                        amount=ref["amount"],
                        refund_date=date.fromisoformat(ref["date"]),
                    )
                    db.add(refund)

        db.commit()
        print("Seed complete.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_orders()
