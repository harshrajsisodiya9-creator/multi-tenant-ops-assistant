"""Regression tests for seed_orders(): every seeded refund must reference a seeded
customer and an order owned by that customer in the same tenant."""
from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.graph.agent_prod.tools.seed_db as seed_db
from app.database import Base
from app.graph.mock_data import MOCK_ORDERS
from app.models import Customer, Order, OrderItem, RefundHistory, Tenant


def _first_order_of(tenant: str, email: str) -> str:
    return next(o["order_id"] for o in MOCK_ORDERS[tenant].values() if o["customer_email"] == email)


@pytest.fixture()
def seed_session(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    tables = [
        Tenant.__table__, Customer.__table__, Order.__table__,
        OrderItem.__table__, RefundHistory.__table__,
    ]
    Base.metadata.create_all(engine, tables=tables)
    # Same flags as app.database.SessionLocal: autoflush=False is what hid pending
    # customers from the refund loop's query.
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    db = factory()
    for name in ("megamart", "tiny-candles"):
        db.add(Tenant(id=uuid4(), name=name, api_key_hash="x"))
    db.commit()
    db.close()

    monkeypatch.setattr(seed_db, "SessionLocal", factory)
    yield factory
    engine.dispose()


def _assert_refunds_are_consistent(db) -> None:
    for refund in db.scalars(select(RefundHistory)):
        assert refund.customer_id is not None
        customer = db.scalar(select(Customer).where(Customer.customer_id == refund.customer_id))
        assert customer is not None and customer.tenant_id == refund.tenant_id
        assert customer.email == refund.customer_email
        order = db.scalar(select(Order).where(Order.order_id == refund.order_id))
        assert order is not None, f"refund {refund.order_id} has no order"
        assert order.tenant_id == refund.tenant_id
        assert order.customer_id == refund.customer_id


def test_seed_with_shipped_mock_data_never_writes_null_customer(seed_session):
    seed_db.seed_orders()

    db = seed_session()
    try:
        assert db.scalar(select(Customer).limit(1)) is not None
        _assert_refunds_are_consistent(db)
    finally:
        db.close()


def test_refunds_are_seeded_only_for_the_customers_own_orders(seed_session, monkeypatch, capsys):
    alice_order = _first_order_of("megamart", "alice@example.com")
    bob_order = _first_order_of("megamart", "bob@example.com")
    monkeypatch.setattr(seed_db, "MOCK_REFUND_HISTORY", {
        "megamart": {
            "alice@example.com": [
                {"order_id": alice_order, "amount": 100, "date": "2026-11-01"},   # valid
                {"order_id": "ORD-MM-9001", "amount": 3000, "date": "2026-09-20"},  # order not seeded
                {"order_id": bob_order, "amount": 50, "date": "2026-11-02"},     # bob's order
            ],
            "nobody@example.com": [
                {"order_id": alice_order, "amount": 10, "date": "2026-11-03"},   # no such customer
            ],
        },
        "tiny-candles": {},
    })

    seed_db.seed_orders()

    db = seed_session()
    try:
        refunds = db.scalars(select(RefundHistory)).all()
        assert [(r.order_id, r.customer_id) for r in refunds] == [(alice_order, "CUST-MM-001")]
        _assert_refunds_are_consistent(db)
    finally:
        db.close()

    out = capsys.readouterr().out
    assert "ORD-MM-9001" in out and "not seeded" in out
    assert bob_order in out and "belongs to another customer" in out
    assert "nobody@example.com" in out and "no seeded customer" in out


def test_seed_is_idempotent(seed_session):
    seed_db.seed_orders()
    seed_db.seed_orders()

    db = seed_session()
    try:
        counts = {m.__tablename__: len(db.scalars(select(m)).all()) for m in (Customer, Order, OrderItem)}
        assert counts == {
            "customers": sum(len({o["customer_email"] for o in t.values()}) for t in MOCK_ORDERS.values()),
            "orders": sum(len(t) for t in MOCK_ORDERS.values()),
            "order_items": sum(len(o.get("items", [])) for t in MOCK_ORDERS.values() for o in t.values()),
        }
        _assert_refunds_are_consistent(db)
    finally:
        db.close()
