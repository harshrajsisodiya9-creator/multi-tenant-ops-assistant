"""tenants.name uniqueness: the ORM declaration and the dedupe plan that must run
before migration ff724bd9c571 can enforce it."""
from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import Customer, Tenant
from scripts.dedupe_tenant_names import (
    Rename,
    TenantRow,
    apply_renames,
    load_tenants,
    plan_renames,
)

A = UUID("f67e4814-c4bf-421b-94d8-0d5894703816")
B = UUID("c2e0689c-1751-4d5c-8c63-9814dc7feefd")


def test_orm_declares_tenant_name_unique():
    assert Tenant.__table__.c.name.unique is True


def test_plan_keeps_name_on_tenant_with_most_dependents():
    rows = [TenantRow(B, "Test Tenant", 0), TenantRow(A, "Test Tenant", 93)]
    assert plan_renames(rows) == [Rename(B, "Test Tenant", "Test Tenant [c2e0689c]")]


def test_plan_breaks_ties_by_id_and_ignores_unique_names():
    rows = [
        TenantRow(A, "dup", 0),
        TenantRow(B, "dup", 0),
        TenantRow(uuid4(), "solo", 5),
    ]
    # str(B) < str(A), so B keeps the name.
    assert plan_renames(rows) == [Rename(A, "dup", f"dup [{A.hex[:8]}]")]


def test_plan_avoids_colliding_with_an_existing_name():
    rows = [
        TenantRow(A, "dup", 1),
        TenantRow(B, "dup", 0),
        TenantRow(uuid4(), f"dup [{B.hex[:8]}]", 0),
    ]
    assert plan_renames(rows) == [Rename(B, "dup", f"dup [{B.hex}]")]


def test_apply_renames_only_names_and_keeps_ids_and_foreign_keys():
    engine = create_engine("sqlite://", poolclass=StaticPool)
    # Pre-migration tenants table: no unique constraint on name yet.
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE tenants (id CHAR(32) PRIMARY KEY, name VARCHAR NOT NULL, "
            "api_key_hash VARCHAR NOT NULL)"
        )
    Base.metadata.create_all(engine, tables=[Customer.__table__])

    with Session(engine) as db:
        db.add_all([
            Tenant(id=A, name="Test Tenant", api_key_hash="ka"),
            Tenant(id=B, name="Test Tenant", api_key_hash="kb"),
        ])
        db.flush()
        db.add(Customer(customer_id="C-1", tenant_id=A, email="a@x.z", name="a"))
        db.commit()

    with engine.begin() as conn:
        rows = load_tenants(conn)
        assert {(r.id, r.dependents) for r in rows} == {(A, 1), (B, 0)}
        apply_renames(conn, plan_renames(rows))

    with Session(engine) as db:
        tenants = {t.id: (t.name, t.api_key_hash) for t in db.scalars(select(Tenant))}
        assert tenants == {A: ("Test Tenant", "ka"), B: ("Test Tenant [c2e0689c]", "kb")}
        assert db.scalar(select(Customer.tenant_id)) == A
    engine.dispose()
