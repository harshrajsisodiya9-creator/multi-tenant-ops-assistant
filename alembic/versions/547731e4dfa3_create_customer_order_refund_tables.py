"""create customers, orders, order_items and refund_history tables

Revision ID: 547731e4dfa3
Revises: a7f2e3c41b90
Create Date: 2026-11-26

These tables were declared in app/models.py but never had a migration, so
`alembic upgrade head` could not build a working schema from an empty database.

uq_refund_history_tenant_order is intentionally NOT created here: the next
revision (d4e8f1a2b3c5) adds it.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "547731e4dfa3"
down_revision: Union[str, Sequence[str], None] = "a7f2e3c41b90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLES = ("customers", "orders", "order_items", "refund_history")


def _fail_if_tables_already_exist() -> None:
    # A database whose tables were created outside Alembic (Base.metadata.create_all)
    # must be checked against this schema and stamped, not migrated over.
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    clashes = [t for t in _TABLES if t in existing]
    if clashes:
        raise RuntimeError(
            f"Tables {clashes} already exist but were not created by Alembic. "
            "Compare them with this migration, then run `alembic stamp` with the "
            "revision their schema matches (547731e4dfa3, or d4e8f1a2b3c5 if "
            "operation_log and uq_refund_history_tenant_order also exist) "
            "before running `alembic upgrade head`."
        )


def upgrade() -> None:
    _fail_if_tables_already_exist()

    op.create_table(
        "customers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("customer_id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id", "customer_id", name="uq_customer_tenant_customer"
        ),
    )
    # Unique index (not just a constraint) because orders.customer_id references it.
    op.create_index(
        op.f("ix_customers_customer_id"), "customers", ["customer_id"], unique=True
    )
    op.create_index(op.f("ix_customers_email"), "customers", ["email"], unique=False)
    op.create_index(
        op.f("ix_customers_tenant_id"), "customers", ["tenant_id"], unique=False
    )

    op.create_table(
        "orders",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("order_id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("customer_id", sa.String(), nullable=False),
        sa.Column("customer_email", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("product", sa.String(), nullable=False),
        sa.Column("amount", sa.Integer(), nullable=False),
        sa.Column("total_amount", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("order_date", sa.Date(), nullable=True),
        sa.Column("delivery_date", sa.Date(), nullable=True),
        sa.Column("tracking_number", sa.String(), nullable=True),
        sa.Column("shipping_carrier", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.customer_id"]),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_orders_customer_email"), "orders", ["customer_email"], unique=False
    )
    op.create_index(
        op.f("ix_orders_customer_id"), "orders", ["customer_id"], unique=False
    )
    op.create_index(op.f("ix_orders_order_id"), "orders", ["order_id"], unique=True)
    op.create_index(
        "ix_orders_tenant_customer", "orders", ["tenant_id", "customer_id"], unique=False
    )
    op.create_index(op.f("ix_orders_tenant_id"), "orders", ["tenant_id"], unique=False)

    op.create_table(
        "order_items",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("item_id", sa.String(), nullable=False),
        sa.Column("order_id", sa.String(), nullable=False),
        sa.Column("product", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("amount", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["order_id"], ["orders.order_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_order_items_item_id"), "order_items", ["item_id"], unique=True
    )
    op.create_index(
        op.f("ix_order_items_order_id"), "order_items", ["order_id"], unique=False
    )

    op.create_table(
        "refund_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("customer_id", sa.String(), nullable=False),
        sa.Column("customer_email", sa.String(), nullable=False),
        sa.Column("order_id", sa.String(), nullable=False),
        sa.Column("amount", sa.Integer(), nullable=False),
        sa.Column("refund_date", sa.Date(), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_refund_history_customer_email"),
        "refund_history",
        ["customer_email"],
        unique=False,
    )
    op.create_index(
        op.f("ix_refund_history_customer_id"),
        "refund_history",
        ["customer_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_refund_history_tenant_id"),
        "refund_history",
        ["tenant_id"],
        unique=False,
    )


def downgrade() -> None:
    # Reverse FK order; dropping a table drops its indexes and constraints.
    op.drop_table("refund_history")
    op.drop_table("order_items")
    op.drop_table("orders")
    op.drop_table("customers")
