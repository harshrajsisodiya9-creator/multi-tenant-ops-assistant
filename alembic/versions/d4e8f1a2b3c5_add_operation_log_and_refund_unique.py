"""Add operation_log table and refund_history unique constraint.

Revision ID: d4e8f1a2b3c5
Revises: 547731e4dfa3
Create Date: 2026-11-26
"""
from alembic import op
import sqlalchemy as sa

revision = "d4e8f1a2b3c5"
# Re-parented from a7f2e3c41b90: refund_history is created in 547731e4dfa3.
down_revision = "547731e4dfa3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "operation_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("customer_id", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=64), nullable=False),
        sa.Column("operation", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("request_args", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("result", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "customer_id",
            "idempotency_key",
            name="uq_operation_log_idempotency",
        ),
    )
    op.create_index(
        "ix_operation_log_tenant_customer",
        "operation_log",
        ["tenant_id", "customer_id"],
    )
    op.create_index(
        op.f("ix_operation_log_tenant_id"),
        "operation_log",
        ["tenant_id"],
    )

    op.create_unique_constraint(
        "uq_refund_history_tenant_order",
        "refund_history",
        ["tenant_id", "order_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_refund_history_tenant_order", "refund_history", type_="unique",
    )
    op.drop_index("ix_operation_log_tenant_customer", table_name="operation_log")
    op.drop_index(op.f("ix_operation_log_tenant_id"), table_name="operation_log")
    op.drop_table("operation_log")
