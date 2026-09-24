"""create chunks table

Revision ID: b1d1d5d2d31a
Revises: f81db64c515e
Create Date: 2026-09-24 16:45:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b1d1d5d2d31a"
down_revision: Union[str, Sequence[str], None] = "f81db64c515e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("source_document_id", sa.String(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "source_document_id",
            "chunk_index",
            name="uq_chunks_tenant_document_chunk",
        ),
    )
    op.create_index(op.f("ix_chunks_tenant_id"), "chunks", ["tenant_id"], unique=False)
    op.create_index(
        op.f("ix_chunks_source_document_id"),
        "chunks",
        ["source_document_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_chunks_source_document_id"), table_name="chunks")
    op.drop_index(op.f("ix_chunks_tenant_id"), table_name="chunks")
    op.drop_table("chunks")
