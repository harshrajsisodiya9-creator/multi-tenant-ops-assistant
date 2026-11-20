"""add content_tsv column with GIN index and trigger

Revision ID: a7f2e3c41b90
Revises: b1d1d5d2d31a
Create Date: 2026-10-01 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

revision: str = "a7f2e3c41b90"
down_revision: Union[str, Sequence[str], None] = "b1d1d5d2d31a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE chunks "
        "ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    op.execute(
        "CREATE INDEX ix_chunks_content_tsv ON chunks USING GIN (content_tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_content_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv")
