"""enable pg_trgm

Revision ID: 63b935d2d7a7
Revises: d4e8f1a2b3c5
Create Date: 2026-11-26

app/hybrid_search.py falls back to similarity() from pg_trgm when full-text
search finds nothing. The extension had only ever been installed by hand.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "63b935d2d7a7"
down_revision: Union[str, Sequence[str], None] = "d4e8f1a2b3c5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")


def downgrade() -> None:
    # No-op on purpose: existing databases installed pg_trgm before this revision,
    # no schema object depends on it, and the trigram search fallback at every
    # revision still needs it.
    pass
