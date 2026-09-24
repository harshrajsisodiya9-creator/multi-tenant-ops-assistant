"""enable pgvector

Revision ID: f81db64c515e
Revises: c63a73a345c2
Create Date: 2026-09-24 15:08:32.074230

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f81db64c515e"
down_revision: Union[str, Sequence[str], None] = "c63a73a345c2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    pass


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
    pass
