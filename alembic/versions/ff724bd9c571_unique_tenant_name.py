"""enforce unique tenants.name

Revision ID: ff724bd9c571
Revises: 63b935d2d7a7
Create Date: 2026-11-26

app/models.py declares Tenant.name unique, but the database never enforced it.
This revision refuses to run while duplicate names exist; resolve them first with
`python -m scripts.dedupe_tenant_names` (dry run) and `--apply`.

The constraint is named tenants_name_key, the name PostgreSQL gives the ORM's
unnamed unique constraint, so migrated and create_all() schemas are identical.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "ff724bd9c571"
down_revision: Union[str, Sequence[str], None] = "63b935d2d7a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    duplicates = op.get_bind().execute(
        sa.text(
            "SELECT name, count(*) FROM tenants GROUP BY name HAVING count(*) > 1 ORDER BY name"
        )
    ).all()
    if duplicates:
        listed = ", ".join(f"{name!r} x{n}" for name, n in duplicates)
        raise RuntimeError(
            f"Cannot enforce unique tenants.name; duplicates exist: {listed}. "
            "Run `python -m scripts.dedupe_tenant_names` to review the rename plan, "
            "then `--apply`, then re-run `alembic upgrade head`. No data was changed."
        )
    op.create_unique_constraint("tenants_name_key", "tenants", ["name"])


def downgrade() -> None:
    op.drop_constraint("tenants_name_key", "tenants", type_="unique")
