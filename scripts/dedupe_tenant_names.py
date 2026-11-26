"""Resolve duplicate tenants.name values so the unique constraint can be added.

Nothing is deleted or re-pointed. For each duplicated name, the tenant with the most
dependent rows (summed over every table with a foreign key to tenants.id) keeps the
name, and every other tenant is renamed to "<name> [<first 8 hex of its id>]".
Tenant ids, API keys and all foreign keys are unchanged, so tenant isolation holds.

Usage (from the repo root; uses DATABASE_URL from app.config):
    python -m scripts.dedupe_tenant_names            # dry run: print the plan
    python -m scripts.dedupe_tenant_names --apply    # perform the renames in one transaction
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import Connection, Integer, String, Uuid, bindparam, inspect, text


@dataclass(frozen=True)
class TenantRow:
    id: UUID
    name: str
    dependents: int


@dataclass(frozen=True)
class Rename:
    id: UUID
    old_name: str
    new_name: str


def plan_renames(rows: list[TenantRow]) -> list[Rename]:
    taken = {r.name for r in rows}
    by_name: dict[str, list[TenantRow]] = defaultdict(list)
    for row in rows:
        by_name[row.name].append(row)

    renames: list[Rename] = []
    for name, group in sorted(by_name.items()):
        if len(group) < 2:
            continue
        # Keeper: most dependent rows; ties broken by id so the plan is deterministic.
        group.sort(key=lambda r: (-r.dependents, str(r.id)))
        for row in group[1:]:
            new_name = f"{name} [{row.id.hex[:8]}]"
            if new_name in taken:
                new_name = f"{name} [{row.id.hex}]"
            taken.add(new_name)
            renames.append(Rename(row.id, name, new_name))
    return renames


def _tenant_fk_columns(conn: Connection) -> list[tuple[str, str]]:
    insp = inspect(conn)
    refs: list[tuple[str, str]] = []
    for table in insp.get_table_names():
        for fk in insp.get_foreign_keys(table):
            if fk["referred_table"] == "tenants" and fk["referred_columns"] == ["id"]:
                refs.append((table, fk["constrained_columns"][0]))
    return refs


def load_tenants(conn: Connection) -> list[TenantRow]:
    counts: dict[UUID, int] = defaultdict(int)
    # Typed columns/params so ids come back as uuid.UUID on every dialect.
    for table, column in _tenant_fk_columns(conn):
        stmt = text(
            f'SELECT "{column}" AS tenant_id, count(*) AS n FROM "{table}" GROUP BY "{column}"'
        ).columns(tenant_id=Uuid, n=Integer)
        for tenant_id, n in conn.execute(stmt):
            counts[tenant_id] += n
    stmt = text("SELECT id, name FROM tenants").columns(id=Uuid, name=String)
    return [
        TenantRow(id=tid, name=name, dependents=counts[tid])
        for tid, name in conn.execute(stmt)
    ]


def apply_renames(conn: Connection, renames: list[Rename]) -> None:
    stmt = text("UPDATE tenants SET name = :new WHERE id = :id AND name = :old").bindparams(
        bindparam("id", type_=Uuid)
    )
    for r in renames:
        conn.execute(stmt, {"new": r.new_name, "id": r.id, "old": r.old_name})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="perform the renames")
    args = parser.parse_args()

    from app.database import engine

    with engine.begin() as conn:
        rows = load_tenants(conn)
        renames = plan_renames(rows)
        if not renames:
            print("No duplicate tenant names.")
            return 0
        deps = {r.id: r.dependents for r in rows}
        for r in renames:
            print(f"rename {r.id}  {r.old_name!r} -> {r.new_name!r}  ({deps[r.id]} dependent rows)")
        if not args.apply:
            print("Dry run: nothing changed. Re-run with --apply to rename.")
            return 0
        apply_renames(conn, renames)
    print(f"Renamed {len(renames)} tenant(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
