# Database operations

## PostgreSQL image

`docker-compose.yml` pins:

```
pgvector/pgvector:0.8.7-pg16-trixie@sha256:c28c0b700f3f1b5eec14082b31ad6e7c8a566d14386323609c615315e7c0ffc8
```

PostgreSQL 16.16, pgvector 0.8.7, Debian 13 "trixie", glibc 2.41.

- Text sort order comes from the operating system's C library (glibc). A btree index on a
  `text`/`varchar` column is only valid while that sort order stays the same.
- The floating `pg16` tag currently resolves to Debian 12 "bookworm" (glibc 2.36). The
  existing dev volume was created under glibc 2.41, so starting it with `pg16` makes
  PostgreSQL warn about a collation version mismatch.
- Always pin by digest. Use the same pinned reference in every environment that shares
  a data directory or restores a physical backup.
- To pick up a pgvector or PostgreSQL patch release, choose a new tag **for the same
  Debian release** (`*-pg16-trixie`) and pin its digest. That doesn't change glibc.
  Afterwards, optionally run `ALTER EXTENSION vector UPDATE;` in each database.

## Collation version mismatch

### Detect

```sql
SELECT datname, datcollversion,
       pg_database_collation_actual_version(oid) AS os_version
FROM pg_database WHERE datallowconn;
```

If `datcollversion` and `os_version` differ, the warning appears on every connection.
Indexes on text columns may then return wrong results or let duplicates past a unique
index.

### Resolve (safest first)

1. **Go back to an image with the original glibc.** This needs no maintenance and is
   right for the dev volume today: use the pinned trixie image above.
2. **Dump and restore** if you must change the OS. Take a logical dump
   (`pg_dump -Fc` per database, plus `pg_dumpall --globals-only`), initialise a new
   volume with the new image, then `pg_restore`. Every index is rebuilt under the new
   collation.
3. **Rebuild in place.** Only with a verified backup and a maintenance window:
   1. Check whether any btree index already disagrees with the new sort order:
      ```sql
      CREATE EXTENSION IF NOT EXISTS amcheck;
      SELECT c.relname, bt_index_check(c.oid, true)
      FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
      JOIN pg_am a ON a.oid = c.relam
      WHERE a.amname = 'btree' AND c.relnamespace = 'public'::regnamespace;
      ```
   2. Rebuild the indexes: `REINDEX DATABASE CONCURRENTLY <db>;` for user indexes, then
      `REINDEX SYSTEM <db>;`. If a unique index fails to rebuild, the mismatch has let
      duplicates in. Resolve them, then retry.
   3. Only then record the new version: `ALTER DATABASE <db> REFRESH COLLATION VERSION;`.
      Do this for every database, including `postgres` and `template1`.

Never run `REFRESH COLLATION VERSION` on its own. It only silences the warning and leaves
broken indexes in place.

## Migrations on an existing database

- `alembic upgrade head` runs every pending revision in **one transaction**. If any
  revision fails, the database stays at its starting revision.
- `ff724bd9c571` (unique `tenants.name`) refuses to run while duplicate names exist.
  Resolve them first:

  ```
  python -m scripts.dedupe_tenant_names          # dry run: prints the rename plan
  python -m scripts.dedupe_tenant_names --apply  # renames; no ids, keys or rows change
  alembic upgrade head
  ```

  The tenant with the most dependent rows keeps the name. Each other tenant becomes
  `"<name> [<first 8 hex of id>]"`. To bring in the other pending revisions before
  deduplicating, run `alembic upgrade 63b935d2d7a7`.
- `547731e4dfa3` refuses to run if `customers`, `orders`, `order_items` or
  `refund_history` already exist (for example, created by `Base.metadata.create_all`).
  Check that they match the migration, then run `alembic stamp` with the matching
  revision.
- The role that runs migrations needs `CREATE EXTENSION` (for `vector` and `pg_trgm`).

## Migration regression tests

```
MIGRATION_TEST_DATABASE_URL=postgresql+psycopg://postgres:<pw>@<host>:<port>/postgres \
  pytest tests/test_migrations.py
```

The role needs `CREATE DATABASE`. Each test creates and drops its own database. Without
the variable, the tests are skipped.
