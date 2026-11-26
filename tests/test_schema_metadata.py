"""ORM declarations that must stay in step with the migrated schema."""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import Chunk, OperationLog, Tenant


def test_chunk_declares_gin_index_on_content_tsv():
    index = next(i for i in Chunk.__table__.indexes if i.name == "ix_chunks_content_tsv")
    assert [c.name for c in index.columns] == ["content_tsv"]
    assert index.dialect_options["postgresql"]["using"] == "gin"


def test_operation_log_has_python_and_server_defaults():
    cols = OperationLog.__table__.c
    assert cols.status.default.arg == "pending"
    assert cols.status.server_default.arg == "pending"
    assert cols.request_args.default.arg == "{}"
    assert cols.request_args.server_default.arg == "{}"


def test_operation_log_defaults_apply_for_orm_and_raw_inserts():
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine, tables=[Tenant.__table__, OperationLog.__table__])
    tenant_id = uuid4()
    with Session(engine) as db:
        db.add(Tenant(id=tenant_id, name="t", api_key_hash="k"))
        db.add(OperationLog(tenant_id=tenant_id, customer_id="C", idempotency_key="orm", operation="refund"))
        db.commit()
        db.execute(text(
            "INSERT INTO operation_log (tenant_id, customer_id, idempotency_key, operation) "
            "VALUES (:t, 'C', 'raw', 'refund')"
        ), {"t": tenant_id.hex})
        db.commit()
        rows = db.execute(text(
            "SELECT idempotency_key, status, request_args FROM operation_log ORDER BY idempotency_key"
        )).all()
    assert [tuple(r) for r in rows] == [("orm", "pending", "{}"), ("raw", "pending", "{}")]
    engine.dispose()
