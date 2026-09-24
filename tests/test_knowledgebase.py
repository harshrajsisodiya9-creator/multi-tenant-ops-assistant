import asyncio
import io
from uuid import uuid4

from fastapi import UploadFile

from app.database import SessionLocal
from app.knowledgebase import answer_question, index_document, read_uploaded_document
from app.models import Tenant
from app.security import hash_key


def test_index_and_answer_question():
    session = SessionLocal()
    try:
        tenant = Tenant(name=f"kb-test-{uuid4()}", api_key_hash=hash_key("demo-key"))
        session.add(tenant)
        session.commit()
        session.refresh(tenant)

        doc_text = "FastAPI is a modern Python web framework for building APIs. It is designed for speed and simplicity."
        stored = index_document(
            db=session,
            tenant_id=tenant.id,
            source_document_id="doc-1",
            text=doc_text,
        )
        assert stored >= 1

        result = answer_question(
            db=session,
            tenant_id=tenant.id,
            question="What is FastAPI?",
            limit=3,
        )
        assert "FastAPI" in result["answer"]
        assert result["matches_found"] >= 1
    finally:
        session.close()


def test_read_uploaded_document_validates_empty_and_unsupported_files():
    empty_file = UploadFile(filename="empty.txt", file=io.BytesIO(b""))
    try:
        asyncio.run(read_uploaded_document(empty_file))
        raise AssertionError("empty files should raise ValueError")
    except ValueError as exc:
        assert "empty or unreadable" in str(exc)

    unsupported_file = UploadFile(filename="notes.csv", file=io.BytesIO(b"name,age\n"))
    try:
        asyncio.run(read_uploaded_document(unsupported_file))
        raise AssertionError("unsupported files should raise ValueError")
    except ValueError as exc:
        assert "Unsupported file type" in str(exc)
