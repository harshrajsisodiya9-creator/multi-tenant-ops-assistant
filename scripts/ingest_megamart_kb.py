"""Ingest all knowledge base files from files/ into the megamart tenant.

Usage:
    python -m scripts.ingest_megamart_kb
"""
import secrets
from pathlib import Path

from sqlalchemy import select

from app.database import SessionLocal
from app.knowledgebase import index_document
from app.models import Tenant
from app.security import hash_key

FILES_DIR = Path(__file__).resolve().parent.parent / "files"


def main():
    db = SessionLocal()
    try:
        tenant = db.scalar(select(Tenant).where(Tenant.name == "megamart"))
        if not tenant:
            raw_key = secrets.token_urlsafe(32)
            tenant = Tenant(name="megamart", api_key_hash=hash_key(raw_key))
            db.add(tenant)
            db.commit()
            db.refresh(tenant)
            print(f"Created tenant megamart ({tenant.id})")
            print(f"API key: {raw_key}")
        else:
            print(f"Found tenant megamart ({tenant.id})")

        md_files = sorted(FILES_DIR.glob("*.md"))
        if not md_files:
            print(f"No .md files found in {FILES_DIR}")
            return

        for path in md_files:
            doc_id = path.stem
            text = path.read_text()
            count = index_document(
                db=db,
                tenant_id=tenant.id,
                source_document_id=doc_id,
                text=text,
            )
            print(f"  {doc_id}: {count} chunks")

        print("Done.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
