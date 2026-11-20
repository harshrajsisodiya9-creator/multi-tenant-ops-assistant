from pathlib import Path
from uuid import UUID

from app.database import get_db
from app.knowledgebase import index_document

TENANT_ID = UUID("f67e4814-c4bf-421b-94d8-0d5894703816")
FILES_DIR = Path(__file__).resolve().parent.parent / "files"

FILES_TO_INGEST = [
    "product_catalog.md",
    "refund_policy_v1.md",
    "support_internal_faq.md",
]


def main():
    db = next(get_db())
    try:
        for filename in FILES_TO_INGEST:
            path = FILES_DIR / filename
            text = path.read_text(encoding="utf-8")
            count = index_document(
                db=db,
                tenant_id=TENANT_ID,
                source_document_id=path.stem,
                text=text,
            )
            print(f"{filename}: {count} chunks ingested")
    finally:
        db.close()


if __name__ == "__main__":
    main()
