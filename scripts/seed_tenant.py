import secrets

from app.database import SessionLocal
from app.models import Tenant
from app.security import hash_key


def seed_tenant():
    db = SessionLocal()
    raw_key = secrets.token_urlsafe(32)
    hashed = hash_key(raw_key)
    try:
        tenant = Tenant(name="Test Tenant", api_key_hash=hashed)

        db.add(tenant)
        db.commit()
        db.refresh(tenant)

        print(f"Created tenant: {tenant.id}")
        print(f"Api-key: {raw_key}")
    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    seed_tenant()
