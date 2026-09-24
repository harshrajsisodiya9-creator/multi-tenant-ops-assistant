from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Tenant
from app.security import hash_key


def get_current_tenant(
    x_api_key: str = Header(...),
    db: Session = Depends(get_db),
) -> Tenant:
    api_key_hash = hash_key(x_api_key)

    stmt = select(Tenant).where(Tenant.api_key_hash == api_key_hash)

    tenant = db.scalar(stmt)

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return tenant
