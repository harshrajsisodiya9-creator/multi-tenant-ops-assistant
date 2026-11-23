from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Customer, Tenant
from app.security import hash_key

_JWT_ALGORITHM = "HS256"
_JWT_ISSUER = "storefront"
_JWT_AUDIENCE = "support-agent"
_JWT_EXPIRE_MINUTES = 60
_JWT_MIN_SECRET_LEN = 32

_bearer_scheme = HTTPBearer(auto_error=False)


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


def _env_key_for_tenant(tenant_id: str) -> str:
    return f"JWT_SECRET_{tenant_id.upper().replace('-', '_')}"


def _get_jwt_secret(tenant_id: str) -> str:
    env_key = _env_key_for_tenant(tenant_id)
    secret = os.environ.get(env_key)
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT verification unavailable for this tenant",
        )
    if len(secret) < _JWT_MIN_SECRET_LEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT secret configuration error",
        )
    return secret


@dataclass(frozen=True)
class TokenClaims:
    tenant_id: str
    customer_id: str
    email: str
    role: str | None = None


def mint_token(
    tenant_id: str,
    customer_id: str,
    *,
    role: str | None = None,
    expires_minutes: int = _JWT_EXPIRE_MINUTES,
) -> str:
    secret = _get_jwt_secret(tenant_id)
    now = datetime.now(timezone.utc)
    payload: dict = {
        "sub": customer_id,
        "tenant_id": tenant_id,
        "iss": _JWT_ISSUER,
        "aud": _JWT_AUDIENCE,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    if role:
        payload["role"] = role
    return jwt.encode(payload, secret, algorithm=_JWT_ALGORITHM)


def _decode_token(raw_token: str) -> dict:
    try:
        unverified = jwt.decode(
            raw_token, options={"verify_signature": False}, algorithms=[_JWT_ALGORITHM],
        )
    except jwt.DecodeError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token",
        )

    tenant_id = unverified.get("tenant_id")
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing tenant_id",
        )

    secret = _get_jwt_secret(str(tenant_id))

    try:
        payload = jwt.decode(
            raw_token,
            secret,
            algorithms=[_JWT_ALGORITHM],
            audience=_JWT_AUDIENCE,
            issuer=_JWT_ISSUER,
            options={"require": ["exp", "sub", "iat"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        )

    if "tenant_id" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing tenant_id",
        )

    return payload


def _lookup_customer(db: Session, tenant_id: str, customer_id: str) -> Customer | None:
    stmt = select(Customer).where(
        Customer.tenant_id == UUID(tenant_id),
        Customer.customer_id == customer_id,
    )
    return db.scalar(stmt)


def get_authenticated_customer(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
) -> TokenClaims:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token",
        )

    payload = _decode_token(credentials.credentials)

    token_tenant_id = str(payload["tenant_id"])
    api_tenant_id = str(tenant.id)
    if token_tenant_id != api_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token tenant_id does not match API key tenant",
        )

    customer = _lookup_customer(db, token_tenant_id, payload["sub"])
    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unknown customer",
        )

    return TokenClaims(
        tenant_id=token_tenant_id,
        customer_id=payload["sub"],
        email=customer.email,
        role=payload.get("role"),
    )


def get_reviewer(
    claims: TokenClaims = Depends(get_authenticated_customer),
) -> TokenClaims:
    if claims.role != "reviewer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint requires a reviewer token",
        )
    return claims
