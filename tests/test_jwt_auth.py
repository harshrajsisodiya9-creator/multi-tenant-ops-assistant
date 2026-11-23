"""JWT authentication tests for /agent and /agent/resume endpoints."""
from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import jwt
import pytest
from fastapi.testclient import TestClient

TENANT_A_ID = str(uuid4())
TENANT_B_ID = str(uuid4())
SECRET_A = "a" * 40
SECRET_B = "b" * 40
CUSTOMER_A = "CUST-MM-001"
CUSTOMER_B = "CUST-MM-002"
UNKNOWN_CUSTOMER = "CUST-GHOST-999"
API_KEY_A = "test-api-key-a"
API_KEY_B = "test-api-key-b"


def _env_key(tenant_id: str) -> str:
    return f"JWT_SECRET_{tenant_id.upper().replace('-', '_')}"


def _make_tenant(tid: str, name: str, api_key: str):
    from app.security import hash_key

    t = MagicMock()
    t.id = tid
    t.name = name
    t.api_key_hash = hash_key(api_key)
    return t


TENANT_A = _make_tenant(TENANT_A_ID, "megamart", API_KEY_A)
TENANT_B = _make_tenant(TENANT_B_ID, "tiny-candles", API_KEY_B)
_TENANTS_BY_HASH = {TENANT_A.api_key_hash: TENANT_A, TENANT_B.api_key_hash: TENANT_B}

_KNOWN_CUSTOMERS = {
    (TENANT_A_ID, CUSTOMER_A): MagicMock(
        customer_id=CUSTOMER_A, tenant_id=TENANT_A_ID, email="alice@example.com",
    ),
    (TENANT_A_ID, CUSTOMER_B): MagicMock(
        customer_id=CUSTOMER_B, tenant_id=TENANT_A_ID, email="bob@example.com",
    ),
    (TENANT_A_ID, "REVIEWER-001"): MagicMock(
        customer_id="REVIEWER-001", tenant_id=TENANT_A_ID, email="reviewer@megamart.com",
    ),
    (TENANT_B_ID, "REVIEWER-B"): MagicMock(
        customer_id="REVIEWER-B", tenant_id=TENANT_B_ID, email="reviewer@tinycandles.com",
    ),
}


def _mock_lookup_customer(db, tenant_id, customer_id):
    return _KNOWN_CUSTOMERS.get((tenant_id, customer_id))


def _mock_get_current_tenant(x_api_key, db=None):
    from app.security import hash_key
    from fastapi import HTTPException

    h = hash_key(x_api_key)
    t = _TENANTS_BY_HASH.get(h)
    if t is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return t


def _mint(
    tenant_id: str,
    customer_id: str,
    secret: str,
    *,
    role: str | None = None,
    exp_minutes: int = 60,
    iss: str = "storefront",
    aud: str = "support-agent",
    extra: dict | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": customer_id,
        "tenant_id": tenant_id,
        "iss": iss,
        "aud": aud,
        "iat": now,
        "exp": now + timedelta(minutes=exp_minutes),
    }
    if role:
        payload["role"] = role
    if extra:
        payload.update(extra)
    return jwt.encode(payload, secret, algorithm="HS256")


def _expired_token(tenant_id: str, customer_id: str, secret: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": customer_id,
        "tenant_id": tenant_id,
        "iss": "storefront",
        "aud": "support-agent",
        "iat": now - timedelta(hours=2),
        "exp": now - timedelta(hours=1),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def _headers(api_key: str, token: str) -> dict:
    return {"X-Api-Key": api_key, "Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def _env_secrets(monkeypatch):
    monkeypatch.setenv(_env_key(TENANT_A_ID), SECRET_A)
    monkeypatch.setenv(_env_key(TENANT_B_ID), SECRET_B)
    monkeypatch.setenv("APP_ENV", "test")


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("AGENT_MODE", "mock")
    monkeypatch.setenv("DATABASE_URL", "sqlite://")

    from app.auth import get_current_tenant
    from app.database import get_db
    from app.main import _THREAD_OWNERS, app

    _THREAD_OWNERS.clear()

    def fake_tenant(x_api_key: str = __import__("fastapi").Header(...)):
        return _mock_get_current_tenant(x_api_key)

    def fake_get_db():
        yield MagicMock()

    app.dependency_overrides[get_current_tenant] = fake_tenant
    app.dependency_overrides[get_db] = fake_get_db

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "tenant_id": TENANT_A_ID,
        "user_message": "hi",
        "router_output": {},
        "policy_decision": {},
        "response": "Hello!",
        "response_phase": "complete",
    }

    with (
        patch("app.main.agent_graph", mock_graph),
        patch("app.auth._lookup_customer", side_effect=_mock_lookup_customer),
    ):
        c = TestClient(app, raise_server_exceptions=False)
        yield c, mock_graph

    app.dependency_overrides.clear()
    _THREAD_OWNERS.clear()


class TestValidToken:
    def test_agent_with_valid_token(self, client):
        c, mock_graph = client
        token = _mint(TENANT_A_ID, CUSTOMER_A, SECRET_A)
        resp = c.post(
            "/agent",
            json={"message": "check my order"},
            headers=_headers(API_KEY_A, token),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "Hello!"
        invoke_args = mock_graph.invoke.call_args
        state = invoke_args[0][0]
        assert state["tenant_id"] == TENANT_A_ID
        assert state["customer_id"] == CUSTOMER_A
        assert state["customer_email"] == "alice@example.com"


class TestNoToken:
    def test_agent_missing_authorization_header(self, client):
        c, _ = client
        resp = c.post(
            "/agent",
            json={"message": "hi"},
            headers={"X-Api-Key": API_KEY_A},
        )
        assert resp.status_code == 401


class TestExpiredToken:
    def test_expired_token_returns_401(self, client):
        c, _ = client
        token = _expired_token(TENANT_A_ID, CUSTOMER_A, SECRET_A)
        resp = c.post(
            "/agent",
            json={"message": "hi"},
            headers=_headers(API_KEY_A, token),
        )
        assert resp.status_code == 401
        assert "expired" in resp.json()["detail"].lower()


class TestWrongSignature:
    def test_wrong_secret_returns_401(self, client):
        c, _ = client
        token = _mint(TENANT_A_ID, CUSTOMER_A, "wrong-secret-that-is-32-chars-xx")
        resp = c.post(
            "/agent",
            json={"message": "hi"},
            headers=_headers(API_KEY_A, token),
        )
        assert resp.status_code == 401


class TestWrongAudience:
    def test_wrong_audience_returns_401(self, client):
        c, _ = client
        token = _mint(TENANT_A_ID, CUSTOMER_A, SECRET_A, aud="wrong-audience")
        resp = c.post(
            "/agent",
            json={"message": "hi"},
            headers=_headers(API_KEY_A, token),
        )
        assert resp.status_code == 401


class TestAlgNoneForgery:
    def test_alg_none_rejected(self, client):
        c, _ = client
        payload = {
            "sub": CUSTOMER_A,
            "tenant_id": TENANT_A_ID,
            "iss": "storefront",
            "aud": "support-agent",
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "none", "typ": "JWT"}).encode()
        ).rstrip(b"=").decode()
        body = base64.urlsafe_b64encode(
            json.dumps(payload, default=str).encode()
        ).rstrip(b"=").decode()
        forged = f"{header}.{body}."

        resp = c.post(
            "/agent",
            json={"message": "hi"},
            headers=_headers(API_KEY_A, forged),
        )
        assert resp.status_code == 401


class TestCrossTenant:
    def test_tenant_a_token_on_tenant_b_api_key(self, client):
        c, _ = client
        token = _mint(TENANT_A_ID, CUSTOMER_A, SECRET_A)
        resp = c.post(
            "/agent",
            json={"message": "hi"},
            headers=_headers(API_KEY_B, token),
        )
        assert resp.status_code == 401
        assert "does not match" in resp.json()["detail"].lower()

    def test_tenant_a_token_signed_with_tenant_b_secret(self, client):
        c, _ = client
        token = _mint(TENANT_A_ID, CUSTOMER_A, SECRET_B)
        resp = c.post(
            "/agent",
            json={"message": "hi"},
            headers=_headers(API_KEY_A, token),
        )
        assert resp.status_code == 401


class TestUnknownCustomer:
    def test_valid_signature_unknown_customer_returns_401(self, client):
        c, _ = client
        token = _mint(TENANT_A_ID, UNKNOWN_CUSTOMER, SECRET_A)
        resp = c.post(
            "/agent",
            json={"message": "hi"},
            headers=_headers(API_KEY_A, token),
        )
        assert resp.status_code == 401
        assert "unknown customer" in resp.json()["detail"].lower()


class TestBodyEmailIgnored:
    def test_customer_email_in_body_ignored(self, client):
        c, mock_graph = client
        token = _mint(TENANT_A_ID, CUSTOMER_A, SECRET_A)
        resp = c.post(
            "/agent",
            json={"message": "hi", "customer_email": "evil@example.com"},
            headers=_headers(API_KEY_A, token),
        )
        assert resp.status_code == 200
        state = mock_graph.invoke.call_args[0][0]
        assert state["customer_email"] == "alice@example.com"


class TestCustomerCrossAccess:
    def test_customer_a_asks_about_customer_b_order(self, client):
        c, mock_graph = client
        token_a = _mint(TENANT_A_ID, CUSTOMER_A, SECRET_A)
        resp = c.post(
            "/agent",
            json={"message": "check order ORD-MM-1002"},
            headers=_headers(API_KEY_A, token_a),
        )
        assert resp.status_code == 200
        state = mock_graph.invoke.call_args[0][0]
        assert state["customer_id"] == CUSTOMER_A


class TestOwnerScopedQueries:
    """Tool-level tests: owner-scoped queries prevent cross-customer data access."""

    @pytest.fixture()
    def tool_db(self):
        from uuid import UUID as _UUID

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        from app.database import Base
        from app.models import Customer, Order, OrderItem, RefundHistory, Tenant

        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        tables = [
            Tenant.__table__, Customer.__table__, Order.__table__,
            OrderItem.__table__, RefundHistory.__table__,
        ]
        Base.metadata.create_all(engine, tables=tables)
        TestSession = sessionmaker(bind=engine)

        db = TestSession()
        db.add(Tenant(id=_UUID(TENANT_A_ID), name="megamart", api_key_hash="x"))
        db.add(Customer(customer_id=CUSTOMER_A, tenant_id=_UUID(TENANT_A_ID), email="alice@example.com", name="Alice"))
        db.add(Customer(customer_id=CUSTOMER_B, tenant_id=_UUID(TENANT_A_ID), email="bob@example.com", name="Bob"))
        db.flush()
        db.add(Order(
            order_id="ORD-MM-1002", tenant_id=_UUID(TENANT_A_ID),
            customer_id=CUSTOMER_B, customer_email="bob@example.com",
            product="Wireless Mouse", amount=1500, total_amount=1500,
            status="delivered", category="electronics",
        ))
        db.flush()
        db.add(OrderItem(
            item_id="ITEM-1002-1", order_id="ORD-MM-1002",
            product="Wireless Mouse", category="electronics",
            amount=1500, status="delivered",
        ))
        db.commit()
        db.close()

        yield TestSession

    def test_cross_customer_order_not_found(self, tool_db):
        from app.graph.agent_prod.tool_context import ToolContext
        from app.graph.agent_prod.tools.order_tools import get_order_details

        ctx_a = ToolContext(tenant_id=TENANT_A_ID, customer_id=CUSTOMER_A)
        with patch("app.graph.agent_prod.tools.order_tools._get_db", return_value=tool_db()):
            result = get_order_details(ctx_a, "ORD-MM-1002")

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_cross_customer_refund_not_found_no_db_write(self, tool_db):
        from app.graph.agent_prod.tools.order_tools import initiate_refund
        from app.models import RefundHistory

        from app.graph.agent_prod.tool_context import ToolContext

        ctx_a = ToolContext(tenant_id=TENANT_A_ID, customer_id=CUSTOMER_A)
        with patch("app.graph.agent_prod.tools.order_tools._get_db", return_value=tool_db()):
            result = initiate_refund(ctx_a, "ORD-MM-1002", 1500)

        assert "error" in result
        assert "not found" in result["error"].lower()

        check_db = tool_db()
        try:
            assert check_db.query(RefundHistory).count() == 0
        finally:
            check_db.close()

    def test_owner_can_access_own_order(self, tool_db):
        from app.graph.agent_prod.tool_context import ToolContext
        from app.graph.agent_prod.tools.order_tools import get_order_details

        ctx_b = ToolContext(tenant_id=TENANT_A_ID, customer_id=CUSTOMER_B)
        with patch("app.graph.agent_prod.tools.order_tools._get_db", return_value=tool_db()):
            result = get_order_details(ctx_b, "ORD-MM-1002")

        assert "error" not in result
        assert result["order_id"] == "ORD-MM-1002"


class TestResumeRequiresReviewer:
    def test_customer_token_on_resume_returns_403(self, client):
        c, _ = client
        from app.main import _THREAD_OWNERS

        thread_id = "test-thread-123"
        _THREAD_OWNERS[thread_id] = (TENANT_A_ID, CUSTOMER_A)

        token = _mint(TENANT_A_ID, CUSTOMER_A, SECRET_A)
        resp = c.post(
            "/agent/resume",
            json={
                "thread_id": thread_id,
                "decisions": [{"index": 0, "action": "approve"}],
            },
            headers=_headers(API_KEY_A, token),
        )
        assert resp.status_code == 403
        assert "reviewer" in resp.json()["detail"].lower()

    def test_reviewer_token_on_resume_succeeds(self, client):
        c, mock_graph = client
        from app.main import _THREAD_OWNERS

        thread_id = "test-thread-456"
        _THREAD_OWNERS[thread_id] = (TENANT_A_ID, CUSTOMER_A)

        mock_graph.invoke.return_value = {
            "tenant_id": TENANT_A_ID,
            "response": "Approved.",
            "intent_results": [],
        }

        token = _mint(TENANT_A_ID, "REVIEWER-001", SECRET_A, role="reviewer")
        resp = c.post(
            "/agent/resume",
            json={
                "thread_id": thread_id,
                "decisions": [{"index": 0, "action": "approve"}],
            },
            headers=_headers(API_KEY_A, token),
        )
        assert resp.status_code == 200


class TestThreadOwnership:
    def test_reuse_another_customers_thread(self, client):
        c, _ = client
        from app.main import _THREAD_OWNERS

        thread_id = "thread-owned-by-a"
        _THREAD_OWNERS[thread_id] = (TENANT_A_ID, CUSTOMER_A)

        token_b = _mint(TENANT_A_ID, CUSTOMER_B, SECRET_A)
        resp = c.post(
            "/agent",
            json={"message": "hi", "thread_id": thread_id},
            headers=_headers(API_KEY_A, token_b),
        )
        assert resp.status_code == 403
        assert "different customer" in resp.json()["detail"].lower()

    def test_own_thread_reuse_succeeds(self, client):
        c, _ = client
        from app.main import _THREAD_OWNERS

        thread_id = "thread-owned-by-a"
        _THREAD_OWNERS[thread_id] = (TENANT_A_ID, CUSTOMER_A)

        token_a = _mint(TENANT_A_ID, CUSTOMER_A, SECRET_A)
        resp = c.post(
            "/agent",
            json={"message": "follow up", "thread_id": thread_id},
            headers=_headers(API_KEY_A, token_a),
        )
        assert resp.status_code == 200

    def test_resume_cross_tenant_thread(self, client):
        c, _ = client
        from app.main import _THREAD_OWNERS

        thread_id = "thread-tenant-a"
        _THREAD_OWNERS[thread_id] = (TENANT_A_ID, CUSTOMER_A)

        token_b_reviewer = _mint(TENANT_B_ID, "REVIEWER-B", SECRET_B, role="reviewer")
        resp = c.post(
            "/agent/resume",
            json={
                "thread_id": thread_id,
                "decisions": [{"index": 0, "action": "approve"}],
            },
            headers=_headers(API_KEY_B, token_b_reviewer),
        )
        assert resp.status_code == 403
        assert "different tenant" in resp.json()["detail"].lower()


class TestDevTokenEndpoint:
    def test_dev_token_available_in_test(self, client):
        c, _ = client
        resp = c.post(
            "/dev/token",
            json={
                "tenant_id": TENANT_A_ID,
                "customer_id": CUSTOMER_A,
            },
        )
        assert resp.status_code == 200
        token = resp.json()["token"]
        decoded = jwt.decode(
            token, SECRET_A, algorithms=["HS256"],
            audience="support-agent", issuer="storefront",
        )
        assert decoded["sub"] == CUSTOMER_A
        assert decoded["tenant_id"] == TENANT_A_ID

    def test_dev_token_absent_in_prod(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "prod")
        monkeypatch.setenv("DATABASE_URL", "sqlite://")
        monkeypatch.setenv("AGENT_MODE", "prod")

        import importlib
        import app.config
        importlib.reload(app.config)

        import app.main
        importlib.reload(app.main)

        from app.main import app as prod_app
        c = TestClient(prod_app, raise_server_exceptions=False)

        resp = c.post(
            "/dev/token",
            json={"tenant_id": TENANT_A_ID, "customer_id": CUSTOMER_A},
        )
        assert resp.status_code == 404 or resp.status_code == 405

        importlib.reload(app.config)
        importlib.reload(app.main)


class TestProdEntityResolution:
    """Prod-mode tests: entity resolution and gate context read from real DB,
    not mock_data. Uses SQLite in-memory DB with StaticPool."""

    @pytest.fixture()
    def prod_db(self):
        from datetime import date as _date
        from uuid import UUID as _UUID

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        from app.database import Base
        from app.models import Customer, Order, OrderItem, RefundHistory, Tenant

        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        tables = [
            Tenant.__table__, Customer.__table__, Order.__table__,
            OrderItem.__table__, RefundHistory.__table__,
        ]
        Base.metadata.create_all(engine, tables=tables)
        ProdSession = sessionmaker(bind=engine)

        db = ProdSession()
        db.add(Tenant(id=_UUID(TENANT_A_ID), name="megamart", api_key_hash="x"))
        db.add(Customer(
            customer_id=CUSTOMER_A, tenant_id=_UUID(TENANT_A_ID),
            email="alice@example.com", name="Alice",
        ))
        db.add(Customer(
            customer_id=CUSTOMER_B, tenant_id=_UUID(TENANT_A_ID),
            email="bob@example.com", name="Bob",
        ))
        db.flush()
        db.add(Order(
            order_id="ORD-PROD-001", tenant_id=_UUID(TENANT_A_ID),
            customer_id=CUSTOMER_A, customer_email="alice@example.com",
            product="Wireless Headphones", amount=4500, total_amount=4500,
            status="delivered", category="electronics",
            delivery_date=_date(2026, 11, 10),
        ))
        db.add(Order(
            order_id="ORD-PROD-002", tenant_id=_UUID(TENANT_A_ID),
            customer_id=CUSTOMER_B, customer_email="bob@example.com",
            product="Cotton T-Shirt", amount=1200, total_amount=1200,
            status="delivered", category="clothing",
            delivery_date=_date(2026, 11, 15),
        ))
        db.add(Order(
            order_id="ORD-PROD-003", tenant_id=_UUID(TENANT_A_ID),
            customer_id=CUSTOMER_A, customer_email="alice@example.com",
            product="Laptop Stand", amount=2500, total_amount=2500,
            status="delivered", category="electronics",
            delivery_date=_date(2026, 11, 12),
        ))
        db.flush()
        db.add(OrderItem(
            item_id="ITEM-P001-1", order_id="ORD-PROD-001",
            product="Wireless Headphones", category="electronics",
            amount=4500, status="delivered",
        ))
        db.add(OrderItem(
            item_id="ITEM-P002-1", order_id="ORD-PROD-002",
            product="Cotton T-Shirt", category="clothing",
            amount=1200, status="delivered",
        ))
        db.add(OrderItem(
            item_id="ITEM-P003-1", order_id="ORD-PROD-003",
            product="Laptop Stand", category="electronics",
            amount=2500, status="delivered",
        ))
        db.commit()
        db.close()

        import app.graph.data_access as _da
        with (
            patch.object(_da.setting, "agent_mode", "prod"),
            patch("app.database.SessionLocal", ProdSession),
        ):
            yield ProdSession

    def test_product_name_resolves_to_db_order(self, prod_db):
        """'return my headphones' resolves to ORD-PROD-001 via real DB."""
        from app.graph.nodes.policy_fan_out import _resolve_order_entity

        entities, early = _resolve_order_entity(
            TENANT_A_ID, CUSTOMER_A, "refund",
            {}, "I want to return my headphones",
        )
        assert early is None
        assert entities.get("order_id") == "ORD-PROD-001"

    def test_gate_reads_facts_from_db_row(self, prod_db):
        """Gate context reads amount/category/delivery_date from DB order."""
        from app.graph.nodes.policy_fan_out import _build_gate_context

        ctx = _build_gate_context(
            tenant_id=TENANT_A_ID,
            customer_id=CUSTOMER_A,
            tenant_info={"active": True},
            intent="refund",
            entities={"order_id": "ORD-PROD-001"},
            confidence=0.95,
            escalation_signals=[],
        )
        assert ctx.order is not None
        assert ctx.order["amount"] == 4500
        assert ctx.order["category"] == "electronics"
        assert ctx.order["delivery_date"] == "2026-11-10"

    def test_mock_order_id_not_found_in_prod(self, prod_db):
        """ORD-MM-1001 from mock_data doesn't exist in the prod DB."""
        from app.graph.nodes.policy_fan_out import _build_gate_context

        ctx = _build_gate_context(
            tenant_id=TENANT_A_ID,
            customer_id=CUSTOMER_A,
            tenant_info={"active": True},
            intent="refund",
            entities={"order_id": "ORD-MM-1001"},
            confidence=0.95,
            escalation_signals=[],
        )
        assert ctx.order is None

    def test_cross_customer_entity_resolution_isolated(self, prod_db):
        """Customer A cannot resolve Customer B's order by product name.
        Alice has 2 orders (headphones, laptop stand), so 'cotton t-shirt'
        matches neither and triggers MISSING_ORDER_ID clarification."""
        from app.graph.nodes.policy_fan_out import _resolve_order_entity

        entities, early = _resolve_order_entity(
            TENANT_A_ID, CUSTOMER_A, "refund",
            {}, "return my cotton t-shirt",
        )
        assert early is not None
        assert early.reason_code == "MISSING_ORDER_ID"
        assert entities.get("order_id") is None
