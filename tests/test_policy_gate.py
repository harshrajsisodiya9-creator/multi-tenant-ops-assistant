from __future__ import annotations

from datetime import date

import pytest

from app.graph.policy_gate import GateContext, evaluate_policy
from app.graph.schemas import RouterOutput

TODAY = date(2026, 10, 4)

MEGAMART_CONFIG = {
    "tenant_id": "megamart",
    "active": True,
    "support_email": "support@megamart.com",
    "max_auto_refund": 10000,
    "max_refunds_per_customer_30d": 3,
    "refund_always_escalate": False,
    "default": {
        "return_policy": {
            "allowed": True,
            "window_days": 14,
            "condition": "any",
            "refund_type": "original_method",
        },
        "exchange_policy": {
            "allowed": True,
            "window_days": 15,
            "same_category_only": False,
        },
    },
    "categories": {
        "electronics": {"return_policy": {"window_days": 30, "condition": "unopened"}},
        "clothing": {
            "return_policy": {"window_days": 7},
            "exchange_policy": {"same_category_only": True},
        },
        "perishables": {
            "return_policy": {"allowed": False},
            "exchange_policy": {"allowed": False},
        },
    },
    "subscription": {
        "cancel_allowed": True,
        "cancel_notice_days": 3,
        "prorated_refund": True,
        "min_commitment_months": 0,
    },
}

TINY_CANDLES_CONFIG = {
    "tenant_id": "tiny-candles",
    "active": True,
    "support_email": "hello@tinycandles.in",
    "max_auto_refund": None,
    "max_refunds_per_customer_30d": 5,
    "refund_always_escalate": True,
    "default": {
        "return_policy": {
            "allowed": True,
            "window_days": 14,
            "condition": "any",
            "refund_type": "original_method",
        },
        "exchange_policy": {
            "allowed": True,
            "window_days": 14,
            "same_category_only": False,
        },
    },
    "categories": {
        "subscription_box": {
            "subscription_policy": {
                "cancel_allowed": True,
                "min_commitment_months": 1,
            },
        },
    },
    "subscription": {
        "cancel_allowed": True,
        "cancel_notice_days": 0,
        "prorated_refund": False,
        "min_commitment_months": 0,
    },
}

INACTIVE_CONFIG = {"tenant_id": "inactive-co", "active": False}


def _ctx(
    tenant_id: str,
    intent: str,
    confidence: float = 0.95,
    entities: dict | None = None,
    escalation_signals: list[str] | None = None,
    tenant_active: bool = True,
    customer_email: str | None = None,
    order: dict | None = None,
    refund_count_30d: int = 0,
) -> GateContext:
    return GateContext(
        tenant_id=tenant_id,
        tenant_active=tenant_active,
        router_output=RouterOutput(
            intent=intent,
            entities=entities or {},
            confidence=confidence,
            escalation_signals=escalation_signals or [],
        ),
        customer_email=customer_email,
        order=order,
        refund_count_30d=refund_count_30d,
        today=TODAY,
    )


# ── Layer 1: global hard rules ──────────────────────────────────────


class TestGlobalHardRules:
    def test_inactive_tenant(self):
        ctx = _ctx("inactive-co", "faq", tenant_active=False)
        d = evaluate_policy(ctx, tenant_config=INACTIVE_CONFIG)
        assert d.action == "block"
        assert d.reason_code == "TENANT_INACTIVE"

    def test_low_confidence(self):
        ctx = _ctx("megamart", "faq", confidence=0.4)
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "clarify"
        assert d.reason_code == "LOW_CONFIDENCE"

    def test_confidence_boundary_below(self):
        ctx = _ctx("megamart", "faq", confidence=0.59)
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "clarify"
        assert d.reason_code == "LOW_CONFIDENCE"

    def test_confidence_boundary_at(self):
        ctx = _ctx("megamart", "faq", confidence=0.6)
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "allow"

    def test_out_of_scope(self):
        ctx = _ctx("megamart", "out_of_scope")
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "block"
        assert d.reason_code == "OUT_OF_SCOPE"

    def test_order_intent_no_identifiers(self):
        ctx = _ctx("megamart", "order_status")
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "clarify"
        assert d.reason_code == "VERIFICATION_REQUIRED"

    def test_refund_intent_no_identifiers(self):
        ctx = _ctx("megamart", "refund")
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "clarify"
        assert d.reason_code == "VERIFICATION_REQUIRED"

    def test_cancel_intent_no_identifiers(self):
        ctx = _ctx("megamart", "cancel_order")
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "clarify"
        assert d.reason_code == "VERIFICATION_REQUIRED"

    def test_identity_mismatch(self):
        order = {
            "order_id": "ORD-MM-1001",
            "customer_email": "alice@example.com",
            "category": "electronics",
            "amount": 4500,
            "status": "delivered",
            "delivery_date": "2026-09-20",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-MM-1001", "email": "wrong@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "clarify"
        assert d.reason_code == "IDENTITY_MISMATCH"

    def test_identity_match_passes(self):
        order = {
            "order_id": "ORD-MM-1001",
            "customer_email": "alice@example.com",
            "category": "electronics",
            "amount": 4500,
            "status": "delivered",
            "delivery_date": "2026-09-20",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-MM-1001", "email": "alice@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "allow"
        assert d.reason_code == "REFUND_ALLOWED"


# ── Layer 2: global safety ceiling ───────────────────────────────────


class TestGlobalCeiling:
    def test_refund_over_global_ceiling(self):
        order = {
            "order_id": "ORD-MM-1004",
            "customer_email": "dave@example.com",
            "category": "electronics",
            "amount": 65000,
            "status": "delivered",
            "delivery_date": "2026-09-10",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-MM-1004", "email": "dave@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "escalate"
        assert d.reason_code == "GLOBAL_REFUND_CEILING"

    def test_refund_at_ceiling_not_escalated(self):
        order = {
            "order_id": "ORD-X",
            "customer_email": "x@example.com",
            "category": "electronics",
            "amount": 50000,
            "status": "delivered",
            "delivery_date": "2026-09-20",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-X", "email": "x@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.reason_code != "GLOBAL_REFUND_CEILING"


# ── Layer 3: tenant intent rules ─────────────────────────────────────


class TestFAQ:
    def test_faq_megamart(self):
        ctx = _ctx("megamart", "faq")
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "allow"
        assert d.reason_code == "FAQ_ALLOWED"

    def test_faq_tiny_candles(self):
        ctx = _ctx("tiny-candles", "faq")
        d = evaluate_policy(ctx, tenant_config=TINY_CANDLES_CONFIG)
        assert d.action == "allow"
        assert d.reason_code == "FAQ_ALLOWED"


class TestOrderStatus:
    def test_order_status_megamart(self):
        ctx = _ctx(
            "megamart",
            "order_status",
            entities={"order_id": "ORD-MM-1001"},
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "allow"
        assert d.reason_code == "ORDER_STATUS_ALLOWED"
        assert d.constraints["allowed_tools"] == ["get_order_status"]

    def test_order_status_tiny_candles(self):
        ctx = _ctx(
            "tiny-candles",
            "order_status",
            entities={"order_id": "ORD-TC-2001"},
        )
        d = evaluate_policy(ctx, tenant_config=TINY_CANDLES_CONFIG)
        assert d.action == "allow"


class TestComplaint:
    def test_complaint_no_keywords(self):
        ctx = _ctx("megamart", "complaint")
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "allow"
        assert d.reason_code == "COMPLAINT_ALLOWED"

    def test_complaint_with_escalation_keywords(self):
        ctx = _ctx(
            "megamart",
            "complaint",
            escalation_signals=["chargeback", "legal"],
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "escalate"
        assert d.reason_code == "ESCALATION_KEYWORDS_DETECTED"


class TestRefund:
    def test_refund_allowed_megamart_electronics_within_window(self):
        order = {
            "order_id": "ORD-MM-1001",
            "customer_email": "alice@example.com",
            "category": "electronics",
            "amount": 4500,
            "status": "delivered",
            "delivery_date": "2026-09-20",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-MM-1001", "email": "alice@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "allow"
        assert d.reason_code == "REFUND_ALLOWED"
        assert d.constraints["condition"] == "unopened"

    def test_refund_clothing_window_expired(self):
        order = {
            "order_id": "ORD-MM-1002",
            "customer_email": "bob@example.com",
            "category": "clothing",
            "amount": 1200,
            "status": "delivered",
            "delivery_date": "2026-09-15",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-MM-1002", "email": "bob@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "block"
        assert d.reason_code == "RETURN_WINDOW_EXPIRED"

    def test_refund_perishables_not_allowed(self):
        order = {
            "order_id": "ORD-MM-1003",
            "customer_email": "charlie@example.com",
            "category": "perishables",
            "amount": 850,
            "status": "delivered",
            "delivery_date": "2026-09-28",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-MM-1003", "email": "charlie@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "block"
        assert d.reason_code == "RETURNS_NOT_ALLOWED"

    def test_refund_over_tenant_auto_limit(self):
        order = {
            "order_id": "ORD-X",
            "customer_email": "x@example.com",
            "category": "electronics",
            "amount": 15000,
            "status": "delivered",
            "delivery_date": "2026-09-20",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-X", "email": "x@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "escalate"
        assert d.reason_code == "REFUND_OVER_LIMIT"

    def test_refund_rate_limited(self):
        order = {
            "order_id": "ORD-MM-1001",
            "customer_email": "alice@example.com",
            "category": "electronics",
            "amount": 4500,
            "status": "delivered",
            "delivery_date": "2026-09-20",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-MM-1001", "email": "alice@example.com"},
            order=order,
            refund_count_30d=3,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "escalate"
        assert d.reason_code == "REFUND_RATE_LIMITED"

    def test_refund_tiny_candles_always_escalate(self):
        order = {
            "order_id": "ORD-TC-2001",
            "customer_email": "grace@example.com",
            "category": "candles",
            "amount": 1450,
            "status": "delivered",
            "delivery_date": "2026-09-25",
        }
        ctx = _ctx(
            "tiny-candles",
            "refund",
            entities={"order_id": "ORD-TC-2001", "email": "grace@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=TINY_CANDLES_CONFIG)
        assert d.action == "escalate"
        assert d.reason_code == "TENANT_REFUND_REVIEW_REQUIRED"

    def test_refund_under_limit_megamart(self):
        order = {
            "order_id": "ORD-MM-1007",
            "customer_email": "frank@example.com",
            "category": "electronics",
            "amount": 1800,
            "status": "delivered",
            "delivery_date": "2026-09-20",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-MM-1007", "email": "frank@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "allow"
        assert d.reason_code == "REFUND_ALLOWED"

    def test_refund_default_category_14_day_window_within(self):
        order = {
            "order_id": "ORD-X",
            "customer_email": "x@example.com",
            "category": "home_decor",
            "amount": 2000,
            "status": "delivered",
            "delivery_date": "2026-09-25",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-X", "email": "x@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "allow"

    def test_refund_default_category_14_day_window_expired(self):
        order = {
            "order_id": "ORD-X",
            "customer_email": "x@example.com",
            "category": "home_decor",
            "amount": 2000,
            "status": "delivered",
            "delivery_date": "2026-09-10",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-X", "email": "x@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "block"
        assert d.reason_code == "RETURN_WINDOW_EXPIRED"


class TestCancel:
    def test_cancel_pending_order(self):
        order = {
            "order_id": "ORD-MM-1007",
            "customer_email": "frank@example.com",
            "category": "electronics",
            "amount": 1800,
            "status": "pending",
        }
        ctx = _ctx(
            "megamart",
            "cancel_order",
            entities={"order_id": "ORD-MM-1007", "email": "frank@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "allow"
        assert d.reason_code == "CANCEL_ALLOWED"

    def test_cancel_processing_order(self):
        order = {
            "order_id": "ORD-MM-1005",
            "customer_email": "alice@example.com",
            "category": "clothing",
            "amount": 3800,
            "status": "processing",
        }
        ctx = _ctx(
            "megamart",
            "cancel_order",
            entities={"order_id": "ORD-MM-1005", "email": "alice@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "allow"

    def test_cancel_shipped_order_blocked(self):
        order = {
            "order_id": "ORD-MM-1006",
            "customer_email": "eve@example.com",
            "category": "electronics",
            "amount": 2500,
            "status": "shipped",
        }
        ctx = _ctx(
            "megamart",
            "cancel_order",
            entities={"order_id": "ORD-MM-1006", "email": "eve@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "block"
        assert d.reason_code == "ORDER_NOT_CANCELLABLE"

    def test_cancel_delivered_order_blocked(self):
        order = {
            "order_id": "ORD-MM-1001",
            "customer_email": "alice@example.com",
            "category": "electronics",
            "amount": 4500,
            "status": "delivered",
        }
        ctx = _ctx(
            "megamart",
            "cancel_order",
            entities={"order_id": "ORD-MM-1001", "email": "alice@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "block"
        assert d.reason_code == "ORDER_NOT_CANCELLABLE"

    def test_cancel_subscription_within_min_commitment(self):
        order = {
            "order_id": "ORD-TC-2003",
            "customer_email": "iris@example.com",
            "category": "subscription_box",
            "amount": 999,
            "status": "active",
            "order_date": "2026-09-20",
        }
        ctx = _ctx(
            "tiny-candles",
            "cancel_order",
            entities={"order_id": "ORD-TC-2003", "email": "iris@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=TINY_CANDLES_CONFIG)
        assert d.action == "block"
        assert d.reason_code == "SUBSCRIPTION_MIN_COMMITMENT"

    def test_cancel_subscription_after_commitment(self):
        order = {
            "order_id": "ORD-MM-1008",
            "customer_email": "frank@example.com",
            "category": "subscription",
            "amount": 499,
            "status": "active",
            "order_date": "2026-08-01",
        }
        ctx = _ctx(
            "megamart",
            "cancel_order",
            entities={"order_id": "ORD-MM-1008", "email": "frank@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "allow"
        assert d.reason_code == "CANCEL_ALLOWED"

    def test_cancel_no_order(self):
        ctx = _ctx(
            "megamart",
            "cancel_order",
            entities={"order_id": "ORD-MM-FAKE"},
            customer_email="nobody@example.com",
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "clarify"
        assert d.reason_code == "VERIFICATION_REQUIRED"


# ── Layer 4: default deny ────────────────────────────────────────────


class TestDefaultDeny:
    def test_unknown_intent_escalates(self):
        ctx = _ctx(
            "megamart",
            "faq",
            confidence=0.95,
        )
        config_no_faq = {**MEGAMART_CONFIG}
        d = evaluate_policy(ctx, tenant_config=config_no_faq)
        assert d.action == "allow"


# ── Category override tests ──────────────────────────────────────────


class TestCategoryOverrides:
    def test_electronics_30_day_window(self):
        order = {
            "order_id": "ORD-X",
            "customer_email": "x@example.com",
            "category": "electronics",
            "amount": 5000,
            "status": "delivered",
            "delivery_date": "2026-09-10",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-X", "email": "x@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "allow"

    def test_clothing_7_day_window_expired(self):
        order = {
            "order_id": "ORD-X",
            "customer_email": "x@example.com",
            "category": "clothing",
            "amount": 800,
            "status": "delivered",
            "delivery_date": "2026-09-20",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-X", "email": "x@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "block"
        assert d.reason_code == "RETURN_WINDOW_EXPIRED"

    def test_electronics_uses_unopened_condition(self):
        order = {
            "order_id": "ORD-X",
            "customer_email": "x@example.com",
            "category": "electronics",
            "amount": 3000,
            "status": "delivered",
            "delivery_date": "2026-09-25",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-X", "email": "x@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.action == "allow"
        assert d.constraints["condition"] == "unopened"

    def test_default_category_uses_any_condition(self):
        order = {
            "order_id": "ORD-X",
            "customer_email": "x@example.com",
            "category": "home_decor",
            "amount": 2000,
            "status": "delivered",
            "delivery_date": "2026-09-25",
        }
        ctx = _ctx(
            "megamart",
            "refund",
            entities={"order_id": "ORD-X", "email": "x@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=MEGAMART_CONFIG)
        assert d.constraints["condition"] == "any"


# ── Cross-tenant tests ───────────────────────────────────────────────


class TestCrossTenant:
    def test_megamart_faq_vs_tiny_candles_faq(self):
        for tid, cfg in [("megamart", MEGAMART_CONFIG), ("tiny-candles", TINY_CANDLES_CONFIG)]:
            ctx = _ctx(tid, "faq")
            d = evaluate_policy(ctx, tenant_config=cfg)
            assert d.action == "allow"

    def test_tiny_candles_refund_always_escalates(self):
        order = {
            "order_id": "ORD-TC-2005",
            "customer_email": "grace@example.com",
            "category": "candles",
            "amount": 2100,
            "status": "delivered",
            "delivery_date": "2026-09-28",
        }
        ctx = _ctx(
            "tiny-candles",
            "refund",
            entities={"order_id": "ORD-TC-2005", "email": "grace@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=TINY_CANDLES_CONFIG)
        assert d.action == "escalate"
        assert d.reason_code == "TENANT_REFUND_REVIEW_REQUIRED"

    def test_tiny_candles_cancel_processing_allowed(self):
        order = {
            "order_id": "ORD-TC-2004",
            "customer_email": "jack@example.com",
            "category": "candles",
            "amount": 750,
            "status": "processing",
        }
        ctx = _ctx(
            "tiny-candles",
            "cancel_order",
            entities={"order_id": "ORD-TC-2004", "email": "jack@example.com"},
            order=order,
        )
        d = evaluate_policy(ctx, tenant_config=TINY_CANDLES_CONFIG)
        assert d.action == "allow"
        assert d.reason_code == "CANCEL_ALLOWED"
