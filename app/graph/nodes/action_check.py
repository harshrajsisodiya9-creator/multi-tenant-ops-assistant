from __future__ import annotations

from datetime import date, timedelta
from uuid import UUID

from app.graph.policy_gate import GLOBAL_REFUND_CEILING
from app.graph.state import AgentState

NEAR_CEILING_THRESHOLD = 0.8
RECENT_DELIVERY_DAYS = 1


# ---------------------------------------------------------------------------
# Verify: confirm write operations actually landed in the DB
# ---------------------------------------------------------------------------

def _verify_refund(tenant_id: str, customer_id: str, order_id: str) -> dict:
    from sqlalchemy import select

    from app.database import SessionLocal
    from app.models import RefundHistory

    db = SessionLocal()
    try:
        stmt = select(RefundHistory).where(
            RefundHistory.tenant_id == UUID(tenant_id),
            RefundHistory.customer_id == customer_id,
            RefundHistory.order_id == order_id,
        )
        record = db.execute(stmt).scalar_one_or_none()
        if record:
            return {"verified": True, "amount": record.amount, "date": str(record.refund_date)}
        return {"verified": False, "reason": "no RefundHistory record found"}
    finally:
        db.close()


def _verify_cancel(tenant_id: str, order_id: str) -> dict:
    from sqlalchemy import select

    from app.database import SessionLocal
    from app.models import Order

    db = SessionLocal()
    try:
        stmt = select(Order).where(
            Order.tenant_id == UUID(tenant_id),
            Order.order_id == order_id,
        )
        order = db.execute(stmt).scalar_one_or_none()
        if not order:
            return {"verified": False, "reason": "order not found"}
        if order.status == "cancelled":
            return {"verified": True, "status": order.status}
        return {"verified": False, "reason": f"order status is '{order.status}', expected 'cancelled'"}
    finally:
        db.close()


VERIFIERS = {
    "initiate_refund": lambda tid, cid, tc: _verify_refund(tid, cid, tc.get("args", {}).get("order_id", "")),
    "cancel_order": lambda tid, cid, tc: _verify_cancel(tid, tc.get("args", {}).get("order_id", "")),
}


# ---------------------------------------------------------------------------
# Audit: flag anomalies worth a human glance
# ---------------------------------------------------------------------------

def _audit_tool_calls(tool_calls: list[dict], tenant_id: str, order_id: str) -> list[dict]:
    flags: list[dict] = []

    write_count = sum(1 for tc in tool_calls if tc.get("tool") in VERIFIERS and not tc.get("blocked"))
    if write_count > 1:
        flags.append({
            "flag": "MULTI_WRITE",
            "detail": f"{write_count} write operations in a single request",
        })

    for tc in tool_calls:
        if tc.get("blocked"):
            continue
        tool = tc.get("tool", "")
        args = tc.get("args", {})

        if tool == "initiate_refund":
            amount = args.get("amount", 0)
            if amount > GLOBAL_REFUND_CEILING * NEAR_CEILING_THRESHOLD:
                flags.append({
                    "flag": "NEAR_CEILING",
                    "detail": f"refund ₹{amount:,} is >{NEAR_CEILING_THRESHOLD:.0%} of ceiling ₹{GLOBAL_REFUND_CEILING:,}",
                })

            oid = args.get("order_id", "")
            delivery = _get_delivery_date(tenant_id, oid)
            if delivery and (date.today() - delivery) < timedelta(days=RECENT_DELIVERY_DAYS):
                flags.append({
                    "flag": "RECENT_DELIVERY_REFUND",
                    "detail": f"refund on order delivered {delivery.isoformat()} (< {RECENT_DELIVERY_DAYS}d ago)",
                })

        if tool == "cancel_order":
            oid = args.get("order_id", "")
            shipping_status = _get_shipping_status(tenant_id, oid)
            if shipping_status in ("shipped", "in_transit"):
                flags.append({
                    "flag": "CANCEL_AFTER_SHIP",
                    "detail": f"cancel requested on order with shipping status '{shipping_status}'",
                })

    return flags


def _get_delivery_date(tenant_id: str, order_id: str) -> date | None:
    from app.config import setting
    if setting.agent_mode != "prod":
        return None

    from sqlalchemy import select

    from app.database import SessionLocal
    from app.models import Order

    db = SessionLocal()
    try:
        stmt = select(Order.delivery_date).where(
            Order.tenant_id == UUID(tenant_id),
            Order.order_id == order_id,
        )
        return db.execute(stmt).scalar_one_or_none()
    finally:
        db.close()


def _get_shipping_status(tenant_id: str, order_id: str) -> str | None:
    from app.config import setting
    if setting.agent_mode != "prod":
        return None

    from sqlalchemy import select

    from app.database import SessionLocal
    from app.models import Order

    db = SessionLocal()
    try:
        stmt = select(Order.status).where(
            Order.tenant_id == UUID(tenant_id),
            Order.order_id == order_id,
        )
        return db.execute(stmt).scalar_one_or_none()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def action_check_node(state: AgentState) -> dict:
    from app.config import setting

    intent_results: list[dict] = list(state.get("intent_results") or [])
    tenant_id = state.get("tenant_id", "")
    customer_id = state.get("customer_id", "")
    is_prod = setting.agent_mode == "prod"

    for result in intent_results:
        if result.get("status") != "fulfilled":
            continue

        tool_calls = result.get("tool_calls", [])
        if not tool_calls:
            continue

        # --- Verify (prod only) ---
        if is_prod:
            for tc in tool_calls:
                if tc.get("blocked"):
                    continue
                tool_name = tc.get("tool", "")
                verifier = VERIFIERS.get(tool_name)
                if not verifier:
                    continue

                verification = verifier(tenant_id, customer_id, tc)
                tc["verification"] = verification

                if not verification.get("verified"):
                    result["status"] = "failed"
                    result["domain_result"] = (
                        f"We attempted to process your request but it didn't complete "
                        f"successfully. Please try again or contact support."
                    )
                    result["failure_reason"] = verification.get("reason", "unknown")
                    break

        # --- Audit (always) ---
        audit_flags = _audit_tool_calls(
            tool_calls, tenant_id,
            result.get("entities", {}).get("order_id", ""),
        )
        if audit_flags:
            result["audit_flags"] = audit_flags

    return {"intent_results": intent_results}
