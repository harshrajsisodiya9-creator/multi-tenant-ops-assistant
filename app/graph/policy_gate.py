from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from app.graph.policy_loader import get_tenant_config, resolve_policy
from app.graph.schemas import PolicyDecision, RouterOutput

GLOBAL_REFUND_CEILING = 50_000
ORDER_INTENTS = {"order_status", "refund", "cancel_order"}
CANCELLABLE_STATUSES = {"pending", "processing", "confirmed"}

# Compound actions are evaluated independently; no rollback across actions.
# Production would wrap these in a transaction.


@dataclass
class GateContext:
    tenant_id: str
    tenant_active: bool
    router_output: RouterOutput
    customer_id: str = ""
    order: dict | None = None
    refund_count_30d: int = 0
    today: date = field(default_factory=date.today)


def evaluate_policy(
    ctx: GateContext, *, tenant_config: dict | None = None
) -> PolicyDecision:
    ro = ctx.router_output
    config = get_tenant_config(ctx.tenant_id, config_override=tenant_config)

    # ── Layer 1: global hard rules (all tenants) ─────────────────────
    if not ctx.tenant_active:
        return PolicyDecision(action="block", reason_code="TENANT_INACTIVE")

    if ro.confidence < 0.6:
        return PolicyDecision(action="clarify", reason_code="LOW_CONFIDENCE")

    if ro.intent == "out_of_scope":
        return PolicyDecision(action="block", reason_code="OUT_OF_SCOPE")

    if ro.intent in ORDER_INTENTS:
        has_order_id = bool(ro.entities.get("order_id"))
        has_customer = bool(ctx.customer_id)
        if not has_order_id and not has_customer:
            return PolicyDecision(action="clarify", reason_code="VERIFICATION_REQUIRED")

    # ── Layer 2: global safety ceiling ───────────────────────────────
    # Moved to _evaluate_refund so the allow-with-approval path
    # can include return-policy constraints for the agent.

    # ── Layer 3: tenant intent rules ─────────────────────────────────
    if ro.intent == "faq":
        return PolicyDecision(action="allow", reason_code="FAQ_ALLOWED")

    if ro.intent == "order_status":
        return PolicyDecision(
            action="allow",
            reason_code="ORDER_STATUS_ALLOWED",
            constraints={"allowed_tools": ["get_order_status"]},
        )

    if ro.intent == "complaint":
        if ro.escalation_signals:
            return PolicyDecision(
                action="escalate", reason_code="ESCALATION_KEYWORDS_DETECTED"
            )
        return PolicyDecision(action="allow", reason_code="COMPLAINT_ALLOWED")

    if ro.intent == "refund":
        return _evaluate_refund(ctx, config)

    if ro.intent == "cancel_order":
        return _evaluate_cancel(ctx, config)

    # ── Layer 4: default deny ────────────────────────────────────────
    return PolicyDecision(action="escalate", reason_code="DEFAULT_DENY")


def _evaluate_refund(ctx: GateContext, config: dict) -> PolicyDecision:
    if not ctx.order:
        if config.get("refund_always_escalate", False) and ctx.customer_id:
            return PolicyDecision(
                action="allow",
                reason_code="REFUND_NEEDS_APPROVAL",
                constraints={
                    "needs_approval": True,
                    "approval_reasons": ["TENANT_REFUND_REVIEW_REQUIRED"],
                },
            )
        return PolicyDecision(action="clarify", reason_code="VERIFICATION_REQUIRED")

    category = ctx.order.get("category")
    policy = resolve_policy(ctx.tenant_id, category, tenant_config=config)

    if not policy.return_policy.allowed:
        return PolicyDecision(action="block", reason_code="RETURNS_NOT_ALLOWED")

    delivery_date_str = ctx.order.get("delivery_date")
    if delivery_date_str:
        delivery_date = date.fromisoformat(delivery_date_str)
        days_since = (ctx.today - delivery_date).days
        if days_since > policy.return_policy.window_days:
            return PolicyDecision(action="block", reason_code="RETURN_WINDOW_EXPIRED")

    base_constraints = {
        "refund_type": policy.return_policy.refund_type,
        "condition": policy.return_policy.condition,
    }

    approval_reasons: list[str] = []
    amount = ctx.order.get("amount", 0)

    if config.get("refund_always_escalate", False):
        approval_reasons.append("TENANT_REFUND_REVIEW_REQUIRED")

    if amount > GLOBAL_REFUND_CEILING:
        approval_reasons.append("GLOBAL_REFUND_CEILING")

    max_auto = config.get("max_auto_refund")
    if max_auto is not None and amount > max_auto:
        approval_reasons.append("REFUND_OVER_LIMIT")

    max_refunds = config.get("max_refunds_per_customer_30d", 999)
    if ctx.refund_count_30d >= max_refunds:
        approval_reasons.append("REFUND_RATE_LIMITED")

    if (
        policy.return_policy.max_amount
        and amount > policy.return_policy.max_amount
        and "REFUND_OVER_LIMIT" not in approval_reasons
    ):
        approval_reasons.append("REFUND_OVER_LIMIT")

    if approval_reasons:
        return PolicyDecision(
            action="allow",
            reason_code="REFUND_NEEDS_APPROVAL",
            constraints={
                **base_constraints,
                "needs_approval": True,
                "approval_reasons": approval_reasons,
            },
        )

    return PolicyDecision(
        action="allow",
        reason_code="REFUND_ALLOWED",
        constraints=base_constraints,
    )


def _evaluate_cancel(ctx: GateContext, config: dict) -> PolicyDecision:
    if not ctx.order:
        return PolicyDecision(action="clarify", reason_code="VERIFICATION_REQUIRED")

    category = ctx.order.get("category", "")
    order_status = ctx.order.get("status", "").lower()

    if category in ("subscription", "subscription_box"):
        policy = resolve_policy(ctx.tenant_id, category, tenant_config=config)
        if policy.subscription_policy:
            sub = policy.subscription_policy
            if not sub.cancel_allowed:
                return PolicyDecision(
                    action="block", reason_code="SUBSCRIPTION_CANCEL_NOT_ALLOWED"
                )
            if sub.min_commitment_months > 0:
                start_str = ctx.order.get("order_date") or ctx.order.get("start_date")
                if start_str:
                    start = date.fromisoformat(start_str)
                    months = (ctx.today.year - start.year) * 12 + (
                        ctx.today.month - start.month
                    )
                    if ctx.today.day < start.day:
                        months -= 1
                    if months < sub.min_commitment_months:
                        return PolicyDecision(
                            action="block",
                            reason_code="SUBSCRIPTION_MIN_COMMITMENT",
                        )
        return PolicyDecision(action="allow", reason_code="CANCEL_ALLOWED")

    if order_status not in CANCELLABLE_STATUSES:
        return PolicyDecision(action="block", reason_code="ORDER_NOT_CANCELLABLE")

    return PolicyDecision(action="allow", reason_code="CANCEL_ALLOWED")
