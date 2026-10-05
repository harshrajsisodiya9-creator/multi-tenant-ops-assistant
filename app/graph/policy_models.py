from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReturnPolicy:
    allowed: bool = True
    window_days: int = 7
    max_amount: float | None = None
    condition: str = "any"  # "unopened", "defective", "any"
    refund_type: str = "original_method"  # "store_credit", "original_method"


@dataclass
class ExchangePolicy:
    allowed: bool = True
    window_days: int = 15
    same_category_only: bool = False


@dataclass
class SubscriptionPolicy:
    cancel_allowed: bool = True
    cancel_notice_days: int = 0
    prorated_refund: bool = False
    min_commitment_months: int = 0


@dataclass
class PolicyRule:
    return_policy: ReturnPolicy = field(default_factory=ReturnPolicy)
    exchange_policy: ExchangePolicy = field(default_factory=ExchangePolicy)
    subscription_policy: SubscriptionPolicy | None = None
