from __future__ import annotations

from pathlib import Path

import yaml

from app.graph.policy_models import (
    ExchangePolicy,
    PolicyRule,
    ReturnPolicy,
    SubscriptionPolicy,
)

POLICIES_DIR = Path(__file__).resolve().parent.parent / "policies"

_cache: dict[str, dict] = {}


def _load_tenant_config(tenant_id: str) -> dict:
    if tenant_id in _cache:
        return _cache[tenant_id]

    path = POLICIES_DIR / f"{tenant_id}.yaml"
    if not path.exists():
        return {}

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    _cache[tenant_id] = config
    return config


def _dict_to_return_policy(d: dict) -> ReturnPolicy:
    valid = {k: v for k, v in d.items() if k in ReturnPolicy.__dataclass_fields__}
    return ReturnPolicy(**valid)


def _dict_to_exchange_policy(d: dict) -> ExchangePolicy:
    valid = {k: v for k, v in d.items() if k in ExchangePolicy.__dataclass_fields__}
    return ExchangePolicy(**valid)


def _merge_return_policy(default: ReturnPolicy, override: dict) -> ReturnPolicy:
    merged = {k: getattr(default, k) for k in ReturnPolicy.__dataclass_fields__}
    merged.update({k: v for k, v in override.items() if k in ReturnPolicy.__dataclass_fields__})
    return ReturnPolicy(**merged)


def _merge_exchange_policy(default: ExchangePolicy, override: dict) -> ExchangePolicy:
    merged = {k: getattr(default, k) for k in ExchangePolicy.__dataclass_fields__}
    merged.update({k: v for k, v in override.items() if k in ExchangePolicy.__dataclass_fields__})
    return ExchangePolicy(**merged)


def resolve_policy(tenant_id: str, category: str | None = None, *, tenant_config: dict | None = None) -> PolicyRule:
    config = tenant_config if tenant_config is not None else _load_tenant_config(tenant_id)
    if not config:
        return PolicyRule()

    default_cfg = config.get("default", {})
    default_return = _dict_to_return_policy(default_cfg.get("return_policy", {}))
    default_exchange = _dict_to_exchange_policy(default_cfg.get("exchange_policy", {}))

    sub_cfg = config.get("subscription")
    default_sub = SubscriptionPolicy(**sub_cfg) if sub_cfg else None

    if category:
        cat_cfg = config.get("categories", {}).get(category, {})
        if cat_cfg:
            ret_override = cat_cfg.get("return_policy", {})
            exc_override = cat_cfg.get("exchange_policy", {})
            merged_return = _merge_return_policy(default_return, ret_override) if ret_override else default_return
            merged_exchange = _merge_exchange_policy(default_exchange, exc_override) if exc_override else default_exchange

            cat_sub_cfg = cat_cfg.get("subscription_policy")
            merged_sub = SubscriptionPolicy(**cat_sub_cfg) if cat_sub_cfg else default_sub

            return PolicyRule(
                return_policy=merged_return,
                exchange_policy=merged_exchange,
                subscription_policy=merged_sub,
            )

    return PolicyRule(
        return_policy=default_return,
        exchange_policy=default_exchange,
        subscription_policy=default_sub,
    )


def get_tenant_config(tenant_id: str, *, config_override: dict | None = None) -> dict:
    if config_override is not None:
        return config_override
    return _load_tenant_config(tenant_id)


def clear_cache() -> None:
    _cache.clear()
