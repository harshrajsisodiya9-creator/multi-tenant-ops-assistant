from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolContext:
    tenant_id: str
    customer_id: str
