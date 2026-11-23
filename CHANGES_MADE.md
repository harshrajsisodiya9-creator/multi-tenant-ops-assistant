# Changes Made — `real_db` branch

Branch: `real_db` | Base commit: `d25c682` | Date: 2026-11-23

Security hardening (Steps 0–6) + production database integration for the multi-tenant AI customer support assistant.

---

## Step 0 — Security Audit

Identified gaps across identity verification, data scoping, and write safety. No code changes — produced the plan that drove Steps 1–6.

---

## Step 1a — JWT Authentication

### Problem
The API accepted `customer_email` in the request body — anyone could impersonate any customer. The API key proved the tenant, but nothing proved the customer.

### Files changed
- **app/auth.py** — New per-tenant HS256 JWT auth system
  - `_decode_token`: two-pass JWT verification (unverified read for tenant_id → load per-tenant secret → full decode with signature, expiry, audience, issuer)
  - `_get_jwt_secret` / `_env_key_for_tenant`: per-tenant signing secrets from env vars (`JWT_SECRET_<TENANT_ID>`)
  - `_lookup_customer`: DB lookup by (tenant_id, customer_id) after token verify — catches deleted accounts, derives email server-side
  - `get_authenticated_customer`: full chain: HTTPBearer → decode → tenant cross-check → DB lookup → TokenClaims
  - `_bearer_scheme = HTTPBearer(auto_error=False)`: returns None instead of 422 when header missing, explicit 401 raise
  - `TokenClaims`: frozen dataclass carrying verified identity through the request

- **app/models.py** — New models
  - `Customer` model with (customer_id, tenant_id, email, name)
  - `RefundHistory` model for duplicate refund detection
  - NOT NULL `customer_id` FK on `Order`

- **app/main.py** — Endpoint security
  - JWT-authenticated `/agent` and `/agent/resume` endpoints
  - `_THREAD_OWNERS` dict mapping thread_id → (tenant_id, customer_id) for thread hijack prevention (403 on mismatch)
  - `/dev/token` endpoint gated behind `APP_ENV != prod`
  - Server-derived email from DB record, not from request body

- **requirements.txt** — Added `PyJWT`

### Fixes applied
- NOT NULL on customer_id FK — stricter is correct when seed rebuilds everything (user insisted, correct call)
- HTTPBearer(auto_error=False) — FastAPI returns 422 for missing required headers; this returns proper 401
- DB lookup after token verify — valid JWT signature doesn't mean customer still exists

---

## Step 2 — Owner-Scoped Queries (ToolContext)

### Problem
The LLM agent calls tools like `get_order_details(order_id)`. If Customer A knows Customer B's order ID, the tool returned it. The LLM picks arguments, so the user controls what gets queried through prompt injection.

### Files changed
- **app/graph/agent_prod/tool_context.py** *(new)* — `ToolContext` frozen dataclass carrying (tenant_id, customer_id), created from JWT claims and passed to every tool function

- **app/graph/agent_prod/tools/order_tools.py** — All queries owner-scoped
  - `_find_order`: WHERE tenant_id AND customer_id (not just order_id)
  - `_find_item`: JOIN through Order to inherit tenant+customer scope (previously had NO tenant filter)
  - Every tool function takes `ctx: ToolContext` as first argument

- **app/graph/agent_prod/tools/__init__.py** — `list_my_orders` replacing `get_orders_by_email`; updated TOOL_REGISTRY and TOOL_DESCRIPTIONS

- **app/graph/agent_prod/react_agent.py** — Identity hidden from LLM
  - `_IDENTITY_PARAMS = frozenset({"tenant_id", "customer_id", "customer_email", "ctx"})` — stripped from LangChain tool schemas automatically
  - `_build_langchain_tools`: skips identity params in Pydantic model generation
  - `run_react_agent`: injects ctx when calling tools, invisible to LLM

- **app/graph/data_access.py** *(new)* — Factory module routing data access
  - Branches on `setting.agent_mode`: "prod" runs SQLAlchemy queries scoped by tenant_id + customer_id; else delegates to mock_data
  - `get_tenant_info`, `get_customer_orders`, `get_order_scoped`, `get_refund_count_30d`

- **app/graph/mock_data.py** — New customer_id-based lookup functions
  - `MOCK_CUSTOMERS` dict mapping customer_id → customer info
  - `get_orders_by_customer(tenant_id, customer_id)` replacing email-based lookup
  - `get_order_scoped(tenant_id, customer_id, order_id)`
  - `get_refund_count_30d_by_customer(tenant_id, customer_id)`

- **app/graph/nodes/pre_check.py** — Imports from `data_access` instead of `mock_data`
- **app/graph/nodes/intent_executor.py** — Uses `get_order_scoped` from `data_access`; customer_id-based execution
- **app/graph/nodes/policy_fan_out.py** — customer_id migration, data_access imports
- **app/graph/policy_gate.py** — Removed gate-level email comparison (owner scoping handles it at DB layer now)
- **app/graph/state.py** — Added `customer_id` to AgentState TypedDict

### Fixes applied
- `_find_item` JOIN through Order — previously had no tenant filter at all, could leak cross-tenant item data
- Removed `get_orders_by_email` — LLM could pass any email to look up orders; `list_my_orders(ctx)` is zero-argument from LLM perspective

---

## Step 3 — Order Lock + Per-Intent Tool Allowlist

### Problem
(a) The LLM could hallucinate or be injected into calling tools with a different order_id than assigned.
(b) A refund intent could call cancel_order, and vice versa — no tool-level intent enforcement.

### Files changed
- **app/graph/agent_prod/react_agent.py**
  - Order lock: pre-call comparison of `tool_args["order_id"]` against assigned `order_id` → BLOCKED ToolMessage on mismatch
  - `INTENT_TOOLS` mapping: each intent → frozenset of allowed tool names
  - Two-layer enforcement: `bind_tools` filters to intent subset AND explicit allowlist check before execution
  - `WRITE_TOOLS` frozenset identifying mutation tools

---

## Step 4 — Safe Writes (FOR UPDATE, Duplicate Check, Policy Re-Run)

### Problem
Between the policy gate's decision (time T1) and the tool's execution (time T2), state can change — another request could refund the same order, the order status could change, or the return window could expire. TOCTOU vulnerability.

### Files changed
- **app/graph/agent_prod/tools/order_tools.py**
  - `_find_order(for_update=True)`: SELECT ... FOR UPDATE row lock (skipped on SQLite via `db.bind.dialect.name != "sqlite"`)
  - `initiate_refund`: FOR UPDATE + RefundHistory duplicate check + `_check_return_policy` re-run
  - `cancel_order`: FOR UPDATE + status check (only pending/processing/confirmed)
  - `initiate_return`: FOR UPDATE + `_check_return_policy` re-run
  - `get_shipping_label`: FOR UPDATE + `_check_return_policy` re-run
  - `_check_return_policy` helper: re-runs return policy inside the write tool's own transaction

### Fixes applied
- SQLite dialect check for FOR UPDATE — SQLite doesn't support it, tests use SQLite in-memory DBs
- `get_shipping_label` missing policy re-check — caught by TestWriteToolRegistry, added `_check_return_policy` call

---

## Step 5 — Factory Guard

### Problem
If APP_ENV=prod but AGENT_MODE=mock (forgot to set it), customers hit hardcoded mock data — wrong answers, no real orders.

### Files changed
- **app/config.py** — After `setting = Setting()`: if `app_env == "prod"` and `agent_mode != "prod"`, raise `RuntimeError` immediately. Process crashes before serving any traffic.

### Fixes applied
- `test_dev_token_absent_in_prod` broke — it set APP_ENV=prod + AGENT_MODE=mock, triggering the new guard. Fixed by changing that test to AGENT_MODE=prod (the test's purpose is verifying dev token absence, not agent mode)

---

## Step 6 — Hardening Tests

### Files changed
- **tests/test_hardening.py** *(new)* — 8 tests covering:
  1. `TestOrderLock` — tool call with different order_id blocked
  2. `TestAllowlist` — cancel_order blocked in refund intent
  3. `TestDoubleRefund` — second refund on same order rejected (real SQLite DB)
  4. `TestCancelShipped` — cancel shipped order rejected
  5. `TestForUpdate` — `_find_order` accepts `for_update=True` and returns order
  6. `TestPolicyReRun` — category changed between gate and write → policy re-run catches it
  7. `TestWriteToolRegistry` — every tool in WRITE_TOOLS has FOR UPDATE + (policy check or duplicate check or status check)
  8. `TestFactoryGuard` — APP_ENV=prod + AGENT_MODE=mock → RuntimeError

- **tests/test_jwt_auth.py** *(new)* — 25 tests:
  - JWT decode, tenant cross-check, expiry, audience/issuer validation
  - HTTPBearer 401 on missing/invalid token
  - Dev token gating
  - 4 prod-mode entity resolution tests (product name → DB order, cross-customer isolation, mock order_id not found in prod, gate reads facts from DB row)

---

## Test Suite Migration (customer_email → customer_id)

All test files migrated from `customer_email` to `customer_id` state keys.

### Files changed
- **tests/test_intent_executor.py** — customer_id params, correct order-to-customer mappings from mock data
- **tests/test_policy_fan_out.py** — customer_id state keys replacing customer_email
- **tests/test_policy_gate.py** — `test_identity_mismatch` → `test_wrong_customer_owner_scoped_no_order` (order=None, VERIFICATION_REQUIRED)
- **tests/test_prompt_injection.py** — `_ctx()` helper migrated, IDENTITY_MISMATCH tests updated for owner scoping

### Fixes applied
- Stale `get_order` references in `intent_executor.py` — `_execute_refund`/`_execute_cancel` used undefined function; migrated to `get_order_scoped`
- Module reference staleness after `importlib.reload` — `data_access.py` held stale reference to old `setting` object; fixed with `patch.object(_da.setting, "agent_mode", "prod")`
- Cross-customer entity resolution test false auto-resolve — Customer A had only 1 order, so "only 1 order" fallback auto-resolved; fixed by giving Customer A 2 orders
- Missing RefundHistory table in prod entity resolution fixture — `_build_gate_context` calls `get_refund_count_30d`; added table to fixture creation

---

## Other New Files

- **app/graph/agent_prod/react_subgraph.py** *(new)* — LangGraph subgraph wrapper for the react agent
- **app/graph/agent_prod/tools/seed_db.py** — Seed data with Customer rows, customer_id on orders
- **app/graph/builder.py** — Minor wiring updates for subgraph
- **.env.example** *(new)* — Example environment variables
- **frontend/** *(new)* — Frontend directory
