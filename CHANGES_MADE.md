# Changes Made — `real_db` / `conditional` branches

Branch: `conditional` (from `real_db`) | Base commit: `d25c682` | Date: 2026-11-23

Security hardening (Steps 0–6) + production database integration + conditional edge routing for the multi-tenant AI customer support assistant.

---

## Conditional Edge Addition

### Problem
The LangGraph pipeline was fully linear — every request traversed all 8 nodes regardless of intent type. Only one conditional edge existed (after `escalation_handler`). Routing functions lacked `Literal` type hints, making LangSmith traces opaque. Multi-intent requests were processed sequentially inside a monolithic `intent_executor`.

### Solution
Added `Send()`-based parallel fan-out per intent with proper conditional edges at every decision point, so traces show which path each intent takes and multi-intent requests execute in parallel.

### New Graph Topology

```
START → router → pre_check → [after_pre_check: bail/continue]
  bail → bail_executor → response_agent → END
  continue → policy_fan_out → [route_intents: Send() per intent]
    → agent_executor / faq_executor / simple_executor (parallel)
    → merge_results → [after_merge: needs_action_check/skip_action_check]
      → action_check (if needed) → response_agent
      → [after_response: complete/needs_escalation]
        → END or escalation_handler → [after_escalation: follow_up/end]
```

### Files changed

- **app/graph/state.py** — Added 3 new fields
  - `current_intent: dict` — set by `Send()`, the specific intent for each executor
  - `processed_results: Annotated[list[dict], operator.add]` — reducer merges parallel branch outputs
  - `has_tool_calls: bool` — computed by `merge_results` for downstream routing

- **app/graph/nodes/intent_executor.py** — Split monolithic executor into 5 nodes
  - `agent_executor_node` — handles allow + refund/cancel_order via react agent subgraph; now handles `pending_approval` from the subgraph — sets status to `"awaiting_human"` with pending tool call details when approval is needed
  - `faq_executor_node` — handles allow + faq/complaint via RAG hybrid search
  - `simple_executor_node` — handles conversational, order_status, block, escalate, clarify
  - `bail_executor_node` — handles pre_check bail results (tenant-inactive, low-confidence)
  - `merge_results_node` — combines parallel `processed_results` into `intent_results`, computes `has_tool_calls` and `response_phase`
  - `_run_react_agent` now returns full `dict` (was `tuple[str, list[dict]]`) to carry `pending_approval` through

- **app/graph/builder.py** — Full rewrite with Send() + 5 conditional edges
  - `after_pre_check` → `Literal["bail", "continue"]` — routes bail to `bail_executor`, else to `policy_fan_out`
  - `route_intents` → returns `list[Send()]` — dispatches each pending intent to `agent_executor`, `faq_executor`, or `simple_executor`
  - `after_merge` → `Literal["needs_action_check", "skip_action_check"]` — skips `action_check` when no tool calls
  - `after_response_agent` → `Literal["complete", "needs_escalation"]` — routes to END or `escalation_handler`
  - `after_escalation` → `Literal["follow_up", "end"]` — routes to `follow_up_response` or END

- **app/graph/agent_prod/react_subgraph.py** — HITL approval gate + Literal type hints
  - `_action_check_in_loop` renamed to `_hard_constraint_check` — only blocks hard violations (wrong refund type, cancel_blocked)
  - New `_needs_approval(tool_name, tool_args, constraints)` — soft violations that pause for human approval:
    - `constraints.needs_approval` flag with custom `approval_reasons`
    - Refund amount > `GLOBAL_REFUND_CEILING` → `["GLOBAL_REFUND_CEILING"]`
  - `tool_execute_node` calls `_needs_approval` after hard check — returns early with `pending_approval` dict and `PENDING APPROVAL` ToolMessage, halting the agent loop
  - `ReactAgentState` gains `pending_approval: dict` field
  - `should_continue` checks `pending_approval` → exits with `"done"`
  - Return type: `Literal["done", "max_iterations", "continue_tools"]` with 3-way path_map

- **app/graph/nodes/action_check.py** — Rewritten from post-hoc blocker to audit + verify node
  - **Verify (prod only):** after write tools execute, re-reads DB to confirm state changes landed (refund record exists, order status changed to cancelled). Marks intent as `"failed"` if verification fails — prevents false success responses.
  - **Audit (always):** flags anomalies on fulfilled tool calls:
    - `NEAR_CEILING` — refund amount > 80% of global ceiling
    - `MULTI_WRITE` — multiple write operations in a single request
    - `RECENT_DELIVERY_REFUND` — refund on order delivered < 1 day ago
    - `CANCEL_AFTER_SHIP` — cancel on order with shipped/in_transit status
  - Flags are informational only — attached to `intent_results` as `audit_flags` for traces and response_agent visibility. No blocking (the in-loop `_action_check_in_loop` in react_subgraph handles that pre-execution).

- **tests/conftest.py** *(new)* — Auto-fixture patching `AGENT_MODE=mock` for all tests
  - Patches both the env var and the live `setting` object, so tests survive `importlib.reload(app.config)`
  - Prod-mode tests (test_jwt_auth, test_hardening) are unaffected — they set up their own SQLite DBs directly

- **tests/test_action_check.py** — Rewritten for audit + verify behavior
  - `TestAuditFlags` — NEAR_CEILING, MULTI_WRITE, blocked-tools-skipped
  - `TestPassthrough` — no tool calls, non-fulfilled skipped, clean cancel, multi-intent flag isolation
  - `TestVerifySkippedInMock` — no verification dict in mock mode

- **tests/test_intent_executor.py** — Updated for split executor nodes
  - `TestSimpleExecutorNode` — order_status, block, escalate, clarify via `simple_executor_node`
  - `TestMergeResultsNode` — mixed statuses, response_phase, has_tool_calls
  - `TestBailExecutorNode` — bail block, skips non-pending

- **app/graph/nodes/\_\_init\_\_.py** — Updated exports for split executor nodes

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

---

## HITL Lifecycle, Thread Ownership and Concurrency

Branch: `change` | Base commit: `7968fde` | Date: 2026-11-26

### Problem
- A reviewer decision with index `-1`, `True` or `False` passed validation (Python allows negative indexes, and `bool` is an `int`), so a malformed decision could approve the wrong pending operation.
- `processed_results` used `operator.add` and was never cleared. A second message on the same `thread_id` merged the previous turn's results, so old answers repeated and an already-approved escalation could interrupt again and refund a second time.
- A new `/agent` message on a thread paused for approval started a fresh turn and silently abandoned the pending approval.
- An unknown client-supplied `thread_id` was claimed by whoever sent it first. Error messages ("belongs to a different customer/tenant") revealed that a foreign thread existed.
- Nothing serialized requests on the same thread: the pending-approval check and the graph invocation could race.
- `/agent/resume` on an idle or completed thread ran no nodes but returned the last turn's `intent_results` as if the decision had been applied.

### Files changed
- **app/graph/nodes/escalation_handler.py**: the decision index must be a real non-negative `int` in range. Booleans, floats, strings, `None` and negatives are skipped and never execute a tool.
- **app/graph/state.py**: `processed_results` uses `reduce_processed_results`. A `None` update clears the list, and list updates append, so parallel `Send()` branches still merge.
- **app/graph/nodes/before_router.py**: at the start of every new turn, resets `processed_results` (to `None`), `intent_results`, `router_output`, `policy_decision`, `has_tool_calls`, `response_phase`, `pre_check_bail` and `response`. Conversation memory (history, customer/orders context, turn summaries) and request identity are untouched. `/agent/resume` resumes inside `escalation_handler` and never runs this node.
- **app/main.py**:
  - **Server-issued thread IDs:** `_create_thread` generates a `uuid4().hex` and registers its owner atomically (`_register_thread` under `_THREAD_OWNERS_GUARD`). A client-supplied `thread_id` must already be registered to the caller's `(tenant_id, customer_id)`. Unknown and foreign threads both get the same `403 THREAD_NOT_ACCESSIBLE` response.
  - **Per-thread lock:** `_thread_lock(thread_id)` keeps a refcounted registry of `threading.Lock` objects (`_THREAD_LOCKS`, guarded by `_THREAD_LOCKS_GUARD`), one lock per thread ID. An entry is removed only when its last holder or waiter leaves. Endpoints are sync and run in FastAPI's thread pool, so threading locks are the right primitive. Different threads never share a lock.
  - **`/agent`:** holds the thread lock across ownership validation, the pending-interrupt check (`409 THREAD_AWAITING_APPROVAL`) and the whole `invoke`.
  - **`/agent/resume`:** holds the same lock across ownership validation (reviewer tenant must match), the pending-approval check and the `Command(resume=...)` invoke. The thread must be paused at `escalation_handler` (`APPROVAL_NODE`), checked via `get_state().tasks[*].interrupts`. Otherwise it returns `409 THREAD_NOT_AWAITING_APPROVAL` without invoking the graph.

### API behavior changes
| Request | Before | After |
|---|---|---|
| `/agent` with unknown `thread_id` | 200, thread claimed by caller | 403 `THREAD_NOT_ACCESSIBLE`, graph not invoked |
| `/agent` on another customer's thread | 403 "Thread belongs to a different customer" | 403 `THREAD_NOT_ACCESSIBLE` (same as unknown) |
| `/agent` on a thread paused for approval | 200, approval abandoned | 409 `THREAD_AWAITING_APPROVAL`, no state or memory change |
| `/agent/resume` unknown thread | 404 "Thread not found" | 403 `THREAD_NOT_ACCESSIBLE` |
| `/agent/resume` other tenant's thread | 403 "Thread belongs to a different tenant" | 403 `THREAD_NOT_ACCESSIBLE` (same as unknown) |
| `/agent/resume` idle/completed thread | 200 with stale `intent_results` | 409 `THREAD_NOT_AWAITING_APPROVAL`, graph not invoked |

### Fixes applied
- **tests/test_jwt_auth.py**:
  - `test_reviewer_token_on_resume_succeeds`: the mocked graph now reports a pending `escalation_handler` interrupt, because resume requires one.
  - `test_reuse_another_customers_thread` and `test_resume_cross_tenant_thread`: still assert 403, but check `THREAD_NOT_ACCESSIBLE` instead of the old messages that revealed the thread's existence.

### Tests (personalTestCases/)
- **test_escalation_handler.py**: negative, float, string and boolean (`True`/`False`) index regressions.
- **test_state_reducer.py**: reducer clear/append semantics and `AgentState` wiring.
- **test_before_router.py**: which per-turn fields reset and which memory is kept.
- **test_graph_integration.py**: real `build_graph()` with `Send()` fan-out, reducer merging, `interrupt()` and `Command(resume=...)`, and no leakage between turns.
- **test_api_turn_lifecycle.py**: `/agent` turn reset, 409 on paused threads with no side effects, resume without reset, acceptance after approval/denial, and unauthorized access.
- **test_thread_ownership_concurrency.py**:
  - Ownership: unknown IDs, server-issued IDs, owner reuse, other customer/tenant, concurrent registration.
  - Concurrency, using Events and an instrumented lock, no sleeps: same-thread serialization, independent threads, 409 when the first request creates the interrupt, `/agent` and resume never interleaving, lock-registry handoff and cleanup.
  - Resume safety.

### Known limitations
- **In-process coordination only:** the per-thread locks, `_THREAD_OWNERS` and `MemorySaver` all live in one process. Separate workers or processes are not coordinated, so the same thread can still race across workers, and a restart loses ownership records and paused approvals. A shared checkpointer, ownership store and distributed lock would be needed.
- **No approval timeout:** a paused thread blocks its customer until a reviewer acts.
- **Lock held across the full invocation:** a slow LLM call delays later requests on the same thread. Other threads are unaffected.
