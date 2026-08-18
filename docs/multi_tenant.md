# Multi-tenant hosts

`init(user_id=...)` is the right API for a single-user process. A sidecar
that serves many callers should not construct one client per user and hold
them forever.

## Preferred: one client, `user_id` on the call

```python
db = contextdb.init(storage_url="postgresql://...")  # no user_id

await db.factual.add("Thursday works", source="user_stated", user_id="caller-7")
hits = await db.factual.recall("when", user_id="caller-7")
pending = await db.factual.pending_confirmations(user_id="caller-7")
```

Cost: one store pool, one in-process vector index, one embed cache.
Isolation is the same SQL `user_id = ?` predicate a scoped client uses.
A scoped client still cannot be widened — `init(user_id="alice")` then
`recall(..., user_id="bob")` raises.

HTTP: omit `user_id` at `init()` and authenticate each request — the
resolved scope comes from `auth_hook` (or, for token-only deployments,
`X-ContextDB-User` / the JSON body). See [serve.md](serve.md#request-scoping).

## What is (and is not) an authorization boundary

A **scoped client** (`init(user_id=...)`) is a boundary: the store
refuses every other user's rows, and ID-based `confirm` / `forget`
reject a foreign `memory_id` as if it did not exist.

A **shared unscoped client with per-call `user_id=`** is a *scoping
convenience* for trusted, in-process code. It keeps users' rows apart,
but nothing stops the caller from passing a different `user_id` on the
next call — do not expose it to untrusted input directly. The
authorization boundary for remote callers is the HTTP/MCP layer
(`create_app` with a token or `auth_hook`), which authenticates the
request, pins the scope, and rejects conflicts with a 400.

Likewise, running the HTTP server anonymously on loopback is local
convenience only — **not** an authorization boundary. Any local process
can claim any `user_id`. Bind non-loopback only behind a token or an
`auth_hook`.

## If you cannot change call sites: `ContextDBPool`

```python
from contextdb import ContextDBPool

pool = ContextDBPool(config, max_clients=256)
db = pool.client("caller-7")
```

Each client keeps a connection and a vector index. At 10k users that is
10k indexes unless you evict. The pool is LRU. Prefer the shared client.

## Tenant vs user

`tenant_id` is a second scope (org). Same `user_id` in two tenants does
not share rows. Set it at `init(tenant_id=...)` for a process that only
serves one org, or stamp it on writes via the item fields.
