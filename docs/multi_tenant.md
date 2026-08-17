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

HTTP: omit `user_id` at `init()` and send `X-ContextDB-User` per request.

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
