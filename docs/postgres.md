# Postgres store

The `postgres` extra used to install `asyncpg` and then fail at runtime.
Self-hosters hit SQLite's single-writer limit immediately.

```bash
pip install 'pycontextdb[postgres]'
```

```python
db = contextdb.init(storage_url="postgresql://user:pass@host:5432/contextdb")
```

`postgres://` and `postgresql+asyncpg://` are accepted. Schema is created
on first `await`. Vectors are `BYTEA` plus the same in-process index SQLite
uses. `pgvector` is not required.

What this is: a multi-writer OSS store with the same trust model, PII
rules, and audit chain.

What this is not: multi-region failover, dashboards, or a managed control
plane — those are out of scope for the OSS store.

## Multi-process behavior

The store is safe to share across processes and workers:

* **Slot writes** (dedupe/corroborate/supersede) serialize on a
  transaction-scoped Postgres advisory lock keyed by
  `(user, tenant, entity, attribute)`, so two workers cannot interleave a
  read-slot-then-write.
* **Audit-chain appends** serialize on an advisory lock as well — the
  hash chain never forks, no matter how many workers append.
* **Vector recall** keeps a process-local index for speed. Every write
  bumps a global revision counter in the database; a process whose index
  predates the current revision rebuilds from the authoritative rows
  before serving a search. A write from worker A is therefore visible to
  worker B's next recall — at the cost of a rebuild per foreign write
  burst. Candidate ids are always scoped by SQL before ranking, so
  recall never leaks across users regardless of index freshness.

SQLite remains the single-process backend: its slot and audit locks are
in-process by design.

## Reusing an application-owned pool

High-throughput hosts should create one asyncpg pool and pass it to scoped
ContextDB runtimes. ContextDB uses the pool but never closes it.

```python
import asyncpg
import contextdb

pool = await asyncpg.create_pool(
    "postgresql://user:pass@host:5432/contextdb",
    min_size=2,
    max_size=20,
)

support = contextdb.init(
    storage_url="postgresql://user:pass@host:5432/contextdb",
    tenant_id="acme",
    agent_id="support",
    postgres_pool=pool,
)
sales = contextdb.init(
    storage_url="postgresql://user:pass@host:5432/contextdb",
    tenant_id="acme",
    agent_id="sales",
    postgres_pool=pool,
)

try:
    await support.factual.recall("open issue", user_id="customer-42")
finally:
    await support.close()
    await sales.close()
    await pool.close()
```

The pool must belong to the same event loop as the runtimes. Passing
`postgres_pool` with a SQLite URL is a configuration error.

## Implementing your own store

Subclass `contextdb.store.base.BaseStore`. You need `initialize`, `add`,
`get`, `update`, `delete`, `search_by_embedding`, `list_memories`,
`count`, and `close`. The client also calls `get_raw`, `list_by_slot`,
`list_by_entity`, `list_pending_consolidation`, `iter_memories`,
`delete_older_than`, `count_any_status`, `count_by_type`, `index_ids`,
`slot_lock`, and `audit_lock`. Copy those signatures from `SQLiteStore` /
`PostgresStore`.

Scope rules you must keep:

* A store constructed with `user_id` cannot see or write another user.
* An unscoped store filters by the per-call `user_id`.
* Writes stamp `item.user_id`.
* Slot locks are keyed on `(user, tenant, entity, attribute)`.

Concurrency and recall rules you must keep:

* `slot_lock` and `audit_lock` must serialize not just within one event
  loop but across every process that shares the database (the Postgres
  store uses transaction-scoped advisory locks). The audit chain forks
  silently without this.
* `search_by_embedding` must restrict vector candidates to the requested
  scope *before* ranking — rank-then-filter leaks the candidate budget
  to foreign tenants. Custom `VectorIndex` implementations must accept
  `search(query, top_k, include_ids=None)`.
* If you keep a process-local index or cache, it must track the store
  revision (`contextdb_meta`) so a write from another process is visible
  to the next search.
