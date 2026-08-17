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

What this is not: multi-region failover, dashboards, or retention
operations. Those stay in ContextDB Cloud.

## Implementing your own store

Subclass `contextdb.store.base.BaseStore`. You need `initialize`, `add`,
`get`, `update`, `delete`, `search_by_embedding`, `list_memories`,
`count`, and `close`. The client also calls `get_raw`, `list_by_slot`,
`list_by_entity`, `list_pending_consolidation`, `iter_memories`,
`delete_older_than`, `count_any_status`, `count_by_type`, `index_ids`,
and `slot_lock`. Copy those signatures from `SQLiteStore` / `PostgresStore`.

Scope rules you must keep:

* A store constructed with `user_id` cannot see or write another user.
* An unscoped store filters by the per-call `user_id`.
* Writes stamp `item.user_id`.
* Slot locks are keyed on `(user, tenant, entity, attribute)`.
