# Connector contracts

ContextDB's public SDK defines a read-only connector contract and a minimal
PostgreSQL/Supabase reference reader. It deliberately does **not** contain a
scheduler, credential service, retry queue, managed worker, or hosted sink.

```python
from contextdb.connectors import (
    ConnectorCursor,
    PostgresConnector,
    PostgresConnectorConfig,
)

source = PostgresConnector(
    PostgresConnectorConfig(
        dsn="postgresql://...",
        ssl_mode="require",
        relation="public.customer_context",
        primary_key="id",
        user_id="customer_id",
        content="context_text",
        updated_at="updated_at",
        deleted_at="deleted_at",
    )
)

await source.validate()

cursor: ConnectorCursor | None = None
async for page in source.records(cursor=cursor, limit=10_000):
    for record in page:
        # The host decides how to form, store, retry, or delete memory.
        ...
    if page:
        last = page[-1]
        cursor = ConnectorCursor(last.updated_at, last.source_key)
```

## Required source shape

- A stable primary key.
- A user-partition column.
- A content column.
- An `updated_at` timestamp.
- Optional `deleted_at` for verifiable soft-delete propagation.

The cursor is `(updated_at, primary_key)`. Timestamp-only cursors can miss rows
when multiple updates share the same timestamp.

Hard deletes are not discoverable through timestamp polling. Hosts must require
soft deletes, periodically reconcile the complete key set, or use a CDC
implementation outside this reference reader.

Credentials are supplied directly by the local host. ContextDB Cloud's managed
Secret Manager, scheduling, checkpoints, retries, DLQ, monitoring, and
operational guarantees are private operated capabilities.
