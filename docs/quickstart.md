# Quick Start

ContextDB is a Python library. Install it with pip:

```bash
pip install pycontextdb
```

## Minimal example

```python
import asyncio
import contextdb

async def main():
    db = contextdb.init(user_id="user_123", llm_api_key="sk-...")
    async with db:
        wish = await db.factual.add(
            "I'd like to come in Thursday",
            source="user_stated",
            confidence=0.5,
            action_relevant=True,
            entity="caller",
            attribute="preferred_visit_day",
        )
        print(wish.requires_confirmation)  # True — a wish is not a booking
        print(await db.factual.recall_for_action("Thursday"))  # []
        await db.factual.confirm(wish.id)
        for hit in await db.factual.recall_for_action("Thursday"):
            print(hit.content)

asyncio.run(main())
```

## Configuration

Pass overrides directly, or via `CONTEXTDB_*` environment variables:

```python
db = contextdb.init(
    storage_url="postgresql://localhost/ctx",
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    enable_multi_graph=True,   # optional local temporal + causal graphs
    enable_rl_manager=False,   # optional local ADD/UPDATE/DELETE/NOOP policy
)
```

## Optional local features

These flags are optional local pathways in the Apache SDK — everything
here works offline.

| Flag | Default | What it enables |
|------|---------|------------------|
| `enable_entity_graph` | `True`  | LLM-extracted named-entity overlay |
| `enable_multi_graph`  | `False` | Temporal + causal graphs |
| `enable_rl_manager`   | `False` | Inference-time memory policy |
| `enable_audit`        | `True`  | Hash-chained audit log |
| `enable_read_audit`   | `True`  | Synchronous `SEARCH` entries in that chain |
| `enable_auto_link`    | `True`  | Mirror each write into graph indices |

`enable_read_audit=False` does not disable write, confirmation, or lifecycle
audit events. It is for high-throughput hosts that register a `recall` hook and
durably persist its PII-processed `audit_details` payload elsewhere. Dropping
that payload silently is not equivalent to auditing reads.

Pass `trust_policy=TrustPolicy.hospital()` (or `.restaurant()`) to change
the action bar without forking the product. Pass `clock=FrozenClock(...)`
in tests so `as_of` and `valid_until` agree about "now."

Realtime hosts should write with `db.factual.add_fast(...)` (no LLM on the
turn path) and gate actions with
`VerifyBeforeAct(db).decide(query)` — `act` / `ask` / `abstain`.

The fabrication bake-off is the public ruler:

```bash
python benchmarks/trust_bakeoff.py
```
