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
        await db.add("My birthday is March 5.")
        hits = await db.search("when is my birthday?")
        for hit in hits:
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
    enable_multi_graph=True,   # unlocks temporal + causal graphs (paid tier)
    enable_rl_manager=False,   # RL ADD/UPDATE/DELETE/NOOP policy (paid tier)
)
```

## Tier flags

| Flag | Default | What it enables |
|------|---------|------------------|
| `enable_entity_graph` | `True`  | LLM-extracted named-entity overlay |
| `enable_multi_graph`  | `False` | Temporal + causal graphs |
| `enable_rl_manager`   | `False` | Inference-time memory policy |
| `enable_audit`        | `True`  | Hash-chained audit log |
| `enable_auto_link`    | `True`  | Mirror each write into graph indices |
