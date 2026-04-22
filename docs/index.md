# ContextDB

**The unified context layer for AI agents.**

ContextDB replaces the Pinecone + Redis + Postgres + glue-code patchwork with
one system that understands memory — factual, experiential, and working —
across semantic, temporal, causal, and entity graphs.

> Databricks Lakebase gives agents a hard drive. ContextDB gives agents a brain.

## Status

v0.1.0 — 82 tests passing, type-checked under `mypy --strict`, ruff clean.
Search p95 under 5ms at 5K memories on commodity hardware.

## Install

```bash
pip install pycontextdb
```

## Quick start

```python
import contextdb

ctx = contextdb.init(user_id="user_123")
await ctx.add("Customer prefers email over phone", memory_type="factual")
results = await ctx.search("How does this customer prefer to be contacted?")
```

## License

Apache 2.0.
