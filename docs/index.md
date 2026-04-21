# ContextDB

**The unified context layer for AI agents.**

ContextDB replaces the Pinecone + Redis + Postgres + glue-code patchwork with
one system that understands memory — factual, experiential, and working —
across semantic, temporal, causal, and entity graphs.

> Databricks Lakebase gives agents a hard drive. ContextDB gives agents a brain.

## Status

Pre-release (v0.1.0). Under active construction — see [`TASKS.md`][tasks] for
the 32-task build plan.

## Install

```bash
pip install contextdb
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

[tasks]: ../TASKS.md
