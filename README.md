# ContextDB

**The unified context layer for AI agents.**

Replace your patchwork of Pinecone + Redis + Postgres + glue code with one system that understands memory.

> Databricks gives agents a hard drive. ContextDB gives agents a brain.

---

## What ContextDB Replaces

| What you use today | What breaks | ContextDB equivalent |
|---|---|---|
| Pinecone / Qdrant | Semantic-only; no temporal or causal awareness | Multi-graph retrieval (semantic + temporal + causal + entity) |
| Redis / Memcached | Ephemeral; lost between sessions | Working memory with token-budget paging + compression |
| PostgreSQL / MongoDB | Static rows; no memory lifecycle | Factual memory with formation, evolution, and learned policies |
| S3 / flat files | Write-only; not queryable | Experiential memory (trajectories, reflections, workflows) |
| Custom glue code | Brittle; no consistency | Unified SDK with one `init()` call |
| *(nothing)* | Agents never learn from outcomes | RL-trained memory manager (ADD/UPDATE/DELETE/NOOP) |
| *(nothing)* | Raw PII stored indefinitely | Privacy-by-design with PII detection and retention policies |

## Quick Start

```python
import contextdb

# Initialize
ctx = contextdb.init(user_id="user_123")

# Add memories
await ctx.add("Customer prefers email over phone", memory_type="factual")
await ctx.add("Resolved billing issue by applying 20% discount", memory_type="experiential")

# Search
results = await ctx.search("How does this customer prefer to be contacted?")

# The agent now remembers
```

## Install

```bash
pip install contextdb
```

## Documentation

Coming soon at [contextdb.dev](https://contextdb.dev)

## License

Apache 2.0 — see [LICENSE](LICENSE)

## Paper

Read the research: [ContextDB: A Unified Context Layer for AI Agents](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=XXXXXXX)

Analyzing 200+ papers on agentic memory to design a memory operating system for AI agents.
