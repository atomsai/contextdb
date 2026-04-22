# ContextDB

**The unified context layer for AI agents.**

[![PyPI version](https://img.shields.io/pypi/v/contextdb.svg)](https://pypi.org/project/contextdb/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-67%20passing-brightgreen.svg)](tests/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Type Checked](https://img.shields.io/badge/mypy-strict-blue.svg)](pyproject.toml)

Replace your patchwork of Pinecone + Redis + Postgres + glue code with one system that understands memory.

---

## The problem

Every team shipping agents assembles the same broken stack: vectors in Pinecone, sessions in Redis, profiles in Postgres, logs in S3, and 2-4 months of glue code holding it together. It breaks at multi-session reasoning (the agent forgets you between calls), temporal understanding ("last Tuesday" is not a vector), and learning from experience (nothing records what worked).

Databricks Lakebase gives agents a hard drive. ContextDB gives agents a brain.

---

## What ContextDB replaces

| What you use today | What breaks | ContextDB equivalent |
|---|---|---|
| Pinecone / Qdrant / Weaviate | Semantic-only; no temporal, causal, or entity awareness | Multi-graph retrieval fused with RRF (semantic + temporal + causal + entity) |
| Redis / Memcached | Ephemeral; lost between sessions; no compression | `WorkingMemory` with token-budget paging and FIFO eviction |
| PostgreSQL / MongoDB | Static rows; no lifecycle; no graph links | `FactualMemory` with formation, evolution, and consolidation |
| S3 / flat files | Write-only archive; not queryable by meaning or time | `ExperientialMemory` for trajectories, reflections, and workflows |
| Custom glue code | Brittle; rebuilt at every company; 2-4 engineering months | One `init()` call, one async SDK, one dependency |
| *(nothing today)* | Agents never learn from outcomes | RL-trained memory manager (`ADD` / `UPDATE` / `DELETE` / `NOOP`) |
| *(nothing today)* | Raw PII stored indefinitely; compliance risk | PII detection, typed TTLs, hash-chained audit log |

---

## Quick start

```python
import asyncio
import contextdb

async def main() -> None:
    db = contextdb.init(
        user_id="cust_42",
        embedding_model="mock", llm_model="mock", llm_api_key="mock",  # drop for real providers
    )
    async with db:
        await db.factual.add("Customer runs a Carrier 24ACC6 AC unit installed 2019.")
        await db.experiential.record_trajectory(
            action="clear condenser coil",
            outcome="restored cooling within 8 minutes",
            success=True,
        )
        hits = await db.search("how do I fix weak airflow on a Carrier unit")
        for h in hits:
            print(h.content)

asyncio.run(main())
```

One import, one `init`, one `async with`. No infra to stand up.

---

## Factual memory

Durable knowledge about users, entities, and the world. Survives sessions; answers "what do I know about X."

```python
import asyncio
import contextdb

async def main() -> None:
    db = contextdb.init(
        user_id="cust_42",
        embedding_model="mock", llm_model="mock", llm_api_key="mock",
    )
    async with db:
        await db.factual.add("Customer owns a Carrier 24ACC6, installed March 2019.")
        await db.factual.add("Preferred contact channel: email. Not phone.")
        await db.factual.add("Home is a 2,400 sq ft single-story in Phoenix, AZ.")

        hits = await db.factual.recall("what AC model does this customer have?")
        print(hits[0].content)  # → "Customer owns a Carrier 24ACC6..."

asyncio.run(main())
```

---

## Experiential memory

What the agent did, what happened, what to do next time. This is the substrate for agent self-improvement.

```python
import asyncio
import contextdb

async def main() -> None:
    db = contextdb.init(
        user_id="cust_42",
        embedding_model="mock", llm_model="mock", llm_api_key="mock",
    )
    async with db:
        traj = await db.experiential.record_trajectory(
            action="walked customer through filter replacement",
            outcome="airflow restored; customer satisfied",
            context={"ticket": "T-1139", "unit": "Carrier 24ACC6"},
            success=True,
        )
        await db.experiential.add_reflection(
            trajectory_id=traj.id,
            insight="For Carrier 24ACC6 weak-airflow cases, check filter before coil.",
        )

        similar = await db.experiential.recall_similar(
            "customer reports weak airflow on Carrier unit"
        )
        for s in similar:
            print(s.content)

asyncio.run(main())
```

---

## Working memory

Token-budgeted session scratchpad. Oldest entries evict FIFO when you cross the budget; the LLM never sees more than you pay for.

```python
import asyncio
import contextdb

async def main() -> None:
    db = contextdb.init(
        user_id="cust_42",
        embedding_model="mock", llm_model="mock", llm_api_key="mock",
    )
    async with db:
        session = db.working(session_id="call_7781", max_tokens=800)

        await session.push("Customer called about AC not cooling.")
        await session.push("Diagnosed: condenser coil clogged with cottonwood fluff.")
        await session.push("Walked through cleaning procedure over phone.")
        await session.push("Customer confirms cold air after 10 minutes.")

        window = await session.context_window()
        print(window)  # exactly what to paste into the next LLM call

asyncio.run(main())
```

---

## Multi-graph retrieval

A semantic-only retriever is fine for "find things about X." It does not know that *"last Tuesday"* is a temporal constraint, that *"why did the repair fail"* is causal, or that *"the Phoenix customer"* is an entity. ContextDB layers four orthogonal graphs over the same memory table and fuses them via Reciprocal Rank Fusion (k=60).

```python
import asyncio
import contextdb

async def main() -> None:
    db = contextdb.init(
        user_id="cust_42",
        embedding_model="mock", llm_model="mock", llm_api_key="mock",
        enable_multi_graph=True,   # unlocks temporal + causal graphs
    )
    async with db:
        await db.add("Coil cleaning on 2026-04-14 restored airflow.")
        await db.add("Compressor trip on 2026-04-18 caused no-cool condition.")
        await db.add("Replacing the contactor cleared the compressor trip.")

        # Temporal intent: the query classifier boosts the temporal graph.
        recent = await db.search("what went wrong last week with this unit?")

        # Causal intent: causal graph boosts cause/effect chains.
        causal = await db.search("why did the compressor stop?")

        for h in recent + causal:
            print(h.content)

asyncio.run(main())
```

The query classifier assigns weights per graph based on markers in the query; each graph returns a ranked list; RRF fuses them. You do not pick which index to hit.

---

## Privacy by design

PII is detected, classified, and redacted **before** the embedder ever sees it. Writes, reads, and deletions are recorded in a hash-chained audit log. Retention TTLs are typed — factual memories live for 5 years, working memories for 24 hours, experiential memories indefinitely — all configurable.

```python
import asyncio
import contextdb

async def main() -> None:
    db = contextdb.init(
        user_id="cust_42",
        embedding_model="mock", llm_model="mock", llm_api_key="mock",
        pii_action="redact",
    )
    async with db:
        item = await db.add(
            "Customer Alex Rivera (alex@example.com, 555-213-8844, "
            "card 4111-1111-1111-1111) reported no cooling."
        )
        print(item.content)
        # → "Customer Alex Rivera ([EMAIL], [PHONE], card [CREDIT_CARD]) reported no cooling."

        # Tamper-evident audit chain covering every CREATE / READ / SEARCH / DELETE.
        assert db.audit is not None
        assert await db.audit.verify_chain() is True

asyncio.run(main())
```

Right-to-erasure is a first-class operation:

```python
await db.forget(entity="Alex Rivera")   # bulk delete + audit entries
```

---

## Framework integrations

Drop ContextDB into whatever stack you already ship. Each adapter is a thin duck-typed wrapper — no hard dependency on the framework.

**LangChain:**

```python
import contextdb
from contextdb.integrations.langchain import ContextDBMemory

db = contextdb.init(user_id="cust_42",
                    embedding_model="mock", llm_model="mock", llm_api_key="mock")
memory = ContextDBMemory(db, session_id="chat-1")
# memory conforms to LangChain's async memory interface:
#   aload_memory_variables(inputs) / asave_context(inputs, outputs) / aclear()
```

**OpenAI Agents (function calling):**

```python
import contextdb
from contextdb.integrations.openai_tools import tool_schemas, make_tool_handlers

db = contextdb.init(user_id="cust_42",
                    embedding_model="mock", llm_model="mock", llm_api_key="mock")
tools = tool_schemas()                      # JSON schemas for chat.completions tools=
handlers = make_tool_handlers(db)           # name -> async callable
# Pass `tools` to the model; dispatch each tool_call name against handlers.
```

**CrewAI:**

```python
import contextdb
from contextdb.integrations.crewai import ContextDBCrewMemory

db = contextdb.init(user_id="cust_42",
                    embedding_model="mock", llm_model="mock", llm_api_key="mock")
crew_memory = ContextDBCrewMemory(db, top_k=5)
# Implements save / search / reset — the full CrewAI memory contract.
```

AutoGen adapter ships in `contextdb.integrations.autogen` with the same shape.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Application Layer                      │
│   (Your agent: support bot, phone agent, copilot, crew)   │
└─────────────────────────────┬────────────────────────────┘
                              │  contextdb SDK (async)
┌─────────────────────────────▼────────────────────────────┐
│                      ContextDB Core                       │
│                                                            │
│   Memory Types          Dynamics Engine      Graph Layer   │
│   ────────────          ───────────────      ───────────   │
│   • Factual             • Formation          • Semantic    │
│   • Experiential        • Evolution          • Temporal    │
│   • Working             • Retrieval (RRF)    • Causal      │
│                                              • Entity      │
└─────────────────────────────┬────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────┐
│                       Storage Layer                       │
│         SQLite (default)  │  PostgreSQL  │  FAISS         │
└─────────────────────────────┬────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────┐
│                       Privacy Layer                       │
│     PII Detection  │  Retention TTLs  │  Hash-Chain Audit │
└──────────────────────────────────────────────────────────┘
```

Five layers, one dependency. Privacy is a layer, not an afterthought — PII never reaches the embedder unprocessed.

---

## Benchmarks

Runnable locally — no API keys required. MockEmbedding (dim=1536) + MockLLM isolate storage and retrieval from provider latency.

```bash
python benchmarks/run_benchmarks.py
```

The suite covers six workloads:

1. **Write throughput** — 1,000 sequential `add()` calls.
2. **Search latency** — p50 / p95 / p99 over 100 queries against 1,000 memories.
3. **Search vs scale** — the same workload at 100, 500, 1K, and 5K memories.
4. **PII detection** — 1,000 redactions with correctness verification.
5. **Vector index** — 10K × 1,536-dim vectors, 100 queries, self-retrieval check.
6. **End-to-end** — a customer-support agent scenario with PII redaction roundtrip.

Representative results (MacBook-class laptop, no FAISS):

| Metric | Result |
|---|---|
| Write throughput | 1,900+ writes/sec |
| Search p50 @ 1K memories | ~3ms |
| Search p95 @ 5K memories | ~5ms |
| PII redaction | 100,000+ texts/sec |
| Vector search @ 10K × 1,536d | p95 ~1ms |
| Self-retrieval accuracy | PASS |

All benchmarks run against a tempfile SQLite DB, so they are reproducible and hermetic.

---

## Why not just use...

**Mem0?** Graph intelligence is gated behind the paid tier. No experiential memory for trajectories and reflections. No RL-trained memory manager. No working memory with token budgets.

**Zep?** Strong bitemporal knowledge graphs. But no experiential memory, no working memory paging, no learned retrieval policies. Scope is narrower than a full memory OS.

**MemGPT / Letta?** OS-style working memory paging is elegant, but there's no persistent factual memory, no graph structures, and no multi-agent primitives. Great for one long chat; thin for a real product.

**Pinecone + Redis + Postgres + glue code?** That is the patchwork. It is exactly what ContextDB replaces. Five dependencies, three query languages, one brittle seam at each boundary, and nobody at your company learns anything from last quarter's tickets.

**Databricks Lakebase?** Storage, not cognition. Lakebase gives your agent a managed Postgres with pgvector and LangGraph checkpointing — a hard drive. ContextDB sits above storage (and can run on Lakebase as a backend) and provides the memory semantics: formation, evolution, retrieval, privacy.

---

## Research

Built on analysis of 200+ papers on agentic memory. The taxonomy — **Forms × Functions × Dynamics** — organizes agent memory along three axes:

- **Forms** — how memory is represented (token, parametric, latent).
- **Functions** — what memory is for (factual, experiential, working).
- **Dynamics** — how memory changes (formation, evolution, retrieval).

ContextDB is the first system to span all three axes in one library.

Paper: [*ContextDB: A Unified Context Layer for AI Agents*](https://zenodo.org/records/19647089) (Zenodo, 2026).

---

## Installation

```bash
pip install contextdb                 # core: SQLite, NumPy vector index, all memory types
pip install "contextdb[faiss]"        # FAISS-accelerated vector index
pip install "contextdb[postgres]"     # asyncpg + pgvector backend
pip install "contextdb[all]"          # everything
```

Python 3.10+. No system dependencies for the default install — SQLite and NumPy ship with Python.

---

## Contributing

Apache 2.0 — see [LICENSE](LICENSE). Contribution guide at `CONTRIBUTING.md` (coming soon). Start with `TASKS.md` for the full build plan and `docs/architecture.md` for the design rationale. Issues and pull requests welcome on [GitHub](https://github.com/atomsai/contextdb).

If you use ContextDB in research, please cite the paper: [zenodo.org/records/19647089](https://zenodo.org/records/19647089).
