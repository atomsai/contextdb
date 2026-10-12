# ContextDB

**The unified context layer for AI agents.**

ContextDB replaces the Pinecone + Redis + Postgres + glue-code patchwork with
one system that understands memory — factual, experiential, and working —
across semantic, temporal, causal, and entity graphs, **and types every
fact so an agent can act without treating a wish as a booking**.

> Databricks Lakebase gives agents a hard drive. ContextDB gives agents a brain.

## Status

CI runs the full test suite, the trust-model acceptance evals in
`tests/evals/`, ruff, and mypy on Python 3.10 to 3.12 and PostgreSQL. With
SQLite and the mock embedder (no model call), search p95 was about 18 ms at
5,000 memories on an Apple M4 Max in October 2026. A real embedding model adds
its own latency. The fabrication bake-off scores eight synthetic traps with a
scripted model: the trust arm fabricated 0% and recalled 100%, against 50% and
88% for an untyped store. That is regression evidence inside the harness, not
a measure of production behavior.

## Install

```bash
pip install pycontextdb
```

## Quick start

```python
import contextdb

db = contextdb.init(user_id="user_123")
await db.factual.add(
    "I'd like to come in Thursday",
    source="user_stated",
    confidence=0.5,
    action_relevant=True,
    entity="caller",
    attribute="preferred_visit_day",
)
# Recalled, but not actionable — it is a wish.
trusted = await db.factual.recall_for_action("come in Thursday")
assert trusted == []
await db.factual.confirm(...)  # user said yes; now it may gate a booking
```

The full walkthrough is in [quickstart.md](quickstart.md). The trust model
is specified by the evals in `tests/evals/test_trust_model.py` — where
prose and evals disagree, the evals win.

Source ingestion contracts and the local PostgreSQL/Supabase reference reader
are documented in [connectors.md](connectors.md).

Caller-selected `ADD` / `UPDATE` / `DELETE` / `NOOP`, lineage, audit, and
consistency semantics are specified in
[memory_evolution.md](memory_evolution.md).

## License

Apache 2.0. The SDK and trust evals are fully public; nothing in this
tree is gated behind a hosted offering. ContextDB is a trademark;
reselling it as a service (or as part of one) under the ContextDB name
requires a commercial agreement — see
[COMMERCIAL.md](../COMMERCIAL.md) (contextdb@atomsai.com). A public
credit line is optional; if you want one, use **Built with ContextDB**
linking to the repository.
