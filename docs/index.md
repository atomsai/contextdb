# ContextDB

**The unified context layer for AI agents.**

ContextDB replaces the Pinecone + Redis + Postgres + glue-code patchwork with
one system that understands memory — factual, experiential, and working —
across semantic, temporal, causal, and entity graphs, **and types every
fact so an agent can act without treating a wish as a booking**.

> Databricks Lakebase gives agents a hard drive. ContextDB gives agents a brain.

## Status

v0.2.0 — 139 tests passing (34 trust-model acceptance evals), type-checked
under `mypy --strict`, ruff clean. Search p95 under 5ms at 5K memories.
Fabrication bake-off: trust arm **0%** fabrication / **100%** recall vs
raw-store baseline 50% / 88%.

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

## License

Apache 2.0. The SDK and trust evals are fully public; nothing in this
tree is gated behind a hosted offering. ContextDB is a trademark;
reselling it as a service (or as part of one) under the ContextDB name
requires a commercial agreement — see
[COMMERCIAL.md](../COMMERCIAL.md) (contextdb@atomsai.com). A public
credit line is optional; if you want one, use **Built with ContextDB**
linking to the repository.
