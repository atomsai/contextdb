# ContextDB

**The unified context layer for AI agents.**

ContextDB replaces the Pinecone + Redis + Postgres + glue-code patchwork with
one system that understands memory — factual, experiential, and working —
across semantic, temporal, causal, and entity graphs, **and types every
fact so an agent can act without treating a wish as a booking**.

> Databricks Lakebase gives agents a hard drive. ContextDB gives agents a brain.

## Status

v0.2.0 — 122 tests passing (34 trust-model acceptance evals), type-checked
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

## License

Apache 2.0. Open-core: the SDK and trust evals stay public; hosted
operations are a separate Cloud product. See [OPEN_CORE.md](../OPEN_CORE.md)
and [COMMERCIAL.md](../COMMERCIAL.md). ContextDB is a trademark. A
public credit line is optional; the preferred form is in
[COMMERCIAL.md](../COMMERCIAL.md#attribution-optional).
