# API Reference

## Top-level

```python
contextdb.init(
    user_id=None,
    config=None,
    *,
    tenant_id=None,
    agent_id=None,
    session_id=None,
    clock=None,
    trust_policy=None,
    **kwargs,
) -> ContextDB
contextdb.__version__
```

`TrustPolicy`, `FrozenClock`, and `utc_now` are exported from `contextdb`.

## ContextDB client

```python
async def add(content, memory_type=FACTUAL, metadata=None, event_time=None,
              source="", entity_mentions=None, *,
              epistemic_source=None, confidence=None, action_relevant=None,
              entity_key=None, attribute_key=None) -> MemoryItem
async def add_fast(content, memory_type=FACTUAL, ...) -> MemoryItem
    # Never calls an LLM. Recallable immediately; consolidate later.
async def search(query, top_k=10, memory_type=None, time_range=None,
                 as_of=None, compose=False, *, user_id=None, entity=None,
                 min_confidence=None, include_third_party=True) -> list[MemoryItem]
    # Queries are PII-redacted before embed. compose hops sibling slots.
async def confirm(memory_id, user_id=None) -> MemoryItem
async def get(memory_id) -> MemoryItem | None
async def update(memory_id, content=None, metadata=None) -> MemoryItem
async def delete(memory_id, hard=False) -> None
async def add_conversation(conversation, source="") -> list[MemoryItem]
async def add_many(items, *, user_id=None) -> list[MemoryItem]
async def add_fast_many(contents, *, user_id=None) -> list[MemoryItem]
async def pending_confirmations(user_id=None, limit=100) -> list[MemoryItem]
async def forget(user_id=None, entity=None, older_than=None, *,
                 memory_id=None, attribute=None) -> int
async def forget_user(user_id) -> int
db.on(event, hook)  # write, recall, confirm, forget, injection_suspect, embed_fallback
async def verify_forgotten(user_id) -> bool
async def explain(memory_id) -> MemoryExplanation
async def stats() -> dict
async def consolidate(min_cluster_size=5) -> list[MemoryItem]
async def consolidate_pending(batch_size=50) -> int
async def prune(strategy="decay", **kwargs) -> int
async def get_timeline(entity=None, start=None, end=None) -> list[MemoryItem]
async def get_entity(name) -> dict
```

### Typed surfaces

* `db.factual` — `FactualMemory`
  * `add(..., source=, confidence=, action_relevant=, entity=, attribute=, user_id=)`
  * `add_fast(content, user_id=)` — no LLM
  * `add_many(items, user_id=)` — batch; missing source uses add_fast
  * `recall(query, top_k=5, as_of=None, user_id=, entity=, min_confidence=, include_third_party=)`
  * `recall_for_action(query, ..., user_id=)` — only facts that pass `db.trust_policy`
  * `confirm(memory_id, user_id=)` — user said yes; graduates the fact
  * `pending_confirmations(user_id=)` — facts waiting on a yes
* `db.experiential` — `ExperientialMemory`
* `db.working(session_id, max_tokens=4000)` — `WorkingMemory`

### Trust

* `db.trust_policy` — `TrustPolicy` (`TrustPolicy.hospital()`, `.restaurant()`)
* `contextdb.integrations.act.VerifyBeforeAct(db).decide(query)` → `act` / `ask` / `abstain`

### Privacy

* `db.privacy` — `RetentionManager`
* `db.audit` — `AuditLogger | None` (SEARCH + DECIDE + FORGET are chained)

### Multi-agent

* `db.bus()` — `MemoryBus`

## Models

`MemoryItem`, `Edge`, `Entity`, `RetentionPolicy`, `PIIAnnotation`,
`MemoryType`, `MemoryStatus`, `PIIType`, `TrustPolicy`, `MemoryExplanation`.

Load-bearing `MemoryItem` fields for the trust model: `epistemic_source`,
`corroborated_by`, `confirmed`, `contested`, `action_relevant`,
`entity_key` / `attribute_key` / `slot_class` / `slot_value`,
`valid_from` / `valid_until` / `superseded_by`, `injection_suspect`.

See [`contextdb/core/models.py`](https://github.com/atomsai/contextdb/blob/main/contextdb/core/models.py)
for the authoritative definitions. Evals in `tests/evals/` win over this page.
