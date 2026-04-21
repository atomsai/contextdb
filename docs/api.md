# API Reference

## Top-level

```python
contextdb.init(user_id=None, config=None, **kwargs) -> ContextDB
contextdb.__version__
```

## ContextDB client

```python
async def add(content, memory_type=FACTUAL, metadata=None, event_time=None,
              source="", entity_mentions=None) -> MemoryItem
async def search(query, top_k=10, memory_type=None, time_range=None)
                -> list[MemoryItem]
async def get(memory_id) -> MemoryItem | None
async def update(memory_id, content=None, metadata=None) -> MemoryItem
async def delete(memory_id, hard=False) -> None
async def add_conversation(conversation, source="") -> list[MemoryItem]
async def forget(user_id=None, entity=None, older_than=None) -> int
async def stats() -> dict
async def consolidate(min_cluster_size=5) -> list[MemoryItem]
async def prune(strategy="decay", **kwargs) -> int
async def get_timeline(entity=None, start=None, end=None) -> list[MemoryItem]
async def get_entity(name) -> dict
```

### Typed surfaces

* `db.factual` — `FactualMemory`
* `db.experiential` — `ExperientialMemory`
* `db.working(session_id, max_tokens=4000)` — `WorkingMemory`

### Privacy

* `db.privacy` — `RetentionManager`
* `db.audit` — `AuditLogger | None`

### Multi-agent

* `db.bus()` — `MemoryBus`

## Models

`MemoryItem`, `Edge`, `Entity`, `RetentionPolicy`, `PIIAnnotation`,
`MemoryType`, `MemoryStatus`, `PIIType`.

See [`contextdb/core/models.py`](https://github.com/contextdb/contextdb/blob/main/contextdb/core/models.py) for the
authoritative definitions.
