# Architecture

ContextDB is a five-layer system:

```
┌──────────────────────────────────────────────┐
│          Client: ContextDB                   │  ← public surface
├──────────────────────────────────────────────┤
│ Memory APIs (factual / experiential / working)│  ← typed surfaces
├──────────────────────────────────────────────┤
│ Dynamics (formation / evolution / retrieval) │  ← pipelines
├──────────────────────────────────────────────┤
│ Graphs (semantic / temporal / causal / entity)│ ← edge indices
├──────────────────────────────────────────────┤
│ Storage (SQLite/Postgres + FAISS/NumPy)      │  ← durable bytes
└──────────────────────────────────────────────┘
         Privacy: PII detection, retention, audit
```

## Why graphs over a single vector table

A single similarity retriever is fine for "find me things about X" but it
does not know that *"last Tuesday"* is a temporal constraint, that *"why did
the deploy fail"* is causal, or that *"Alice"* is an entity with a biography
stretched across dozens of memories. ContextDB layers four orthogonal
graphs over the same `memories` table so each of those signals contributes
to the ranking via Reciprocal Rank Fusion (k=60).

## Dynamics

Three pipelines operate over the storage layer:

1. **Formation** — a conversation becomes memories. Segment → extract facts
   with the LLM → run PII detection → embed → write.
2. **Evolution** — memories age. Auto-linker mirrors each new write into
   graph indices; consolidator merges dense semantic clusters into
   summaries; pruner drops stale / redundant memories by policy.
3. **Retrieval** — a query becomes an answer. Query classifier picks graph
   weights; each graph produces a ranking; RRF fuses them.

## Privacy is a layer, not an afterthought

PII detection runs before the embedder ever sees a raw email address or
SSN. The audit logger hash-chains every write, search, and deletion. The
retention manager applies typed TTLs and honors right-to-erasure requests.
