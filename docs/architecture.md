# Architecture

ContextDB is a five-layer system:

```
┌──────────────────────────────────────────────┐
│          Client: ContextDB                   │  ← public surface
├──────────────────────────────────────────────┤
│ TrustPolicy + VerifyBeforeAct + confirm()    │  ← action bar
├──────────────────────────────────────────────┤
│ Memory APIs (factual / experiential / working)│  ← typed surfaces
├──────────────────────────────────────────────┤
│ Dynamics (formation / evolution / retrieval) │  ← pipelines
├──────────────────────────────────────────────┤
│ Graphs (semantic / temporal / causal / entity)│ ← edge indices
├──────────────────────────────────────────────┤
│ Storage (SQLite/Postgres + FAISS/NumPy)      │  ← durable bytes
│         scoped by user / tenant / agent      │
└──────────────────────────────────────────────┘
         Privacy: PII-before-embedder, retention, hash-chained audit
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
   with the LLM (epistemic source, confidence, action_relevant, slot keys)
   → run PII detection → embed only the redacted text → write through the
   trust engine (dedupe / corroborate / supersede / contest).
2. **Evolution** — memories age. Auto-linker mirrors each new write into
   graph indices; consolidator merges dense semantic clusters into
   summaries that inherit *worst-case* trust (no laundering); pruner drops
   stale / redundant memories by policy.
3. **Retrieval** — a query becomes an answer. The query is PII-redacted
   before embed. Query classifier picks graph weights; each graph produces
   a ranking; RRF fuses them; salience (recency × frequency × criticality)
   multiplies the fused score. `factual.recall` hops sibling slots of the
   same entity. `recall_for_action` applies `TrustPolicy`.

## Trust write path

Memories that share `(entity_key, attribute_key)` are about the same thing:

* same value, new speaker → independent corroboration
* same value, same speaker → no-op on the count
* different value, same speaker → supersede (`valid_until`, `superseded_by`)
* different value, independent speaker → **contest** (both current, neither
  actionable until `confirm()`)

`add_fast` never calls an LLM. The deterministic slotter still keys the
write so a later consolidator cannot race a newer typed fact.

## Privacy is a layer, not an afterthought

PII detection runs before the embedder ever sees a raw email address or
SSN — on **writes and queries**. The audit logger hash-chains every write,
search, `DECIDE` (act/ask/abstain), and deletion — and because the chain
is append-only, only the PII-processed form of a query is ever logged.
`pii_action="encrypt"` fails closed: without a key the client refuses to
initialize rather than store plaintext annotation originals.
`forget_user` walks graph edges and signs the deletion set. The retention
manager applies typed TTLs and honors right-to-erasure requests. Isolation
is a store predicate (`user_id` / `tenant_id` / `agent_id`), not a
convention.
