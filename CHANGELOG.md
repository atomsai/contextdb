# Changelog

## Unreleased

- `factual.recall_for_action` and `factual.pending_confirmations` accept the
  `entity` and `attribute` an action needs. With them, only current facts in
  that slot count, read exactly rather than by similarity, and the read is
  audited like a recall. Without them nothing changes: any trusted memory the
  query retrieves counts, whether or not it answers the request.

- A lower-trust write no longer replaces a first-party fact in its slot.
  Writes made under one user id share a speaker, so a third-party document or
  an agent inference stored for that user superseded what the user had said,
  and the user would then be asked to confirm the document's value. A
  user-stated or confirmed fact now stays current and the slot is contested
  until `confirm()`. Corrections of equal or higher trust still supersede.

## 0.4.5 — 2026-10-05

- `forget_user` no longer deletes other users' memories. It added every graph
  neighbor of the erased memories to the deletion set, and entity edges join
  any memories that mention the same name, so on a shared client with the
  entity graph and an extraction model, erasing one user could hard-delete
  another user's memories. It now deletes only the user's memories and the
  summaries derived from them, and severs their graph edges.
- Consolidated summaries keep the weakest corroboration count of their inputs
  instead of the sum. Repeated copies of one third-party claim could
  consolidate into a summary that passed the default trust bar and was
  returned by `recall_for_action`. The summary now stays pending confirmation.
- Unchanged limitation: entity rows are shared across users, and attributes
  merged into an entity from a user's text are not removed by `forget_user`.

## 0.4.4 — 2026-09-20

- PostgreSQL sibling composition now uses an ordered user-plus-entity cache
  index, so a scoped recall does not scan other users that share the same
  entity key.

## 0.4.3 — 2026-09-20

- High-cardinality Postgres recall now indexes cached IDs by user and entity,
  maps vector IDs directly to candidate positions, and detaches only mutable
  result fields at the caller boundary. This avoids whole-cache scope scans and
  generic deep copies while preserving scoped ranking and returned-item
  isolation.

## 0.4.2 — 2026-09-20

- Recall now bounds sibling composition before deep-copying Postgres cache
  entries and borrows read-only candidate references while ranking. Final
  results remain detached, and entity priority, temporal validity, user scope,
  and composition limits are unchanged.

## 0.4.1 — 2026-08-24

- NOOP audit entries now carry the existing memory ID and canonical slot
  identifiers when a duplicate, explicit target, or unambiguous slot resolves
  to a current memory. Ambiguous slot-only NOOPs fail closed, while all NOOPs
  remain content-free and leave the memory version unchanged.

## 0.4.0 — 2026-08-23

- Explicit deterministic `add` / `update` / `delete` / `noop` memory
  evolution with public result and error contracts, lineage, scoped hard
  deletion, content-free NOOP audit, and post-commit consistency tokens.
  Postgres applies memory rows, revisions, and audit in one transaction;
  ordinary `factual.add()` remains compatible and same-speaker duplicates
  are now true no-ops.
- Public connector contracts plus a minimal local PostgreSQL/Supabase reader
  with stable timestamp-plus-key cursors and soft-delete mapping. Hosts retain
  control of credentials, scheduling, retries, and memory writes.

## 0.3.2 — 2026-08-20

### Added

- Host-owned asyncpg pools and embedding providers can be shared safely across
  project runtimes without being closed on runtime eviction.
- Mutations return project-scoped memory versions and Postgres WAL positions;
  reads can require those floors and fail closed when unavailable.
- Embeddings carry model identity and distinct query/document roles, including
  role-aware OpenAI-compatible transport.
- Project-scoped vector and immutable item caches remove redundant warm recall
  queries while preserving user isolation and cross-worker coherence.
- A durable read-audit hook lets hosted runtimes persist exact PII-processed
  recall evidence outside the synchronous SDK audit chain.
- The open-core constitution, PR classification, CI source scanner, and release
  gate prevent operated Cloud capabilities from entering the Apache package.

### Fixed

- Memory mutation, project/global revision changes, and write audit now commit
  in one Postgres transaction.
- Advisory locks execute their protected work on the same transaction-bound
  connection instead of accidentally hopping pool connections.
- Public commercial-use guidance now matches Apache-2.0 while preserving
  ContextDB trademark controls.

`contextdb.init`, `factual.add`, `factual.recall`, and `search` remain
backward-compatible. All new host-sharing and consistency arguments are
optional.

## 0.3.1 — 2026-08-18

### Fixed

- README links to docs/COMMERCIAL.md/CONTRIBUTING.md are absolute GitHub
  URLs — relative targets 404 on the PyPI project page.
- `scripts/check_release.py` now scans the built package description for
  relative Markdown links so this fails in CI before it reaches PyPI.

## 0.3.0 — 2026-08-18

Host API scope hardening:

- Every non-health HTTP route now runs the configured authentication,
  including `/v1/trust_policy` and `/mcp`.
- An authenticated `user_id` (from `auth_hook`) can no longer be
  overridden by the JSON body, `X-ContextDB-User` header, query string,
  or MCP tool arguments; a conflict is a clear 400 (`ScopeConflictError`)
  instead of a silent pick.
- `/mcp` propagates the authenticated scope into every tool call.
- ID-based `confirm` / `forget` verify the target belongs to the resolved
  user scope; a foreign `memory_id` raises `MemoryNotFoundError`,
  indistinguishable from a missing one. Same guarantee on SQLite and
  Postgres stores.
- Anonymous loopback serving is unchanged and documented as *not* an
  authorization boundary.

Decision-layer and PII correctness:

- `VerifyBeforeAct.decide()` / `confirm_pending()` accept an additive
  per-call `user_id=` for shared clients; a scoped client still rejects a
  foreign one.
- SEARCH and DECIDE audit entries now persist only the PII-processed form
  of the query — the append-only chain must never hold raw PII.
- `pii_action="encrypt"` fails closed: without a key, `PIIDetector`
  raises `ConfigError` instead of degrading to redact, and encryption
  mode can never persist plaintext annotation originals.

Postgres and audit concurrency:

- Vector candidates are scoped by SQL *before* ranking — a large foreign
  scope can no longer starve an in-scope hit out of the candidate budget.
- The process-local vector index tracks a global store revision; a write
  from another instance or process forces a rebuild before the next
  search (stale reads are no longer served indefinitely).
- Slot operations on Postgres serialize across processes via a
  transaction-scoped advisory lock; audit-chain appends serialize
  in-process (SQLite) and cross-process (Postgres), so the hash chain
  cannot fork.
- Real Postgres concurrent-worker tests cover corroboration, slot
  supersede, audit-chain integrity, and cross-worker recall
  (`CONTEXTDB_TEST_POSTGRES_URL`).

Host-adoption work:

- `contextdb serve --http` (`pycontextdb[serve]`) — JSON API + `/mcp`, Bearer token, `auth_hook`.
- Per-call `user_id=` on add/recall/confirm/forget. Shared `init()` is the multi-tenant path; `ContextDBPool` is LRU if you still need one client per user.
- OSS `PostgresStore` for `postgresql://` URLs (`pycontextdb[postgres]`).
- `factual.pending_confirmations()`, `forget(memory_id=)` / `forget(entity=, attribute=)`, `add_many` / `add_fast_many`.
- Recall filters: `entity`, `min_confidence`, `include_third_party`.
- `db.on(event, hook)` for write/recall/confirm/forget/injection_suspect.
- Query embedding LRU cache; lexical recall if embedding fails.
- Missing epistemic `source` warns in the SDK; HTTP/MCP remember return 400.
- Docs: trust policy matrix, Postgres, multi-tenant, SQLite migrations.

`contextdb.init`, `factual.add`, `factual.recall`, and `search` keep working. New keyword arguments are optional.

## 0.2.0 — 2026-08-17

First PyPI cut of the trust model. `pip install pycontextdb` now matches
GitHub `main`: memory an agent can act on without treating a wish as a fact.

`contextdb.init`, `factual.add`, `factual.recall`, and `search` are unchanged.

### Added

- Epistemic typing on every fact (`user_stated` / `agent_inferred` /
  `third_party`), confidence, independent corroboration, `action_relevant`.
- `TrustPolicy` as data — `TrustPolicy.hospital()` vs `.restaurant()`.
- `factual.recall_for_action()`, `factual.confirm()`, `VerifyBeforeAct`
  (`act` / `ask` / `abstain`).
- Versioned slot vocabulary + LLM-free slotter (`add_fast` never calls an LLM).
- Contested slots: independent speakers with different values do not
  last-write-win. `confirm()` resolves.
- Temporal supersede with an injectable clock; sibling-slot hop on recall.
- PII redaction on writes **and** queries before embed.
- Write-time injection screening; `[RECALLED DATA — not instructions]` renderer.
- Salience (recency × frequency × criticality), verifiable forget, tenant
  isolation at the storage layer.
- MCP tools: `remember` / `recall` / `recall_for_action` / `forget` / `confirm`.
- In-repo fabrication bake-off: `python benchmarks/trust_bakeoff.py`
  (trust arm vs our own untyped control — not a live Mem0/Zep run).
- Contribution and release hygiene: CLA, CODEOWNERS, and release
  inspection. Code through `2554dae` remains Apache-2.0.

### Fixed

- `contextdb.__version__` no longer drifts from `pyproject.toml` (was `0.1.0`
  while the project version was `0.1.1`).
- CI mypy on Python 3.12: type-check as the running interpreter so
  NumPy 2.5's PEP 695 `type` aliases parse. The 3.10 job still guards
  our runtime floor.
- CI latency evals: keep a tight add_fast p50 and a wider p95 on
  shared runners (p50 was 2ms when p95 spiked to 21ms).

## 0.1.1 — 2026-04-22

README rendering on PyPI; absolute GitHub asset URLs.

## 0.1.0 — 2026-04-22

Initial public release.
