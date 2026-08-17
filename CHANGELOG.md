# Changelog

## Unreleased

Host-adoption work from serviceagent integration:

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
- Open-core governance: [OPEN_CORE.md](OPEN_CORE.md),
  [COMMERCIAL.md](COMMERCIAL.md), CLA, CODEOWNERS, package-boundary CI,
  and release inspection. Code through `2554dae` remains Apache-2.0.
  Cloud is a private repo that depends on this SDK, never the reverse.

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
